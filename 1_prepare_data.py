#!/usr/bin/env python3
import argparse
import json
import struct
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer


DTYPE = np.int32
DTYPE_CODE = 4  # int32 in Megatron's dtype map


def prepare_dataset(
    input_file: str,
    output_prefix: str,
    tokenizer_name: str,
    seq_length: int,
    min_doc_tokens: int = 64,
):
    """Convert JSONL to clean Megatron indexed dataset."""
    
    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    eos_token_id = tokenizer.eos_token_id or 0
    
    # Collect all tokens from documents
    print(f"Reading and tokenizing {input_file}...")
    all_tokens = []
    docs_processed = 0
    docs_skipped = 0
    
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            try:
                doc = json.loads(line)
                text = doc.get("text", "")
            except json.JSONDecodeError:
                docs_skipped += 1
                continue
            
            if not text or len(text.strip()) < 10:
                docs_skipped += 1
                continue
            
            tokens = tokenizer.encode(text, add_special_tokens=False)
            
            if len(tokens) < min_doc_tokens:
                docs_skipped += 1
                continue
            
            all_tokens.extend(tokens)
            all_tokens.append(eos_token_id)
            docs_processed += 1
            
            if docs_processed % 100000 == 0:
                print(f"  {docs_processed:,} docs, {len(all_tokens):,} tokens...")
    
    print(f"\nTokenization complete:")
    print(f"  Documents processed: {docs_processed:,}")
    print(f"  Documents skipped: {docs_skipped:,}")
    print(f"  Total tokens: {len(all_tokens):,}")
    
    # Chunk into fixed-length sequences
    print(f"\nChunking into {seq_length}-token sequences...")
    num_sequences = len(all_tokens) // seq_length
    
    if num_sequences == 0:
        raise ValueError(f"Not enough tokens ({len(all_tokens)}) for even one sequence of {seq_length}")
    
    # Trim to exact multiple of seq_length
    all_tokens = all_tokens[:num_sequences * seq_length]
    tokens_array = np.array(all_tokens, dtype=DTYPE)
    
    print(f"  Created {num_sequences:,} sequences")
    print(f"  Tokens used: {len(tokens_array):,}")
    print(f"  Tokens discarded: {len(all_tokens) - len(tokens_array):,}")
    
    # Write binary file
    output_path = Path(output_prefix)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    bin_path = Path(f"{output_prefix}.bin")
    idx_path = Path(f"{output_prefix}.idx")
    
    print(f"\nWriting {bin_path}...")
    with open(bin_path, "wb") as f:
        tokens_array.tofile(f)
    
    bin_size = bin_path.stat().st_size
    bytes_per_seq = seq_length * DTYPE.itemsize
    
    # Write index file
    print(f"Writing {idx_path}...")
    with open(idx_path, "wb") as f:
        # Header
        f.write(b'MMIDIDX\x00\x00')           # Magic (9 bytes)
        f.write(struct.pack('<Q', 1))          # Version (8 bytes)
        f.write(struct.pack('<B', DTYPE_CODE)) # dtype code (1 byte)
        f.write(struct.pack('<Q', num_sequences))  # num_docs (8 bytes)
        f.write(struct.pack('<Q', num_sequences))  # num_seqs (8 bytes)
        
        # Document indices (1:1 mapping - each sequence is its own document)
        doc_idx = np.arange(num_sequences, dtype=np.int64)
        f.write(doc_idx.tobytes())
        
        # Sequence pointers (in BYTES - no ambiguity)
        pointers = np.arange(num_sequences, dtype=np.int64) * bytes_per_seq
        f.write(pointers.tobytes())
        
        # Sequence lengths (in ELEMENTS - no ambiguity)
        lengths = np.full(num_sequences, seq_length, dtype=np.int32)
        f.write(lengths.tobytes())
    
    # Verify
    print(f"\nVerifying dataset...")
    verify_dataset(output_prefix, seq_length)
    
    print(f"\nDone!")
    print(f"  Binary file: {bin_path} ({bin_size / 1024 / 1024:.1f} MB)")
    print(f"  Index file: {idx_path} ({idx_path.stat().st_size / 1024:.1f} KB)")
    print(f"  Sequences: {num_sequences:,}")
    print(f"  Seq length: {seq_length}")
    print(f"\nDataset ready for training at: {output_prefix}")


def verify_dataset(path: str, expected_seq_length: int):
    """Verify the created dataset is valid."""
    bin_path = Path(f"{path}.bin")
    idx_path = Path(f"{path}.idx")
    
    with open(idx_path, 'rb') as f:
        magic = f.read(9)
        assert magic == b'MMIDIDX\x00\x00', f"Invalid magic: {magic}"
        
        version = struct.unpack('<Q', f.read(8))[0]
        assert version == 1, f"Invalid version: {version}"
        
        dtype_code = struct.unpack('<B', f.read(1))[0]
        assert dtype_code == DTYPE_CODE, f"Unexpected dtype: {dtype_code}"
        
        num_docs = struct.unpack('<Q', f.read(8))[0]
        num_seqs = struct.unpack('<Q', f.read(8))[0]
        assert num_docs == num_seqs, f"num_docs ({num_docs}) != num_seqs ({num_seqs})"
        
        # Skip doc_idx
        f.seek(9 + 8 + 1 + 8 + 8 + num_docs * 8)
        
        # Read pointers and lengths
        pointers = np.frombuffer(f.read(num_seqs * 8), dtype=np.int64)
        lengths = np.frombuffer(f.read(num_seqs * 4), dtype=np.int32)
    
    bin_size = bin_path.stat().st_size
    bytes_per_element = DTYPE.itemsize
    bytes_per_seq = expected_seq_length * bytes_per_element
    
    # Verify all lengths are correct
    assert np.all(lengths == expected_seq_length), "Not all sequences have expected length"
    
    # Verify pointers are valid
    assert np.all(pointers >= 0), "Negative pointers found"
    assert np.all(pointers < bin_size), "Out-of-bounds pointers found"
    
    # Verify alignment
    assert np.all(pointers % bytes_per_element == 0), "Unaligned pointers found"
    
    # Verify sequential layout
    expected_pointers = np.arange(num_seqs, dtype=np.int64) * bytes_per_seq
    assert np.array_equal(pointers, expected_pointers), "Non-sequential pointers"
    
    # Verify binary file size
    expected_bin_size = num_seqs * bytes_per_seq
    assert bin_size == expected_bin_size, f"Binary size {bin_size} != expected {expected_bin_size}"
    
    # Spot check: read first and last sequence
    with open(bin_path, 'rb') as f:
        # First sequence
        first_seq = np.frombuffer(f.read(bytes_per_seq), dtype=DTYPE)
        assert len(first_seq) == expected_seq_length, "First sequence wrong length"
        assert np.all(first_seq >= 0), "Negative tokens in first sequence"
        
        # Last sequence
        f.seek((num_seqs - 1) * bytes_per_seq)
        last_seq = np.frombuffer(f.read(bytes_per_seq), dtype=DTYPE)
        assert len(last_seq) == expected_seq_length, "Last sequence wrong length"
        assert np.all(last_seq >= 0), "Negative tokens in last sequence"
    
    print(f"  All {num_seqs:,} sequences verified OK")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare clean Megatron-format dataset from JSONL"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="/data/allenai-c4-1m-raw.jsonl",
        help="Input JSONL file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/data/allenai-c4-1m",
        help="Output prefix (creates .bin and .idx files)",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="meta-llama/Llama-3.1-8B",
        help="HuggingFace tokenizer to use",
    )
    parser.add_argument(
        "--seq-length",
        type=int,
        default=2048,
        help="Sequence length (default: 2048)",
    )
    parser.add_argument(
        "--min-doc-tokens",
        type=int,
        default=64,
        help="Minimum tokens per document (default: 64)",
    )
    
    args = parser.parse_args()
    
    prepare_dataset(
        input_file=args.input,
        output_prefix=args.output,
        tokenizer_name=args.tokenizer,
        seq_length=args.seq_length,
        min_doc_tokens=args.min_doc_tokens,
    )


if __name__ == "__main__":
    main()
