#!/usr/bin/env python3
import argparse
import json
import struct
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer


DTYPE = np.dtype(np.int32)
DTYPE_CODE = 4  # int32 in Megatron's dtype map


def encode_dataset(
    input_file: str,
    output_prefix: str,
    tokenizer_name: str,
    seq_length: int,
):
    """Tokenize clean JSONL and create Megatron indexed dataset."""
    
    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    eos_token_id = tokenizer.eos_token_id or 0
    
    print(f"Tokenizing {input_file}...")
    all_tokens = []
    docs_processed = 0
    
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            doc = json.loads(line)
            text = doc.get("text", "")
            
            tokens = tokenizer.encode(text, add_special_tokens=False)
            all_tokens.extend(tokens)
            all_tokens.append(eos_token_id)
            docs_processed += 1
            
            if docs_processed % 100000 == 0:
                print(f"  {docs_processed:,} docs, {len(all_tokens):,} tokens...")
    
    print(f"\nTokenization complete:")
    print(f"  Documents: {docs_processed:,}")
    print(f"  Total tokens: {len(all_tokens):,}")
    
    # Chunk into fixed-length sequences
    print(f"\nChunking into {seq_length}-token sequences...")
    num_sequences = len(all_tokens) // seq_length
    
    if num_sequences == 0:
        raise ValueError(f"Not enough tokens ({len(all_tokens)}) for seq_length {seq_length}")
    
    tokens_used = num_sequences * seq_length
    tokens_array = np.array(all_tokens[:tokens_used], dtype=DTYPE)
    
    print(f"  Sequences: {num_sequences:,}")
    print(f"  Tokens used: {tokens_used:,}")
    print(f"  Tokens discarded: {len(all_tokens) - tokens_used:,}")
    
    # Write binary file
    output_path = Path(output_prefix)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    bin_path = Path(f"{output_prefix}.bin")
    idx_path = Path(f"{output_prefix}.idx")
    
    print(f"\nWriting {bin_path}...")
    with open(bin_path, "wb") as f:
        tokens_array.tofile(f)
    
    bytes_per_seq = seq_length * DTYPE.itemsize
    
    # Write index file
    print(f"Writing {idx_path}...")
    with open(idx_path, "wb") as f:
        f.write(b'MMIDIDX\x00\x00')
        f.write(struct.pack('<Q', 1))
        f.write(struct.pack('<B', DTYPE_CODE))
        f.write(struct.pack('<Q', num_sequences))
        f.write(struct.pack('<Q', num_sequences))
        
        doc_idx = np.arange(num_sequences, dtype=np.int64)
        f.write(doc_idx.tobytes())
        
        pointers = np.arange(num_sequences, dtype=np.int64) * bytes_per_seq
        f.write(pointers.tobytes())
        
        lengths = np.full(num_sequences, seq_length, dtype=np.int32)
        f.write(lengths.tobytes())
    
    bin_size = bin_path.stat().st_size
    idx_size = idx_path.stat().st_size
    
    print(f"\nDone!")
    print(f"  Binary: {bin_path} ({bin_size / 1024 / 1024:.1f} MB)")
    print(f"  Index: {idx_path} ({idx_size / 1024:.1f} KB)")
    print(f"  Sequences: {num_sequences:,} x {seq_length} tokens")
    print(f"\nDataset ready: {output_prefix}")
    
    return tokens_array, num_sequences


def write_nemo_index(output_prefix: str, tokens_array: np.ndarray, seq_length: int, num_sequences: int):
    """Write NeMo/Megatron-Core format dataset.
    
    Megatron-Core IndexedDataset index format (from megatron/core/datasets/indexed_dataset.py):
    - Header: magic (9 bytes) + version (8) + dtype (1) + num_sequences (8) + num_documents (8)
    - Data: sequence_lengths (N x int32) + sequence_pointers (N x int64) + document_indices ((D+1) x int64)
    
    Critical assertion in _IndexReader: sequence_lengths.shape[0] == document_indices[-1]
    This means: len(sequence_lengths) must equal the last value of document_indices.
    """
    
    bin_path = Path(f"{output_prefix}.bin")
    idx_path = Path(f"{output_prefix}.idx")
    bytes_per_seq = seq_length * DTYPE.itemsize
    
    print(f"\nWriting NeMo/Megatron-Core format: {output_prefix}")
    
    # Binary file (same token data)
    print(f"  Writing {bin_path}...")
    with open(bin_path, "wb") as f:
        tokens_array.tofile(f)
    
    # Each sequence = 1 document in this simple case
    num_documents = num_sequences
    
    # Prepare arrays
    # 1. Sequence lengths: all same length
    sequence_lengths = np.full(num_sequences, seq_length, dtype=np.int32)
    
    # 2. Sequence pointers: byte offsets in binary file
    sequence_pointers = np.arange(num_sequences, dtype=np.int64) * bytes_per_seq
    
    # 3. Document indices: maps document -> sequence range
    # For N documents with 1 sequence each: [0, 1, 2, ..., N]
    # document_indices[i] = first sequence of document i
    # document_indices[-1] = total number of sequences (CRITICAL for assertion)
    document_indices = np.arange(num_documents + 1, dtype=np.int64)
    
    # Verify the critical assertion before writing
    assert sequence_lengths.shape[0] == document_indices[-1], \
        f"Pre-write assertion failed: {sequence_lengths.shape[0]} != {document_indices[-1]}"
    
    print(f"  Writing {idx_path}...")
    print(f"    num_sequences={num_sequences}, num_documents={num_documents}")
    print(f"    sequence_lengths: {sequence_lengths.shape}, dtype={sequence_lengths.dtype}")
    print(f"    sequence_pointers: {sequence_pointers.shape}, dtype={sequence_pointers.dtype}")
    print(f"    document_indices: {document_indices.shape}, dtype={document_indices.dtype}")
    print(f"    Assertion check: len(lengths)={len(sequence_lengths)} == doc_indices[-1]={document_indices[-1]}")
    
    with open(idx_path, "wb") as f:
        # Header
        f.write(b'MMIDIDX\x00\x00')                    # 9 bytes: magic
        f.write(struct.pack('<Q', 1))                  # 8 bytes: version
        f.write(struct.pack('<B', DTYPE_CODE))         # 1 byte: dtype code
        f.write(struct.pack('<Q', num_sequences))      # 8 bytes: number of sequences
        f.write(struct.pack('<Q', num_documents))      # 8 bytes: number of documents
        
        # Data arrays (order is critical!)
        f.write(sequence_lengths.tobytes())            # N x int32
        f.write(sequence_pointers.tobytes())           # N x int64
        f.write(document_indices.tobytes())            # (D+1) x int64
    
    # Verify file sizes
    bin_size = bin_path.stat().st_size
    idx_size = idx_path.stat().st_size
    
    expected_idx_size = 9 + 8 + 1 + 8 + 8 + (num_sequences * 4) + (num_sequences * 8) + ((num_documents + 1) * 8)
    expected_bin_size = num_sequences * bytes_per_seq
    
    print(f"  Binary: {bin_size:,} bytes (expected {expected_bin_size:,})")
    print(f"  Index: {idx_size:,} bytes (expected {expected_idx_size:,})")
    
    if bin_size != expected_bin_size:
        print(f"  WARNING: Binary size mismatch!")
    if idx_size != expected_idx_size:
        print(f"  WARNING: Index size mismatch!")


TOKENIZERS = {
    "llama": "meta-llama/Llama-3.1-8B",
    "qwen": "Qwen/Qwen2.5-7B",
}


def main():
    # Default data directory (consistent with 01_fetch_deps.py and 02_clean_data.py)
    default_data_dir = "/data/tprimat"
    
    parser = argparse.ArgumentParser(description="Encode JSONL to Megatron format")
    parser.add_argument(
        "--input",
        type=str,
        default=f"{default_data_dir}/allenai-c4.jsonl",
        help="Input clean JSONL file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=default_data_dir,
        help="Output directory for encoded datasets",
    )
    parser.add_argument(
        "--seq-length",
        type=int,
        default=2048,
        help="Sequence length (default: 2048)",
    )
    
    args = parser.parse_args()
    
    # Encode for each tokenizer
    for model_name, tokenizer_name in TOKENIZERS.items():
        output_prefix = f"{args.output_dir}/allenai-c4-{model_name}-mega"
        print(f"\n{'='*60}")
        print(f"Encoding for {model_name.upper()}")
        print(f"{'='*60}")
        
        # Standard format (for DeepSpeed, FSDP, Transformers, Mega)
        tokens_array, num_sequences = encode_dataset(
            args.input, output_prefix, tokenizer_name, args.seq_length
        )
        
        # NeMo format (different index structure)
        nemo_prefix = f"{args.output_dir}/allenai-c4-{model_name}-nemo"
        write_nemo_index(nemo_prefix, tokens_array, args.seq_length, num_sequences)


if __name__ == "__main__":
    main()
