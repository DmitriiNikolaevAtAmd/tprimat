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
    """Write NeMo/Megatron format dataset (lengths → pointers → doc_indices)."""
    
    bin_path = Path(f"{output_prefix}.bin")
    idx_path = Path(f"{output_prefix}.idx")
    bytes_per_seq = seq_length * DTYPE.itemsize
    
    print(f"\nWriting NeMo format: {output_prefix}")
    
    # Binary file (same data)
    print(f"  Writing {bin_path}...")
    with open(bin_path, "wb") as f:
        tokens_array.tofile(f)
    
    # Index file with NeMo/Megatron order: lengths → pointers → doc_indices
    # Based on Megatron _IndexReader which reads in this exact order
    print(f"  Writing {idx_path}...")
    with open(idx_path, "wb") as f:
        f.write(b'MMIDIDX\x00\x00')
        f.write(struct.pack('<Q', 1))
        f.write(struct.pack('<B', DTYPE_CODE))
        f.write(struct.pack('<Q', num_sequences))  # num_sequences
        f.write(struct.pack('<Q', num_sequences))  # num_documents
        
        # 1. Sequence lengths (N elements, int32)
        lengths = np.full(num_sequences, seq_length, dtype=np.int32)
        f.write(lengths.tobytes())
        
        # 2. Sequence pointers (N elements, int64)
        pointers = np.arange(num_sequences, dtype=np.int64) * bytes_per_seq
        f.write(pointers.tobytes())
        
        # 3. Document indices (N+1 elements, int64)
        # Critical: doc_indices[-1] must equal num_sequences for NeMo assertion
        doc_indices = np.arange(num_sequences + 1, dtype=np.int64)
        f.write(doc_indices.tobytes())
    
    bin_size = bin_path.stat().st_size
    idx_size = idx_path.stat().st_size
    print(f"  Done: {bin_size / 1024 / 1024:.1f} MB bin, {idx_size / 1024:.1f} KB idx")


TOKENIZERS = {
    "llama": "meta-llama/Llama-3.1-8B",
    "qwen": "Qwen/Qwen2.5-7B",
}


def main():
    parser = argparse.ArgumentParser(description="Encode JSONL to Megatron format")
    parser.add_argument(
        "--input",
        type=str,
        default="/data/tprimat/allenai-c4-100k.jsonl",
        help="Input clean JSONL file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/data/tprimat",
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
        output_prefix = f"{args.output_dir}/allenai-c4-100k-{model_name}-mega"
        print(f"\n{'='*60}")
        print(f"Encoding for {model_name.upper()}")
        print(f"{'='*60}")
        
        # Standard format (for DeepSpeed, FSDP, Transformers, Mega)
        tokens_array, num_sequences = encode_dataset(
            args.input, output_prefix, tokenizer_name, args.seq_length
        )
        
        # NeMo format (different index structure)
        nemo_prefix = f"{args.output_dir}/allenai-c4-100k-{model_name}-nemo"
        write_nemo_index(nemo_prefix, tokens_array, args.seq_length, num_sequences)


if __name__ == "__main__":
    main()
