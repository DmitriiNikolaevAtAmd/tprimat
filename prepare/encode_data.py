#!/usr/bin/env python3
import argparse
import json
import os
import struct
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer

DATA_DIR = os.environ.get("DATA_DIR", "/data")

DTYPE = np.dtype(np.int32)
DTYPE_CODE = 4


def encode_dataset(input_file: str, output_prefix: str, tokenizer_name: str, seq_length: int):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    eos_token_id = tokenizer.eos_token_id or 0
    
    all_tokens = []
    
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
    
    num_sequences = len(all_tokens) // seq_length
    
    if num_sequences == 0:
        raise ValueError(f"Not enough tokens ({len(all_tokens)}) for seq_length {seq_length}")
    
    tokens_used = num_sequences * seq_length
    tokens_array = np.array(all_tokens[:tokens_used], dtype=DTYPE)
    
    output_path = Path(output_prefix)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    bin_path = Path(f"{output_prefix}.bin")
    idx_path = Path(f"{output_prefix}.idx")
    
    with open(bin_path, "wb") as f:
        tokens_array.tofile(f)
    
    bytes_per_seq = seq_length * DTYPE.itemsize
    
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
    
    return tokens_array, num_sequences


def write_nemo_index(output_prefix: str, tokens_array: np.ndarray, seq_length: int, num_sequences: int):
    bin_path = Path(f"{output_prefix}.bin")
    idx_path = Path(f"{output_prefix}.idx")
    bytes_per_seq = seq_length * DTYPE.itemsize
    
    with open(bin_path, "wb") as f:
        tokens_array.tofile(f)
    
    num_documents = num_sequences
    sequence_lengths = np.full(num_sequences, seq_length, dtype=np.int32)
    sequence_pointers = np.arange(num_sequences, dtype=np.int64) * bytes_per_seq
    document_indices = np.arange(num_documents + 1, dtype=np.int64)
    
    assert sequence_lengths.shape[0] == document_indices[-1]
    
    with open(idx_path, "wb") as f:
        f.write(b'MMIDIDX\x00\x00')
        f.write(struct.pack('<Q', 1))
        f.write(struct.pack('<B', DTYPE_CODE))
        f.write(struct.pack('<Q', num_sequences))
        f.write(struct.pack('<Q', num_documents))
        # Megatron core expects lengths, pointers, then doc indices.
        f.write(sequence_lengths.tobytes())
        f.write(sequence_pointers.tobytes())
        f.write(document_indices.tobytes())


TOKENIZERS = {
    "llama": "meta-llama/Llama-3.1-8B",
    "qwen": "Qwen/Qwen2.5-7B",
}


def main():
    parser = argparse.ArgumentParser(description="Encode JSONL to Megatron format")
    parser.add_argument(
        "--input",
        type=str,
        default=f"{DATA_DIR}/allenai-c4.jsonl",
        help="Input clean JSONL file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DATA_DIR,
        help="Output directory for encoded datasets",
    )
    parser.add_argument(
        "--seq-length",
        type=int,
        default=2048,
        help="Sequence length (default: 2048)",
    )
    parser.add_argument(
        "--nemo",
        action="store_true",
        help="Generate NeMo format",
    )
    parser.add_argument(
        "--mega",
        action="store_true",
        help="Generate Mega format",
    )
    
    args = parser.parse_args()
    
    if not args.nemo and not args.mega:
        # Default to both if none specified
        args.nemo = True
        args.mega = True
    
    for model_name, tokenizer_name in TOKENIZERS.items():
        output_prefix = f"{args.output_dir}/allenai-c4-{model_name}-mega"
        
        # Always need the tokens_array
        tokens_array, num_sequences = encode_dataset(
            args.input, output_prefix, tokenizer_name, args.seq_length
        )
        
        # If not mega, we delete the mega .idx file after generation
        if not args.mega:
            mega_idx = Path(f"{output_prefix}.idx")
            if mega_idx.exists():
                mega_idx.unlink()
        
        # If nemo, generate nemo format
        if args.nemo:
            nemo_prefix = f"{args.output_dir}/allenai-c4-{model_name}-nemo"
            write_nemo_index(nemo_prefix, tokens_array, args.seq_length, num_sequences)


if __name__ == "__main__":
    main()
