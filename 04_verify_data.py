#!/usr/bin/env python3
import argparse
import struct
import random
import sys
import numpy as np
from pathlib import Path
from collections import Counter


DTYPE = np.int32
DTYPE_CODE = 4


def verify_dataset(input_prefix: str, tokenizer_name: str, num_samples: int) -> bool:
    """Verify encoded dataset integrity. Returns True if valid."""
    
    bin_path = Path(f"{input_prefix}.bin")
    idx_path = Path(f"{input_prefix}.idx")
    
    errors = []
    warnings = []
    
    print(f"Verifying {input_prefix}...")
    
    # Check files exist
    if not bin_path.exists():
        print(f"FAIL: Binary file not found: {bin_path}")
        return False
    if not idx_path.exists():
        print(f"FAIL: Index file not found: {idx_path}")
        return False
    
    # Read index file
    print(f"\n[1/5] Checking index file...")
    try:
        with open(idx_path, 'rb') as f:
            magic = f.read(9)
            if magic != b'MMIDIDX\x00\x00':
                errors.append(f"Invalid magic: {magic}")
            
            version = struct.unpack('<Q', f.read(8))[0]
            if version != 1:
                errors.append(f"Unsupported version: {version}")
            
            dtype_code = struct.unpack('<B', f.read(1))[0]
            if dtype_code != DTYPE_CODE:
                warnings.append(f"Unexpected dtype code: {dtype_code} (expected {DTYPE_CODE})")
            
            num_docs = struct.unpack('<Q', f.read(8))[0]
            num_seqs = struct.unpack('<Q', f.read(8))[0]
            
            if num_docs != num_seqs:
                warnings.append(f"num_docs ({num_docs}) != num_seqs ({num_seqs})")
            
            # Skip doc_idx
            f.seek(9 + 8 + 1 + 8 + 8 + num_docs * 8)
            
            pointers = np.frombuffer(f.read(num_seqs * 8), dtype=np.int64)
            lengths = np.frombuffer(f.read(num_seqs * 4), dtype=np.int32)
        
        print(f"  Sequences: {num_seqs:,}")
        print(f"  Seq length: {lengths[0] if len(lengths) > 0 else 'N/A'}")
        
    except Exception as e:
        errors.append(f"Failed to read index: {e}")
        print(f"FAIL: {e}")
        return False
    
    # Check binary file size
    print(f"\n[2/5] Checking binary file...")
    bin_size = bin_path.stat().st_size
    seq_length = int(lengths[0]) if len(lengths) > 0 else 0
    bytes_per_seq = seq_length * DTYPE.itemsize
    expected_size = num_seqs * bytes_per_seq
    
    print(f"  Binary size: {bin_size:,} bytes ({bin_size / 1024 / 1024:.1f} MB)")
    
    if bin_size != expected_size:
        errors.append(f"Binary size {bin_size} != expected {expected_size}")
    
    # Check all lengths are uniform
    if not np.all(lengths == seq_length):
        errors.append(f"Non-uniform sequence lengths found")
    
    # Check pointers are valid
    print(f"\n[3/5] Checking pointers...")
    if np.any(pointers < 0):
        errors.append("Negative pointers found")
    if np.any(pointers >= bin_size):
        errors.append("Out-of-bounds pointers found")
    if np.any(pointers % DTYPE.itemsize != 0):
        errors.append("Unaligned pointers found")
    
    # Check sequential layout
    expected_pointers = np.arange(num_seqs, dtype=np.int64) * bytes_per_seq
    if not np.array_equal(pointers, expected_pointers):
        warnings.append("Non-sequential pointer layout")
    
    print(f"  Pointer range: [{pointers.min():,}, {pointers.max():,}]")
    
    # Load tokenizer for validation
    print(f"\n[4/5] Validating tokens...")
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        vocab_size = tokenizer.vocab_size
        print(f"  Tokenizer: {tokenizer_name}")
        print(f"  Vocab size: {vocab_size:,}")
    except Exception as e:
        warnings.append(f"Could not load tokenizer: {e}")
        vocab_size = None
    
    # Read and validate random samples
    print(f"\n[5/5] Sampling {num_samples} sequences...")
    
    sample_indices = random.sample(range(num_seqs), min(num_samples, num_seqs))
    all_tokens = []
    decode_errors = 0
    
    with open(bin_path, 'rb') as f:
        for i, idx in enumerate(sample_indices):
            offset = int(pointers[idx])
            f.seek(offset)
            data = f.read(bytes_per_seq)
            tokens = np.frombuffer(data, dtype=DTYPE)
            
            # Check token count
            if len(tokens) != seq_length:
                errors.append(f"Sequence {idx}: got {len(tokens)} tokens, expected {seq_length}")
                continue
            
            # Check token range
            if np.any(tokens < 0):
                errors.append(f"Sequence {idx}: negative tokens found")
            
            if vocab_size and np.any(tokens >= vocab_size):
                errors.append(f"Sequence {idx}: tokens exceed vocab size")
            
            all_tokens.extend(tokens.tolist())
            
            # Try decoding
            if tokenizer:
                try:
                    text = tokenizer.decode(tokens[:50], skip_special_tokens=False)
                    if i < 3:
                        preview = text[:80].replace('\n', ' ')
                        print(f"  Sample {idx}: \"{preview}...\"")
                except Exception:
                    decode_errors += 1
    
    if decode_errors > 0:
        warnings.append(f"{decode_errors} sequences failed to decode")
    
    # Token distribution
    if all_tokens:
        token_counts = Counter(all_tokens)
        unique_tokens = len(token_counts)
        most_common = token_counts.most_common(5)
        
        print(f"\n  Unique tokens in sample: {unique_tokens:,}")
        print(f"  Most common: {most_common[:3]}")
        
        if vocab_size:
            coverage = unique_tokens / vocab_size * 100
            print(f"  Vocab coverage: {coverage:.1f}%")
    
    # Report results
    print("\n" + "=" * 50)
    
    if errors:
        print("ERRORS:")
        for e in errors:
            print(f"  - {e}")
    
    if warnings:
        print("WARNINGS:")
        for w in warnings:
            print(f"  - {w}")
    
    if errors:
        print(f"\nFAIL: {len(errors)} error(s) found")
        return False
    else:
        print(f"\nPASS: Dataset verified OK")
        if warnings:
            print(f"  ({len(warnings)} warning(s))")
        return True


def main():
    parser = argparse.ArgumentParser(description="Verify encoded Megatron dataset")
    parser.add_argument(
        "--input",
        type=str,
        default="/data/tprimat/allenai-c4-100k",
        help="Input prefix (.bin and .idx files)",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="meta-llama/Llama-3.1-8B",
        help="HuggingFace tokenizer for validation",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="Number of random sequences to validate (default: 100)",
    )
    
    args = parser.parse_args()
    
    success = verify_dataset(args.input, args.tokenizer, args.samples)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
