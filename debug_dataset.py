#!/usr/bin/env python3
"""Debug script to examine indexed dataset structure"""
import struct
import numpy as np
from pathlib import Path

dataset_path = "/data/tprimat/allenai-c4-1m"
idx_path = Path(dataset_path + '.idx')

print("Reading index file...")
with open(idx_path, 'rb') as f:
    # Read header
    magic = f.read(9)
    print(f"Magic: {magic}")
    
    version = struct.unpack('<Q', f.read(8))[0]
    print(f"Version: {version}")
    
    dtype_code = struct.unpack('<B', f.read(1))[0]
    print(f"Dtype code: {dtype_code}")
    
    num_docs = struct.unpack('<Q', f.read(8))[0]
    print(f"Number of documents: {num_docs}")
    
    num_seqs = struct.unpack('<Q', f.read(8))[0]
    print(f"Number of sequences: {num_seqs}")
    
    # Read first 10 document indices
    doc_idx = np.frombuffer(f.read(min(10, num_docs) * 8), dtype=np.int64)
    print(f"\nFirst 10 document indices: {doc_idx}")
    
    # Skip rest of doc indices
    if num_docs > 10:
        f.seek(8 + 1 + 8 + 8 + num_docs * 8, 0)
    
    # Read first 20 sequence pointers and lengths
    seq_pointers = np.frombuffer(f.read(min(20, num_seqs) * 8), dtype=np.int64)
    seq_lengths = np.frombuffer(f.read(min(20, num_seqs) * 4), dtype=np.int32)
    
    print(f"\nFirst 20 sequence pointers:")
    for i in range(min(20, len(seq_pointers))):
        print(f"  Seq {i}: pointer={seq_pointers[i]}, length={seq_lengths[i]}")
