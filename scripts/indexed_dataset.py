#!/usr/bin/env python3
"""
Indexed Dataset Loader for Megatron-style datasets
Compatible with datasets created by Megatron's preprocess_data.py
"""
import struct
import numpy as np
import torch
from pathlib import Path


class IndexedDataset:
    """
    Loader for indexed datasets (binary format with .bin and .idx files)
    Used by Megatron-LM and NeMo for efficient data loading
    """
    
    def __init__(self, path):
        self.path = Path(path)
        self.bin_path = Path(str(path) + '.bin')
        self.idx_path = Path(str(path) + '.idx')
        
        if not self.bin_path.exists():
            raise FileNotFoundError(f"Binary file not found: {self.bin_path}")
        if not self.idx_path.exists():
            raise FileNotFoundError(f"Index file not found: {self.idx_path}")
        
        # Read index file
        with open(self.idx_path, 'rb') as f:
            # Read header: magic number, version
            magic = f.read(9)
            assert magic == b'MMIDIDX\x00\x00', "Invalid index file format"
            version = struct.unpack('<Q', f.read(8))[0]
            assert version == 1, f"Unsupported index version: {version}"
            
            # Read dtype code
            dtype_code = struct.unpack('<B', f.read(1))[0]
            self.dtype = self._code_to_dtype(dtype_code)
            
            # Read number of documents and sequences
            self.num_docs = struct.unpack('<Q', f.read(8))[0]
            self.num_seqs = struct.unpack('<Q', f.read(8))[0]
            
            # Read document indices
            self.doc_idx = np.frombuffer(
                f.read(self.num_docs * 8), 
                dtype=np.int64
            )
            
            # Read sequence offsets and lengths
            self.seq_pointers = np.frombuffer(
                f.read(self.num_seqs * 8), 
                dtype=np.int64
            )
            self.seq_lengths = np.frombuffer(
                f.read(self.num_seqs * 4), 
                dtype=np.int32
            )
        
        # Track binary file size for sanity checks
        self.bin_size = self.bin_path.stat().st_size
        
        # Detect pointer/length units to match dataset format
        dtype_size = np.dtype(self.dtype).itemsize
        max_ptr = int(self.seq_pointers.max()) if len(self.seq_pointers) else 0
        max_len = int(self.seq_lengths.max()) if len(self.seq_lengths) else 0
        
        candidates = []
        for ptr_scale in (1, dtype_size):
            for len_scale in (1, dtype_size):
                max_end = max_ptr * ptr_scale + max_len * len_scale
                if max_end <= self.bin_size:
                    candidates.append((max_end, ptr_scale, len_scale))
        
        if candidates:
            _, self._ptr_scale, self._len_scale = max(candidates, key=lambda x: x[0])
        else:
            # Fallback to default Megatron-style (ptr in bytes, len in elements)
            self._ptr_scale, self._len_scale = 1, dtype_size
        
        # Don't open binary file here - will open lazily per process
        self._bin_file = None
        
    def _code_to_dtype(self, code):
        """Convert dtype code to numpy dtype"""
        dtype_map = {
            1: np.uint8,
            2: np.int8,
            3: np.int16,
            4: np.int32,
            5: np.int64,
            6: np.float32,
            7: np.float64,
            8: np.uint16,
        }
        return dtype_map.get(code, np.int64)
    
    @property
    def bin_file(self):
        """Lazy file handle - opens once per process"""
        if self._bin_file is None:
            # Use buffered binary reading with explicit buffer size
            self._bin_file = open(self.bin_path, 'rb', buffering=8192*1024)
        return self._bin_file
    
    def __len__(self):
        return len(self.seq_lengths)
    
    def __getitem__(self, idx):
        """Get sequence at index"""
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range [0, {len(self)})")
        
        # Get pointer and length (convert to Python int to avoid numpy type issues)
        pointer = int(self.seq_pointers[idx])
        length = int(self.seq_lengths[idx])
        
        # Handle empty sequences - return a minimal token sequence
        if length <= 0:
            # Return a single EOS token for empty sequences
            return torch.tensor([0], dtype=torch.long)
        
        # Validate pointer
        if pointer < 0:
            raise ValueError(f"Invalid pointer {pointer} at index {idx}")
        
        dtype_size = np.dtype(self.dtype).itemsize
        pointer_bytes = pointer * self._ptr_scale
        bytes_to_read = length * self._len_scale
        
        # Read data using pread to avoid seek issues in multi-threaded/multi-process context
        import os
        try:
            fd = self.bin_file.fileno()
            data = os.pread(fd, bytes_to_read, pointer_bytes)
        except (AttributeError, OSError):
            # Fallback to seek if pread not available or fails
            self.bin_file.seek(pointer_bytes, os.SEEK_SET)
            data = self.bin_file.read(bytes_to_read)
        
        # Verify we read the expected amount
        if len(data) != bytes_to_read:
            raise IOError(f"Expected to read {bytes_to_read} bytes but got {len(data)}")
        
        # Convert to numpy array
        if bytes_to_read % dtype_size != 0:
            raise IOError(f"Sequence byte size {bytes_to_read} not aligned to dtype {dtype_size}")
        
        tokens = np.frombuffer(data, dtype=self.dtype)
        return torch.from_numpy(tokens.astype(np.int64))
    
    def __del__(self):
        if hasattr(self, '_bin_file') and self._bin_file is not None:
            self._bin_file.close()
    
    def __getstate__(self):
        """Handle pickling for multiprocessing"""
        state = self.__dict__.copy()
        # Don't pickle the file handle
        state['_bin_file'] = None
        return state
    
    def __setstate__(self, state):
        """Handle unpickling for multiprocessing"""
        self.__dict__.update(state)
        # File will be opened lazily in the new process


def check_indexed_dataset_exists(path):
    """Check if indexed dataset files exist"""
    bin_path = Path(str(path) + '.bin')
    idx_path = Path(str(path) + '.idx')
    return bin_path.exists() and idx_path.exists()
