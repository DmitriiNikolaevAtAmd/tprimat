#!/usr/bin/env python3
"""
Script to verify if indexed datasets exist and can be loaded
"""
import sys
from pathlib import Path

def check_data_files():
    """Check if data files exist"""
    dataset_path = "/data/tprimat/allenai-c4-500k"
    
    print("=" * 80)
    print("DATA LOADING VERIFICATION")
    print("=" * 80)
    
    bin_path = Path(dataset_path + '.bin')
    idx_path = Path(dataset_path + '.idx')
    
    print(f"\nChecking for indexed dataset files:")
    print(f"  Binary file: {bin_path}")
    print(f"    Exists: {bin_path.exists()}")
    if bin_path.exists():
        size_mb = bin_path.stat().st_size / (1024 * 1024)
        print(f"    Size: {size_mb:.2f} MB")
    
    print(f"\n  Index file: {idx_path}")
    print(f"    Exists: {idx_path.exists()}")
    if idx_path.exists():
        size_kb = idx_path.stat().st_size / 1024
        print(f"    Size: {size_kb:.2f} KB")
    
    if bin_path.exists() and idx_path.exists():
        print("\n✓ Dataset files found! Attempting to load...")
        try:
            from indexed_dataset import IndexedDataset
            dataset = IndexedDataset(dataset_path)
            print(f"✓ Successfully loaded dataset")
            print(f"  Number of sequences: {len(dataset)}")
            
            # Try loading first sample
            sample = dataset[0]
            print(f"  First sequence length: {len(sample)} tokens")
            print(f"  First 10 tokens: {sample[:10].tolist()}")
            
            print("\n✓ All frameworks will now use REAL DATA for training!")
            return True
        except Exception as e:
            print(f"\n✗ Failed to load dataset: {e}")
            print("  Frameworks will fall back to synthetic data")
            import traceback
            traceback.print_exc()
            return False
    else:
        print("\n✗ Dataset files NOT found")
        print("  All frameworks will use SYNTHETIC DATA (random tokens)")
        print("\n  To use real data:")
        print("    1. Prepare text data")
        print("    2. Use Megatron's preprocess_data.py to create indexed dataset")
        print("    3. Place files at: /data/tprimat/allenai-c4-500k.{bin,idx}")
        return False


if __name__ == "__main__":
    success = check_data_files()
    sys.exit(0 if success else 1)
