#!/usr/bin/env python3
"""
Validate that all expected training outputs exist before running comparisons.

Usage:
    python3 validate_outputs.py [--platform nvd|amd|all] [--results-dir ./output]
    
Exit codes:
    0 - All expected outputs are present
    1 - Some outputs are missing
"""

import argparse
import os
import sys
from pathlib import Path


# Define expected outputs for each platform
EXPECTED_NVD = [
    "train_nvd_deep_llama.json",
    "train_nvd_deep_qwen.json",
    "train_nvd_fsdp_llama.json",
    "train_nvd_fsdp_qwen.json",
    "train_nvd_mega_llama.json",
    "train_nvd_mega_qwen.json",
    "train_nvd_nemo_llama.json",
    "train_nvd_nemo_qwen.json",
    "train_nvd_tran_llama.json",
    "train_nvd_tran_qwen.json",
]

EXPECTED_AMD = [
    "train_amd_prim_llama.json",
    "train_amd_prim_qwen.json",
]


def validate_outputs(results_dir: str, platform: str) -> tuple[list[str], list[str]]:
    """
    Check which expected outputs exist and which are missing.
    
    Returns:
        Tuple of (found_files, missing_files)
    """
    results_path = Path(results_dir)
    
    if platform == "nvd":
        expected = EXPECTED_NVD
    elif platform == "amd":
        expected = EXPECTED_AMD
    elif platform == "all":
        expected = EXPECTED_NVD + EXPECTED_AMD
    else:
        raise ValueError(f"Unknown platform: {platform}")
    
    found = []
    missing = []
    
    for filename in expected:
        filepath = results_path / filename
        if filepath.exists():
            found.append(filename)
        else:
            missing.append(filename)
    
    return found, missing


def main():
    parser = argparse.ArgumentParser(
        description='Validate that all expected training outputs exist'
    )
    default_dir = os.environ.get('OUTPUT_DIR', './output')
    parser.add_argument('--results-dir', default=default_dir,
                       help='Directory containing benchmark JSON files (default: ./output)')
    parser.add_argument('--platform', choices=['nvd', 'amd', 'all'], default='all',
                       help='Platform to validate (default: all)')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Only output missing files, no success messages')
    
    args = parser.parse_args()
    
    results_path = Path(args.results_dir)
    if not results_path.exists():
        print(f"[x] Output directory does not exist: {args.results_dir}")
        return 1
    
    found, missing = validate_outputs(args.results_dir, args.platform)
    
    if not args.quiet:
        print("=" * 70)
        print(f"TRAINING OUTPUT VALIDATION - Platform: {args.platform.upper()}")
        print("=" * 70)
        print(f"\nResults directory: {args.results_dir}")
        print(f"\nFound {len(found)}/{len(found) + len(missing)} expected outputs:")
    
    if found and not args.quiet:
        for f in found:
            print(f"  [+] {f}")
    
    if missing:
        if not args.quiet:
            print(f"\nMissing {len(missing)} outputs:")
        for f in missing:
            print(f"  [x] {f}")
        
        if not args.quiet:
            print("\n" + "-" * 70)
            print("VALIDATION FAILED")
            print("-" * 70)
            print("\nRun the corresponding training scripts to generate missing outputs:")
            
            # Map filenames to training scripts
            for f in missing:
                # train_nvd_nemo_llama.json -> train/nvd_nemo_llama.sh
                parts = f.replace("train_", "").replace(".json", "")
                script = f"train/{parts}.sh"
                print(f"  ./train/{parts}.sh  ->  output/{f}")
        
        return 1
    
    if not args.quiet:
        print("\n" + "-" * 70)
        print("VALIDATION PASSED - All expected outputs are present")
        print("-" * 70)
        print("\nYou can now run the comparison scripts:")
        if args.platform in ("nvd", "all"):
            print("  ./evaluate/compare_nvd.sh")
        if args.platform in ("amd", "all"):
            print("  ./evaluate/compare_amd.sh")
        if args.platform == "all":
            print("  ./evaluate/compare_all.sh")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
