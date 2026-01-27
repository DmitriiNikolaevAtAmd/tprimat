#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from datasets import load_dataset


def download_c4(num_samples: int, output_file: str, min_length: int = 100):
    """Download C4 dataset and save as JSONL."""
    
    print(f"Downloading allenai/c4 (streaming {num_samples:,} samples)...")
    
    dataset = load_dataset(
        "allenai/c4",
        "en",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    saved = 0
    skipped = 0
    
    print(f"Writing to {output_file}...")
    with open(output_path, "w", encoding="utf-8") as f:
        for i, example in enumerate(dataset):
            text = example.get("text", "")
            
            # Skip short or empty texts
            if not text or len(text) < min_length:
                skipped += 1
                continue
            
            json.dump({"text": text}, f, ensure_ascii=False)
            f.write("\n")
            saved += 1
            
            if saved % 100000 == 0:
                print(f"  {saved:,} / {num_samples:,} samples saved...")
            
            if saved >= num_samples:
                break
    
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    
    print(f"\nDone!")
    print(f"  Saved: {saved:,} samples")
    print(f"  Skipped: {skipped:,} (too short)")
    print(f"  File size: {file_size_mb:.1f} MB")
    print(f"  Output: {output_file}")
    
    print(f"\nNext step: Tokenize with 1.tokenize_data.py")


def main():
    parser = argparse.ArgumentParser(
        description="Download AllenAI C4 dataset for LLM pretraining"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1_000_000,
        help="Number of samples to download (default: 1M)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/data/allenai-c4-1m-raw.jsonl",
        help="Output JSONL file path",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=100,
        help="Minimum text length in characters (default: 100)",
    )
    
    args = parser.parse_args()
    download_c4(args.samples, args.output, args.min_length)


if __name__ == "__main__":
    main()
