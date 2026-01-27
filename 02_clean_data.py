#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def clean_data(input_file: str, output_file: str, min_chars: int, min_words: int):
    """Clean raw JSONL: remove empty, short, and malformed entries."""
    
    print(f"Cleaning {input_file}...")
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    total = 0
    kept = 0
    skipped_empty = 0
    skipped_short_chars = 0
    skipped_short_words = 0
    skipped_malformed = 0
    
    with open(input_file, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        
        for line in fin:
            total += 1
            line = line.strip()
            
            if not line:
                skipped_empty += 1
                continue
            
            try:
                doc = json.loads(line)
                text = doc.get("text", "")
            except json.JSONDecodeError:
                skipped_malformed += 1
                continue
            
            if not text or not text.strip():
                skipped_empty += 1
                continue
            
            text = text.strip()
            
            if len(text) < min_chars:
                skipped_short_chars += 1
                continue
            
            word_count = len(text.split())
            if word_count < min_words:
                skipped_short_words += 1
                continue
            
            json.dump({"text": text}, fout, ensure_ascii=False)
            fout.write("\n")
            kept += 1
            
            if total % 100000 == 0:
                print(f"  {total:,} processed, {kept:,} kept...")
    
    skipped_total = total - kept
    
    print(f"\nDone!")
    print(f"  Input: {total:,} documents")
    print(f"  Output: {kept:,} documents ({100*kept/total:.1f}%)")
    print(f"  Skipped: {skipped_total:,}")
    print(f"    - Empty: {skipped_empty:,}")
    print(f"    - Short (<{min_chars} chars): {skipped_short_chars:,}")
    print(f"    - Short (<{min_words} words): {skipped_short_words:,}")
    print(f"    - Malformed JSON: {skipped_malformed:,}")
    print(f"  Output file: {output_file}")
    print(f"\nNext step: python 03_encode_data.py")


def main():
    parser = argparse.ArgumentParser(description="Clean raw JSONL data")
    parser.add_argument(
        "--input",
        type=str,
        default="/data/tprimat/allenai-c4-500k-raw.jsonl",
        help="Input raw JSONL file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/data/tprimat/allenai-c4-500k.jsonl",
        help="Output clean JSONL file",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=100,
        help="Minimum text length in characters (default: 100)",
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=20,
        help="Minimum word count (default: 20)",
    )
    
    args = parser.parse_args()
    clean_data(args.input, args.output, args.min_chars, args.min_words)


if __name__ == "__main__":
    main()
