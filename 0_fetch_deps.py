#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from datasets import load_dataset


MODELS = {
    "meta-llama/Llama-3.1-8B": "meta-llama--llama-31-8b",
    "Qwen/Qwen2.5-7B": "qwen--qwen25-7b",
}


def fetch_tokenizers(output_dir: str):
    """Download and cache tokenizers locally."""
    from transformers import AutoTokenizer
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Fetching tokenizers to {output_path}...")
    
    for hf_name, local_name in MODELS.items():
        local_path = output_path / local_name
        tokenizer_marker = local_path / "tokenizer_config.json"
        
        if tokenizer_marker.exists():
            print(f"  {hf_name}: tokenizer already cached")
            continue
        
        print(f"  {hf_name}: downloading tokenizer...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(hf_name, trust_remote_code=True)
            tokenizer.save_pretrained(local_path)
            print(f"  {hf_name}: saved to {local_path}")
        except Exception as e:
            print(f"  {hf_name}: FAILED - {e}")
    
    print(f"Tokenizers cached in {output_path}")
    return output_path


def fetch_c4(num_samples: int, output_file: str):
    """Fetch C4 dataset and save as raw JSONL."""
    
    print(f"Fetching allenai/c4 (streaming {num_samples:,} samples)...")
    
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
    
    print(f"Writing to {output_file}...")
    with open(output_path, "w", encoding="utf-8") as f:
        for example in dataset:
            text = example.get("text", "")
            json.dump({"text": text}, f, ensure_ascii=False)
            f.write("\n")
            saved += 1
            
            if saved % 100000 == 0:
                print(f"  {saved:,} / {num_samples:,} samples...")
            
            if saved >= num_samples:
                break
    
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    
    print(f"\nDone!")
    print(f"  Saved: {saved:,} samples")
    print(f"  File size: {file_size_mb:.1f} MB")
    print(f"  Output: {output_file}")
    print(f"\nNext step: python 1_clean_data.py")


def main():
    parser = argparse.ArgumentParser(description="Fetch AllenAI C4 dataset and tokenizers")
    parser.add_argument(
        "--samples",
        type=int,
        default=500_000,
        help="Number of samples to fetch (default: 500K)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/data/tprimat/allenai-c4-500k-raw.jsonl",
        help="Output JSONL file path",
    )
    parser.add_argument(
        "--skip-tokenizers",
        action="store_true",
        help="Skip downloading tokenizers",
    )
    
    args = parser.parse_args()
    
    output_dir = str(Path(args.output).parent)
    
    if not args.skip_tokenizers:
        fetch_tokenizers(output_dir)
        print()
    
    fetch_c4(args.samples, args.output)


if __name__ == "__main__":
    main()
