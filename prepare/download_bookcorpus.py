#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import nltk
from datasets import load_dataset


def download_bookcorpus(output_dir: Path, max_samples: int = None, hf_token: str = None):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "bookcorpus_megatron.json"
    
    if output_file.exists():
        print(f"[Info] Dataset already exists: {output_file}")
        return output_file
    
    print("[Info] Downloading NLTK punkt tokenizer...")
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    
    print("[Info] Loading BookCorpus dataset from HuggingFace (streaming mode)...")
    print("       URL: https://huggingface.co/datasets/bookcorpus")
    
    # Use streaming to avoid downloading the entire dataset
    dataset = load_dataset(
        "bookcorpus",
        split="train",
        streaming=True,
        trust_remote_code=True,
        token=hf_token
    )
    
    import json
    
    if max_samples:
        print(f"[Info] Saving {max_samples} samples to {output_file}...")
        samples = []
        for i, sample in enumerate(dataset):
            if i >= max_samples:
                break
            samples.append(sample)
        
        with open(output_file, "w") as f:
            for sample in samples:
                f.write(json.dumps({"text": sample["text"]}) + "\n")
        
        print(f"[Info] Saved {len(samples)} samples to {output_file}")
    else:
        print("[Info] No max_samples specified, downloading full dataset...")
        # Fall back to non-streaming for full dataset
        dataset = load_dataset(
            "bookcorpus",
            split="train",
            trust_remote_code=True,
            token=hf_token
        )
        print(f"[Info] Dataset loaded: {len(dataset)} samples")
        print(f"[Info] Saving dataset to {output_file}...")
        dataset.to_json(str(output_file))
    
    print(f"[Info] Download completed: {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(
        description="Download BookCorpus dataset from HuggingFace"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="./data/bookcorpus",
        help="Directory to save the dataset (default: ./data/bookcorpus)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=10000,
        help="Maximum samples to download (default: 10000)"
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=os.environ.get("HF_TOKEN"),
        help="HuggingFace token (default: from HF_TOKEN env var)"
    )
    args = parser.parse_args()
    
    download_bookcorpus(
        output_dir=Path(args.output_dir),
        max_samples=args.max_samples,
        hf_token=args.hf_token
    )


if __name__ == "__main__":
    main()
