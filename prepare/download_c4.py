#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

from datasets import load_dataset


def download_c4(
    output_dir: Path,
    subset: str = "en",
    split: str = "train",
    max_samples: int = None,
    streaming: bool = True,
    hf_token: str = None
):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("[Info] Loading C4 dataset from HuggingFace...")
    print("       URL: https://huggingface.co/datasets/allenai/c4")
    print(f"       Subset: {subset}, Split: {split}")
    
    dataset = load_dataset(
        "allenai/c4",
        subset,
        split=split,
        streaming=streaming,
        token=hf_token,
        trust_remote_code=True
    )
    
    if streaming:
        print("[Info] Dataset loaded in streaming mode")
        
        if max_samples:
            output_file = output_dir / f"c4_{subset}_{split}_{max_samples}.jsonl"
            print(f"[Info] Saving {max_samples} samples to {output_file}...")
            
            samples = []
            for i, sample in enumerate(dataset):
                if i >= max_samples:
                    break
                samples.append(sample)
            
            import json
            with open(output_file, "w") as f:
                for sample in samples:
                    f.write(json.dumps(sample) + "\n")
            
            print(f"[Info] Saved {len(samples)} samples to {output_file}")
        else:
            print("[Info] Streaming mode - use dataset directly in training")
            print("[Info] Example usage:")
            print("       from datasets import load_dataset")
            print("       ds = load_dataset('allenai/c4', 'en', split='train', streaming=True)")
            print("       for sample in ds:")
            print("           print(sample['text'])")
    else:
        output_file = output_dir / f"c4_{subset}_{split}.jsonl"
        print(f"[Info] Dataset loaded: {len(dataset)} samples")
        print(f"[Info] Saving to {output_file}...")
        
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        
        dataset.to_json(str(output_file))
        print(f"[Info] Saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Download C4 dataset from HuggingFace"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="./data/c4",
        help="Directory to save the dataset (default: ./data/c4)"
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="en",
        help="Dataset subset (default: en)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "validation"],
        help="Dataset split (default: train)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples to download (default: all)"
    )
    parser.add_argument(
        "--no-streaming",
        action="store_true",
        help="Disable streaming mode (downloads full dataset)"
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=os.environ.get("HF_TOKEN"),
        help="HuggingFace token (default: from HF_TOKEN env var)"
    )
    args = parser.parse_args()
    
    download_c4(
        output_dir=Path(args.output_dir),
        subset=args.subset,
        split=args.split,
        max_samples=args.max_samples,
        streaming=not args.no_streaming,
        hf_token=args.hf_token
    )


if __name__ == "__main__":
    main()
