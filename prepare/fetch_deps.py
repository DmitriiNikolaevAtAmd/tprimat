#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from datasets import load_dataset

os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")

DATA_DIR = os.environ.get("DATA_DIR", "/data")

MODELS = {
    "meta-llama/Llama-3.1-8B": "meta-llama-llama-31-8b",
    "Qwen/Qwen2.5-7B": "qwen-qwen25-7b",
}


def fetch_tokenizers(output_dir: str):
    from transformers import AutoTokenizer
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for hf_name, local_name in MODELS.items():
        local_path = output_path / local_name
        tokenizer_marker = local_path / "tokenizer_config.json"
        
        if tokenizer_marker.exists():
            continue
        
        tokenizer = AutoTokenizer.from_pretrained(hf_name, trust_remote_code=True)
        tokenizer.save_pretrained(local_path)
    
    return output_path


def fetch_c4(num_samples: int, output_file: str, max_retries: int = 3):
    from datasets import DownloadConfig
    import time
    
    download_config = DownloadConfig(
        num_proc=1,
        max_retries=5,
    )
    
    for attempt in range(max_retries):
        try:
            dataset = load_dataset(
                "allenai/c4",
                "en",
                split="train",
                streaming=True,
                trust_remote_code=True,
                download_config=download_config,
                data_files=["en/c4-train.00000-of-01024.json.gz", "en/c4-train.00001-of-01024.json.gz"],
            )
            break
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 10 * (attempt + 1)
                time.sleep(wait)
            else:
                raise
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    saved = 0
    
    with open(output_path, "w", encoding="utf-8") as f:
        for example in dataset:
            text = example.get("text", "")
            json.dump({"text": text}, f, ensure_ascii=False)
            f.write("\n")
            saved += 1
            
            if saved >= num_samples:
                break


def main():
    parser = argparse.ArgumentParser(description="Fetch AllenAI C4 dataset and tokenizers")
    parser.add_argument(
        "--samples",
        type=int,
        default=int(os.environ.get("DATA_SAMPLES", 100000)),
        help="Number of samples to fetch (default: DATA_SAMPLES env or 100K)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=f"{DATA_DIR}/allenai-c4-raw.jsonl",
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
    
    fetch_c4(args.samples, args.output)


if __name__ == "__main__":
    main()
