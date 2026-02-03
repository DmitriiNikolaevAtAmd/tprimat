#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

from transformers import AutoTokenizer


# Supported tokenizers with their HuggingFace paths
TOKENIZER_REGISTRY = {
    "llama": "meta-llama/Llama-3.1-8B",
    "qwen": "Qwen/Qwen2.5-7B",
}


def download_tokenizer(
    model_name: str,
    output_dir: Path,
    hf_token: str = None
):
    if model_name.lower() in TOKENIZER_REGISTRY:
        hf_path = TOKENIZER_REGISTRY[model_name.lower()]
        print(f"[Info] Resolved '{model_name}' -> '{hf_path}'")
    else:
        hf_path = model_name
        print(f"[Info] Using HuggingFace path directly: {hf_path}")
    
    tokenizer_name = hf_path.replace("/", "_")
    save_dir = output_dir / tokenizer_name
    
    if save_dir.exists() and (save_dir / "tokenizer.json").exists():
        print(f"[Info] Tokenizer already exists: {save_dir}")
        return save_dir
    
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[Info] Downloading tokenizer from: {hf_path}")
    print(f"       URL: https://huggingface.co/{hf_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        hf_path,
        token=hf_token,
        trust_remote_code=True
    )
    
    print(f"[Info] Saving tokenizer to: {save_dir}")
    tokenizer.save_pretrained(str(save_dir))
    
    print(f"[Info] Tokenizer downloaded successfully")
    print(f"       Vocab size: {tokenizer.vocab_size}")
    print(f"       Location: {save_dir}")
    
    return save_dir


def list_tokenizers():
    print("\nAvailable tokenizers:")
    print("-" * 60)
    for name, path in sorted(TOKENIZER_REGISTRY.items()):
        print(f"  {name:20s} -> {path}")
    print("-" * 60)
    print("\nYou can also use any HuggingFace model path directly.")


def main():
    parser = argparse.ArgumentParser(
        description="Download tokenizer from HuggingFace"
    )
    parser.add_argument(
        "model",
        type=str,
        nargs="?",
        help="Model name (llama, qwen) or HuggingFace path"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="./data/tokenizers",
        help="Directory to save tokenizers (default: ./data/tokenizers)"
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=os.environ.get("HF_TOKEN"),
        help="HuggingFace token (default: from HF_TOKEN env var)"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List all available tokenizers"
    )
    args = parser.parse_args()
    
    if args.list or not args.model:
        list_tokenizers()
        return
    
    download_tokenizer(
        model_name=args.model,
        output_dir=Path(args.output_dir),
        hf_token=args.hf_token
    )


if __name__ == "__main__":
    main()
