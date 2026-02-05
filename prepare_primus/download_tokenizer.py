#!/usr/bin/env python3
###############################################################################
# Download tokenizer from HuggingFace
# Supports all model families: Llama, DeepSeek, Qwen, Mixtral, etc.
###############################################################################

import argparse
import os
from pathlib import Path

from transformers import AutoTokenizer


# Supported tokenizers with their HuggingFace paths
TOKENIZER_REGISTRY = {
    # Llama 2
    "llama2-7b": "meta-llama/Llama-2-7b-hf",
    "llama2-70b": "meta-llama/Llama-2-70b-hf",
    # Llama 3
    "llama3-8b": "meta-llama/Meta-Llama-3-8B",
    "llama3-70b": "meta-llama/Meta-Llama-3-70B",
    # Llama 3.1
    "llama3.1-8b": "meta-llama/Llama-3.1-8B",
    "llama3.1-70b": "meta-llama/Llama-3.1-70B",
    "llama3.1-405b": "meta-llama/Llama-3.1-405B",
    # Llama 3.3
    "llama3.3-70b": "meta-llama/Llama-3.3-70B-Instruct",
    # Llama 4
    "llama4-scout": "meta-llama/Llama-4-Scout-17B-16E",
    "llama4-maverick": "meta-llama/Llama-4-Maverick-17B-128E",
    # DeepSeek
    "deepseek-v2": "deepseek-ai/DeepSeek-V2",
    "deepseek-v2-lite": "deepseek-ai/DeepSeek-V2-Lite",
    "deepseek-v3": "deepseek-ai/DeepSeek-V3",
    "deepseek-v3.1": "deepseek-ai/DeepSeek-V3.1-Base",
    "deepseek-moe-16b": "deepseek-ai/deepseek-moe-16b-base",
    # Qwen
    "qwen2.5-7b": "Qwen/Qwen2.5-7B",
    "qwen2.5-72b": "Qwen/Qwen2.5-72B",
    "qwen3-0.6b": "Qwen/Qwen3-0.6B",
    "qwen3-1.7b": "Qwen/Qwen3-1.7B",
    "qwen3-32b": "Qwen/Qwen3-32B",
    # Mixtral
    "mixtral-8x7b": "mistralai/Mixtral-8x7B-v0.1",
    "mixtral-8x22b": "mistralai/Mixtral-8x22B-v0.1",
}


def download_tokenizer(
    model_name: str,
    output_dir: Path,
    hf_token: str = None
):
    """
    Download tokenizer from HuggingFace.
    
    Args:
        model_name: Model name (from registry) or HuggingFace path
        output_dir: Directory to save the tokenizer
        hf_token: Optional HuggingFace token
    """
    # Resolve model name to HuggingFace path
    if model_name.lower() in TOKENIZER_REGISTRY:
        hf_path = TOKENIZER_REGISTRY[model_name.lower()]
        print(f"[Info] Resolved '{model_name}' -> '{hf_path}'")
    else:
        hf_path = model_name
        print(f"[Info] Using HuggingFace path directly: {hf_path}")
    
    # Determine output directory
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
    """Print all available tokenizers in the registry."""
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
        help="Model name (e.g., llama3-8b) or HuggingFace path"
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
