import os
import argparse
import json
from datasets import load_dataset
from huggingface_hub import login
from pathlib import Path

def prepare_data():
    parser = argparse.ArgumentParser(description="Prepare HuggingFace dataset for NeMo training")
    parser.add_argument("--dataset", type=str, required=True, help="HuggingFace dataset name (e.g., 'wikitext')")
    parser.add_argument("--subset", type=str, default=None, help="Dataset subset")
    parser.add_argument("--split", type=str, default="train", help="Dataset split (default: train)")
    parser.add_argument("--text_column", type=str, default="text", help="Column containing the text data")
    parser.add_argument("--output_file", type=str, default="/data/raw_data.jsonl", help="Output JSONL file path")
    parser.add_argument("--token", type=str, help="HuggingFace API token")
    
    args = parser.parse_args()
    
    if args.token:
        login(token=args.token)
    elif "HF_TOKEN" in os.environ:
        login(token=os.environ["HF_TOKEN"])
    
    print(f"📥 Downloading dataset: {args.dataset} (subset: {args.subset}, split: {args.split})...")
    dataset = load_dataset(args.dataset, args.subset, split=args.split)
    
    print(f"✍️ Writing to {args.output_file}...")
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in dataset:
            text = entry[args.text_column]
            if text and text.strip():
                # Mega-LM preprocessor expects a JSON object per line with a 'text' key
                json.dump({"text": text}, f, ensure_ascii=False)
                f.write("\n")
                
    print(f"  + Successfully saved {len(dataset)} examples to {args.output_file}")
    print("\nNext step: Run the NeMo preprocessor to convert JSONL to .bin and .idx format.")
    print("Example command for Llama 3.1 (uses HuggingFace tokenizer):")
    print(f"python /opt/NeMo/scripts/nlp_language_modeling/preprocess_data_for_mega.py \\")
    print(f"    --input {args.output_file} \\")
    print(f"    --output-prefix /data/llama_dataset \\")
    print(f"    --tokenizer-library huggingface \\")
    print(f"    --tokenizer-type meta-llama/Llama-3.1-8B \\")
    print(f"    --dataset-impl mmap \\")
    print(f"    --workers 16")

if __name__ == "__main__":
    prepare_data()
