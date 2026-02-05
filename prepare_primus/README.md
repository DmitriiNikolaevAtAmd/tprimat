# Primus Data Preparation Scripts

This folder contains standalone scripts for downloading and preprocessing datasets for training with Primus.

## Quick Start

```bash
# Set your HuggingFace token (required for gated models like Llama)
export HF_TOKEN=your_huggingface_token

# Prepare everything (datasets + tokenizers + tokenization)
./prepare.sh --all

# Or run individual steps
./prepare.sh --dataset bookcorpus    # Download BookCorpus
./prepare.sh --dataset c4            # Download C4 samples
./prepare.sh --tokenizer llama3-8b   # Download Llama 3 tokenizer
./prepare.sh --tokenize              # Tokenize for Megatron
```

## Scripts

| Script | Description |
|--------|-------------|
| `prepare.sh` | Master script that orchestrates all preparation steps |
| `download_bookcorpus.py` | Downloads BookCorpus dataset from HuggingFace |
| `download_c4.py` | Downloads C4 dataset (streaming or sampled) |
| `download_tokenizer.py` | Downloads tokenizers for various model families |
| `tokenize_megatron.py` | Tokenizes data into Megatron binary format |

## Datasets

### BookCorpus
- **URL**: https://huggingface.co/datasets/bookcorpus
- **Size**: ~5GB
- **Format**: Text (books)
- **Usage**: Megatron pretraining

```bash
python download_bookcorpus.py --output-dir ./data/bookcorpus
```

### C4 (Colossal Clean Crawled Corpus)
- **URL**: https://huggingface.co/datasets/allenai/c4
- **Size**: ~750GB (full), streaming supported
- **Format**: Web text
- **Usage**: MaxText, TorchTitan pretraining

```bash
# Download samples
python download_c4.py --output-dir ./data/c4 --max-samples 10000

# Or use streaming in training (no download needed)
python download_c4.py --output-dir ./data/c4
```

## Tokenizers

### Available Tokenizers

| Name | HuggingFace Path |
|------|------------------|
| `llama2-7b` | `meta-llama/Llama-2-7b-hf` |
| `llama2-70b` | `meta-llama/Llama-2-70b-hf` |
| `llama3-8b` | `meta-llama/Meta-Llama-3-8B` |
| `llama3-70b` | `meta-llama/Meta-Llama-3-70B` |
| `llama3.1-8b` | `meta-llama/Llama-3.1-8B` |
| `llama3.1-70b` | `meta-llama/Llama-3.1-70B` |
| `llama3.1-405b` | `meta-llama/Llama-3.1-405B` |
| `llama3.3-70b` | `meta-llama/Llama-3.3-70B-Instruct` |
| `llama4-scout` | `meta-llama/Llama-4-Scout-17B-16E` |
| `llama4-maverick` | `meta-llama/Llama-4-Maverick-17B-128E` |
| `deepseek-v2` | `deepseek-ai/DeepSeek-V2` |
| `deepseek-v2-lite` | `deepseek-ai/DeepSeek-V2-Lite` |
| `deepseek-v3` | `deepseek-ai/DeepSeek-V3` |
| `qwen2.5-7b` | `Qwen/Qwen2.5-7B` |
| `qwen2.5-72b` | `Qwen/Qwen2.5-72B` |
| `qwen3-0.6b` | `Qwen/Qwen3-0.6B` |
| `qwen3-1.7b` | `Qwen/Qwen3-1.7B` |
| `qwen3-32b` | `Qwen/Qwen3-32B` |
| `mixtral-8x7b` | `mistralai/Mixtral-8x7B-v0.1` |
| `mixtral-8x22b` | `mistralai/Mixtral-8x22B-v0.1` |

```bash
# List all tokenizers
python download_tokenizer.py --list

# Download specific tokenizer
python download_tokenizer.py llama3-8b --output-dir ./data/tokenizers

# Or use HuggingFace path directly
python download_tokenizer.py meta-llama/Llama-3.1-8B --output-dir ./data/tokenizers
```

## Tokenization for Megatron

Megatron requires data in binary `.bin`/`.idx` format:

```bash
python tokenize_megatron.py \
    --input ./data/bookcorpus/bookcorpus_megatron.json \
    --output-prefix ./data/megatron/bookcorpus \
    --tokenizer-model meta-llama/Meta-Llama-3-8B \
    --workers 8 \
    --split-sentences \
    --append-eod
```

This creates:
- `bookcorpus_text_sentence.bin` - Binary token data
- `bookcorpus_text_sentence.idx` - Index file

## Environment Variables

| Variable | Description |
|----------|-------------|
| `HF_TOKEN` | HuggingFace token (required for gated models) |
| `DATA_DIR` | Data directory (default: `./data`) |
| `WORKERS` | Number of workers for tokenization |

## Output Structure

After running `./prepare.sh --all`:

```
data/
├── bookcorpus/
│   └── bookcorpus_megatron.json     # Raw BookCorpus data
├── c4/
│   └── c4_en_train_10000.jsonl      # C4 samples
├── tokenizers/
│   └── meta-llama_Meta-Llama-3-8B/  # Downloaded tokenizer
│       ├── tokenizer.json
│       ├── tokenizer_config.json
│       └── ...
└── megatron/
    ├── bookcorpus_text_sentence.bin # Tokenized binary data
    └── bookcorpus_text_sentence.idx # Index file
```

## Requirements

```bash
pip install datasets transformers nltk numpy
```

Or use the project's requirements:

```bash
pip install -r ../requirements.txt
```
