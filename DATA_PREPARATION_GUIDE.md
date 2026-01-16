# Data Preparation Guide for NeMo Training

This guide explains how to convert datasets from HuggingFace to the Megatron-LM (`.bin` and `.idx`) format required for high-performance training in NeMo.

## Prerequisites

1. **HuggingFace Account & Token**:
   - Create an account at [huggingface.co](https://huggingface.co/).
   - Generate an Access Token: Settings -> Access Tokens.
   - For Llama 3.1, you must also request access to the [Meta Llama 3.1 8B](https://huggingface.co/meta-llama/Llama-3.1-8B) model to download its tokenizer.

2. **Docker Environment**:
   Ensure you are running the `primat` container with `/data` and `/checkpoints` mounted.

## Step 1: Download & Format Dataset from HuggingFace

Use the `prepare_data.py` script to download a dataset and convert it to the intermediate JSONL format.

```bash
# Inside the container
export HF_TOKEN="your_huggingface_token_here"

# Example: Download wikitext
python3 prepare_data.py \
    --dataset wikitext \
    --subset wikitext-103-v1 \
    --split train \
    --output_file /data/raw_data.jsonl
```

## Step 2: Download the Tokenizer

You need the `tokenizer.model` for the Llama 3.1 model. You can download it using the provided `convert_llama.py` (which also downloads the weights) or download just the tokenizer using `huggingface-cli`.

```bash
# Download just the tokenizer files
huggingface-cli download meta-llama/Llama-3.1-8B tokenizer.model --local-dir /data
```

## Step 3: Convert JSONL to Megatron Binary Format

Now use NeMo's built-in preprocessing script to create the `.bin` and `.idx` files. This step performs tokenization and shuffles the data for efficient disk-mapped access during training.

```bash
python /opt/NeMo/scripts/nlp_language_modeling/preprocess_data_for_megatron.py \
    --input /data/raw_data.jsonl \
    --output-prefix /data/llama_dataset \
    --tokenizer-library sentencepiece \
    --tokenizer-model /data/tokenizer.model \
    --dataset-impl mmap \
    --workers $(nproc)
```

This will produce:
- `/data/llama_dataset_text_document.bin`
- `/data/llama_dataset_text_document.idx`

## Step 4: Update Configuration

Update your `config.yaml` to point to the newly created dataset:

```yaml
training:
  data:
    dataset_path: "/data/llama_dataset_text_document" # Prefix (without .bin/.idx)
    tokenizer_path: "/data/tokenizer.model"
```

## Step 5: Start Training

```bash
python3 pretrain_llama.py
```
