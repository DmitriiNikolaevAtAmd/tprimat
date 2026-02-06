#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "$SCRIPT_DIR/config.env"

export DATA_DIR
export OUTPUT_DIR
export HF_HOME

mkdir -p "$OUTPUT_DIR"

"$SCRIPT_DIR/purge.sh" --keep-data

echo "=== AMD Megatron ==="
"$SCRIPT_DIR/train/train_amd_mega_llama.sh"
"$SCRIPT_DIR/train/train_amd_mega_qwen.sh"

echo "=== AMD NeMo ==="
"$SCRIPT_DIR/train/train_amd_nemo_llama.sh"
"$SCRIPT_DIR/train/train_amd_nemo_qwen.sh"

echo "=== Validate & Compare ==="
"$SCRIPT_DIR/evaluate/validate_outputs.sh" --platform amd
"$SCRIPT_DIR/evaluate/compare.sh"
