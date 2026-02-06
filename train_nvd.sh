#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "$SCRIPT_DIR/config.env"

export DATA_DIR
export OUTPUT_DIR
export HF_HOME

mkdir -p "$OUTPUT_DIR"

"$SCRIPT_DIR/purge.sh" --keep-data

echo "=== NVIDIA Megatron ==="
"$SCRIPT_DIR/train/train_nvd_mega_llama.sh"
"$SCRIPT_DIR/train/train_nvd_mega_qwen.sh"

echo "=== NVIDIA NeMo ==="
"$SCRIPT_DIR/train/train_nvd_nemo_llama.sh"
"$SCRIPT_DIR/train/train_nvd_nemo_qwen.sh"

echo "=== Validate & Compare ==="
"$SCRIPT_DIR/evaluate/validate_outputs.sh" --platform nvd
"$SCRIPT_DIR/evaluate/compare.sh"
