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

