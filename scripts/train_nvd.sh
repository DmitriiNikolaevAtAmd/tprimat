#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

source "$ROOT_DIR/config.env"

export DATA_DIR
export OUTPUT_DIR
export HF_HOME

mkdir -p "$OUTPUT_DIR"

"$ROOT_DIR/train/train_nvd_mega_llama.sh"
"$ROOT_DIR/train/train_nvd_mega_qwen.sh"
"$ROOT_DIR/train/train_nvd_nemo_llama.sh"
"$ROOT_DIR/train/train_nvd_nemo_qwen.sh"
"$ROOT_DIR/evaluate/compare.sh"
"$ROOT_DIR/scripts/wrap.sh"
