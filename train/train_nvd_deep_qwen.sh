#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Source config.env if it exists
if [ -f "$ROOT_DIR/config.env" ]; then
    set -a
    source "$ROOT_DIR/config.env"
    set +a
fi

NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
export OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/output}"
export DATA_DIR="${DATA_DIR:-/data/tprimat}"
mkdir -p "$OUTPUT_DIR"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=INFO

deepspeed --num_gpus="$NUM_GPUS" "$SCRIPT_DIR/train_nvd_deep.py" qwen
