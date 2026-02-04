#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Source config.env if it exists
if [ -f "$ROOT_DIR/config.env" ]; then
    set -a
    source "$ROOT_DIR/config.env"
    set +a
fi

cd "$SCRIPT_DIR"
NUM_GPUS="${NUM_GPUS:-8}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/output}"
mkdir -p "$OUTPUT_DIR"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export HSA_NO_SCRATCH_RECLAIM=1
export HSA_ENABLE_SDMA=1
export NCCL_DEBUG=INFO
export RCCL_DEBUG=INFO

deepspeed --num_gpus="$NUM_GPUS" \
          --num_nodes=1 \
          --master_addr=localhost \
          --master_port=29500 \
          amd_deep.py qwen

