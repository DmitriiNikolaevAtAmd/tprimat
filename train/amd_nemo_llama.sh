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
MODEL="llama"
NUM_GPUS="${NUM_GPUS:-8}"

TP="${TP:-1}"
PP="${PP:-1}"
GACC="${GRAD_ACCUM:-16}"

DP=$((NUM_GPUS / (TP * PP)))

OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/output}"
mkdir -p "$OUTPUT_DIR"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export HSA_NO_SCRATCH_RECLAIM=1
export HSA_ENABLE_SDMA=1
export NCCL_DEBUG=INFO
export RCCL_DEBUG=INFO
export NCCL_TIMEOUT=3600
export TORCH_DIST_TIMEOUT=3600
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PARALLEL="amd_optimized"
python3 -u amd_nemo.py llama

