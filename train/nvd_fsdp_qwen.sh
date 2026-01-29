#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
export OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/output}"
export DATA_DIR="${DATA_DIR:-/data/tprimat}"
mkdir -p "$OUTPUT_DIR"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=INFO

torchrun --nproc_per_node="$NUM_GPUS" \
         --nnodes=1 \
         --node_rank=0 \
         --master_addr=localhost \
         --master_port=29500 \
         "$SCRIPT_DIR/nvd_fsdp.py" qwen
