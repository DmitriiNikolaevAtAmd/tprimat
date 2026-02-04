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
export USE_TF=NO
export USE_APEX=NO
export TRANSFORMERS_NO_APEX=1

if [ "$NUM_GPUS" -gt 1 ]; then
    torchrun --nproc_per_node="$NUM_GPUS" \
             --nnodes=1 \
             --node_rank=0 \
             --master_addr=localhost \
             --master_port=29500 \
             "$SCRIPT_DIR/nvd_tran.py" llama
else
    python3 -u "$SCRIPT_DIR/nvd_tran.py" llama
fi
