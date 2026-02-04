#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

source "$ROOT_DIR/config.env"

export DATA_DIR
export OUTPUT_DIR
export HF_HOME

# Data paths
DATA_PREFIX="${DATA_DIR}/allenai-c4-qwen-mega"
TOKENIZER_PATH="${DATA_DIR}/qwen-qwen25-7b"

# Verify data files exist
if [ ! -f "${DATA_PREFIX}.bin" ] || [ ! -f "${DATA_PREFIX}.idx" ]; then
    echo "ERROR: Data files not found at ${DATA_PREFIX}.bin/.idx"
    echo "       Run prepare/prepare.sh first to generate the dataset"
    exit 1
fi
echo "Data prefix: ${DATA_PREFIX}"

NUM_GPUS="${NUM_GPUS:-8}"
mkdir -p "$OUTPUT_DIR"

# AMD performance settings
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HSA_NO_SCRATCH_RECLAIM=1
export HSA_ENABLE_SDMA=1
export HSA_FORCE_FINE_GRAIN_PCIE=1
export RCCL_DEBUG=ERROR
export NCCL_DEBUG=ERROR

cd "$SCRIPT_DIR"

if [ "$NUM_GPUS" -gt 1 ]; then
    torchrun --nproc_per_node="$NUM_GPUS" \
             --nnodes=1 \
             --node_rank=0 \
             --master_addr=localhost \
             --master_port=29500 \
             train_amd_mega.py qwen
else
    python3 -u train_amd_mega.py qwen
fi
