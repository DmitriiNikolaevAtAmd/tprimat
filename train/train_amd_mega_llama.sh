#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TPRIMAT_PATH="$(cd "$SCRIPT_DIR/.." && pwd)"

source "$TPRIMAT_PATH/config.env"

mkdir -p "$TPRIMAT_PATH/output"

# Data paths - uses DATASET from config.env (bc or c4)
DATASET="${DATASET:-bc}"
DATA_PREFIX="${DATA_DIR}/${DATASET}-train"

# Verify data files exist
if [ ! -f "${DATA_PREFIX}.bin" ] || [ ! -f "${DATA_PREFIX}.idx" ]; then
    echo "ERROR: Data files not found:"
    echo "  ${DATA_PREFIX}.bin"
    echo "  ${DATA_PREFIX}.idx"
    echo "  Run prepare/data.sh first to generate the dataset (DATASET=${DATASET})"
    exit 1
fi

export DATA_DIR
export DATASET
export OUTPUT_DIR="${OUTPUT_DIR:-$TPRIMAT_PATH/output}"
export HF_HOME

NUM_GPUS="${NUM_GPUS:-8}"

echo "Config: NUM_GPUS=${NUM_GPUS}"
echo "Dataset: ${DATA_PREFIX} (${DATASET})"

# AMD performance settings
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HSA_NO_SCRATCH_RECLAIM=1
export HSA_ENABLE_SDMA=1
export HSA_FORCE_FINE_GRAIN_PCIE=1
export RCCL_DEBUG=ERROR
export NCCL_DEBUG=ERROR

cd "$SCRIPT_DIR"

# Kill stale distributed training processes from previous runs
pkill -9 -f "torchrun.*train_amd_mega" 2>/dev/null || true
sleep 1

# Use random port to avoid conflicts with stale processes
MASTER_PORT=${MASTER_PORT:-$((30000 + RANDOM % 5000))}
echo "Using MASTER_PORT: $MASTER_PORT"

if [ "$NUM_GPUS" -gt 1 ]; then
    torchrun --nproc_per_node="$NUM_GPUS" \
             --nnodes=1 \
             --node_rank=0 \
             --master_addr=localhost \
             --master_port="$MASTER_PORT" \
             train_amd_mega.py llama
else
    python3 -u train_amd_mega.py llama
fi
