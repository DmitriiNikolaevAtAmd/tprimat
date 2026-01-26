#!/bin/bash
set -e

TPRIMAT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
export PYTHONHASHSEED="42"
export HSA_NO_SCRATCH_RECLAIM=1
export HSA_ENABLE_SDMA=1
export HSA_FORCE_FINE_GRAIN_PCIE=1
export RCCL_DEBUG=INFO
export NCCL_DEBUG=INFO

mkdir -p "$TPRIMAT_PATH/output"

cd "$TPRIMAT_PATH"

python3 train_nvd_mega.py llama

python3 train_nvd_mega.py qwen

