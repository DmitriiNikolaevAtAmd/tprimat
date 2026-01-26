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

# Train Llama 3.1 8B with Megatron (without Primus)
echo "Training Llama 3.1 8B with Megatron..."
python3 run_mega_plain.py llama

# Train Qwen 2.5 7B with Megatron (without Primus)
echo "Training Qwen 2.5 7B with Megatron..."
python3 run_mega_plain.py qwen

echo "All Megatron training completed!"
