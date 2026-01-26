#!/bin/bash

set -e


NUM_GPUS="${CONFIG_AMD_NUM_GPUS:-8}"
OUTPUT_DIR="${CONFIG_OUTPUT_DIR:-./output}"
MEGATRON_DEEPSPEED_PATH="${MEGATRON_DEEPSPEED_PATH:-/workspace/Megatron-DeepSpeed}"

mkdir -p "$OUTPUT_DIR"


if [ ! -d "$MEGATRON_DEEPSPEED_PATH" ]; then
    exit 1
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HSA_NO_SCRATCH_RECLAIM=1
export HSA_ENABLE_SDMA=1
export NCCL_DEBUG=INFO
export RCCL_DEBUG=INFO
export MEGATRON_DEEPSPEED_PATH

python3 -u train_all_mgds.py qwen

