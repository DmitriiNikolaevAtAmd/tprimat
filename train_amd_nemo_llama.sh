#!/bin/bash

set -e


MODEL="${CONFIG_LLAMA_NAME:-llama}"
NUM_GPUS="${CONFIG_AMD_NUM_GPUS:-8}"

TP="${CONFIG_LLAMA_AMD_TP:-1}"
PP="${CONFIG_LLAMA_AMD_PP:-1}"
GACC="${CONFIG_LLAMA_AMD_GACC:-16}"

DP=$((NUM_GPUS / (TP * PP)))

OUTPUT_DIR="${CONFIG_OUTPUT_DIR:-./output}"
mkdir -p "$OUTPUT_DIR"


export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HSA_NO_SCRATCH_RECLAIM=1
export HSA_ENABLE_SDMA=1
export NCCL_DEBUG=INFO
export RCCL_DEBUG=INFO

export PARALLEL="amd_optimized"

python3 -u train_nemo_llama.py

