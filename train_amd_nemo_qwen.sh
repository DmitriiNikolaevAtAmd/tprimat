#!/bin/bash

set -e


MODEL="${CONFIG_QWEN_NAME:-qwen}"
NUM_GPUS="${CONFIG_AMD_NUM_GPUS:-8}"

TP="${CONFIG_QWEN_AMD_TP:-1}"
PP="${CONFIG_QWEN_AMD_PP:-1}"
GACC="${CONFIG_QWEN_AMD_GACC:-16}"

DP=$((NUM_GPUS / (TP * PP)))

OUTPUT_DIR="${CONFIG_OUTPUT_DIR:-./output}"
mkdir -p "$OUTPUT_DIR"


export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HSA_NO_SCRATCH_RECLAIM=1
export HSA_ENABLE_SDMA=1
export NCCL_DEBUG=INFO
export RCCL_DEBUG=INFO

export PARALLEL="amd_optimized"

# Use the NVD NeMo script (works on both platforms)
python3 -u train_nvd_nemo_qwen.py

