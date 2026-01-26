#!/bin/bash

set -e


# Hardcoded configuration for standalone execution
MODEL="qwen"
NUM_GPUS=8

TP=1
PP=1
GACC=16

DP=$((NUM_GPUS / (TP * PP)))

OUTPUT_DIR="./output"
mkdir -p "$OUTPUT_DIR"


export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HSA_NO_SCRATCH_RECLAIM=1
export HSA_ENABLE_SDMA=1
export NCCL_DEBUG=INFO
export RCCL_DEBUG=INFO

export PARALLEL="amd_optimized"

# Use the platform-agnostic NeMo script
python3 -u train_all_nemo.py qwen

