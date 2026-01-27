#!/bin/bash
set -e
MODEL="qwen"
NUM_GPUS=8

TP=1
PP=1
GACC=16

DP=$((NUM_GPUS / (TP * PP)))

OUTPUT_DIR="./output"
mkdir -p "$OUTPUT_DIR"
export HF_HOME="./cache"
mkdir -p "$HF_HOME"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export HSA_NO_SCRATCH_RECLAIM=1
export HSA_ENABLE_SDMA=1

export NCCL_DEBUG=INFO
export RCCL_DEBUG=INFO
export NCCL_TIMEOUT=3600
export TORCH_DIST_TIMEOUT=3600
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

export PARALLEL="amd_optimized"

python3 -u 45_train_amd_nemo.py qwen

