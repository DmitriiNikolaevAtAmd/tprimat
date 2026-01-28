#!/bin/bash
set -e
cd "$(dirname "$0")"
OUTPUT_DIR="./output"
mkdir -p "$OUTPUT_DIR"
export HF_HOME="./cache"
mkdir -p "$HF_HOME"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=INFO

python3 -u 14_train_nvd_nemo.py qwen
