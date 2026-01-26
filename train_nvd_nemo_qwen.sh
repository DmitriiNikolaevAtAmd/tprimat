#!/bin/bash
set -e
OUTPUT_DIR="./output"
mkdir -p "$OUTPUT_DIR"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=INFO

python3 -u train_nvd_nemo.py qwen
