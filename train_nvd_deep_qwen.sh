#!/bin/bash
set -e
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
OUTPUT_DIR="./output"
mkdir -p "$OUTPUT_DIR"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=INFO

deepspeed --num_gpus="$NUM_GPUS" train_nvd_deep.py qwen
