#!/bin/bash
set -e
cd "$(dirname "$0")"
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
OUTPUT_DIR="./output"
mkdir -p "$OUTPUT_DIR"
export HF_HOME="./cache"
mkdir -p "$HF_HOME"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=INFO

deepspeed --num_gpus="$NUM_GPUS" 11_train_nvd_deep.py llama
