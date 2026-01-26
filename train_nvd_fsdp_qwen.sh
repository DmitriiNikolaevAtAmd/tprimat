#!/bin/bash
set -e
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
OUTPUT_DIR="./output"
mkdir -p "$OUTPUT_DIR"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=INFO

torchrun --nproc_per_node="$NUM_GPUS" \
         --nnodes=1 \
         --node_rank=0 \
         --master_addr=localhost \
         --master_port=29500 \
         train_nvd_fsdp.py qwen
