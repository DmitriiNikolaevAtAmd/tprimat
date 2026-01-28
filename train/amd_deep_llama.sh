#!/bin/bash
set -e
cd "$(dirname "$0")"
NUM_GPUS=8
OUTPUT_DIR="./output"
mkdir -p "$OUTPUT_DIR"
export HF_HOME="./cache"
mkdir -p "$HF_HOME"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export HSA_NO_SCRATCH_RECLAIM=1
export HSA_ENABLE_SDMA=1
export NCCL_DEBUG=INFO
export RCCL_DEBUG=INFO

deepspeed --num_gpus="$NUM_GPUS" \
          --num_nodes=1 \
          --master_addr=localhost \
          --master_port=29500 \
          amd_deep.py llama

