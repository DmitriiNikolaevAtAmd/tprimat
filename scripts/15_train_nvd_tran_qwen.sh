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
export USE_TF=NO
export USE_APEX=NO
export TRANSFORMERS_NO_APEX=1

if [ "$NUM_GPUS" -gt 1 ]; then
    torchrun --nproc_per_node="$NUM_GPUS" \
             --nnodes=1 \
             --node_rank=0 \
             --master_addr=localhost \
             --master_port=29500 \
             15_train_nvd_tran.py qwen
else
    python3 -u 15_train_nvd_tran.py qwen
fi
