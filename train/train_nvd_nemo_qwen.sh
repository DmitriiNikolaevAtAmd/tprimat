#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

export OUTPUT_DIR="$ROOT_DIR/output"
export DATA_DIR="/data/tprimat"
mkdir -p "$OUTPUT_DIR"

export PYTORCH_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=INFO

# Parallel config - qwen NVIDIA (identical_config 01)
export TP=4
export PP=2
export DP=1
export GA=128

# Batch config
export MBS=1
export GBS=1024  # MBS * NUM_GPUS * GA = 1 * 8 * 128

# Training schedule
export SEQ_LEN=2048
export TRAIN_ITERS=50
export WARMUP_STEPS=10
export LR=3.0e-4
export WEIGHT_DECAY=0.1

# Profiling
export PROFILING=false
export PROFILE_WAIT=5
export PROFILE_WARMUP=1
export PROFILE_ACTIVE=2
export PROFILE_REPEAT=1

python3 -u "$SCRIPT_DIR/train_nvd_nemo.py" qwen
