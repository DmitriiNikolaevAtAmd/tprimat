#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Source shared config (includes profiling settings)
source "$ROOT_DIR/config.env"

export OUTPUT_DIR="$ROOT_DIR/output"
mkdir -p "$OUTPUT_DIR"

export PYTORCH_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=INFO

# Parallel config - qwen NVIDIA (identical_config 01)
export TP=${TP:-1}
export PP=${PP:-1}
export DP=${DP:-8}
export GA=${GA:-8}

# Batch config
export MBS=${MBS:-1}
export GBS=$((MBS * DP * GA))

# Training schedule
export SEQ_LEN=${SEQ_LEN:-2048}
export TRAIN_ITERS=${TRAIN_ITERS:-50}
export WARMUP_STEPS=${WARMUP_STEPS:-10}
export LR=${LR:-3.0e-4}
export WEIGHT_DECAY=${WEIGHT_DECAY:-0.1}

# Profiling (from config.env)
export PROFILING=${PROFILING:-false}
export PROFILE_WAIT=${PROFILE_WAIT:-5}
export PROFILE_WARMUP=${PROFILE_WARMUP:-1}
export PROFILE_ACTIVE=${PROFILE_ACTIVE:-2}
export PROFILE_REPEAT=${PROFILE_REPEAT:-1}

python3 -u "$SCRIPT_DIR/train_nvd_nemo.py" qwen
