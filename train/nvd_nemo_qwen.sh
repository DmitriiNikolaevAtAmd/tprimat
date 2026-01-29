#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

export OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/output}"
export DATA_DIR="${DATA_DIR:-/data/tprimat}"
mkdir -p "$OUTPUT_DIR"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=INFO

# Kineto GPU profiling
export PROFILING=true
export PROFILE_WAIT=2
export PROFILE_WARMUP=2
export PROFILE_ACTIVE=5
export PROFILE_REPEAT=1

python3 -u "$SCRIPT_DIR/nvd_nemo.py" qwen
