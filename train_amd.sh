#!/bin/bash
set -e

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Source config.env if it exists (sets DATA_DIR, OUTPUT_DIR, HF_HOME, etc.)
if [ -f "$ROOT_DIR/config.env" ]; then
    set -a
    source "$ROOT_DIR/config.env"
    set +a
fi

if [ "${DATA_DIR:-}" = "/data/tprimat" ] && [ ! -e "/data/tprimat" ] && [ -d "$ROOT_DIR/data" ]; then
    mkdir -p /data
    ln -s "$ROOT_DIR/data" /data/tprimat
fi

# ./prepare/fetch_deps.py
# ./prepare/clean_data.py
# ./prepare/encode_data.py
# ./prepare/verify_data.py

./train/amd_prim_llama.sh
./train/amd_prim_qwen.sh

./evaluate/validate_outputs.sh --platform amd && ./evaluate/compare_amd.sh

./wrap.sh
