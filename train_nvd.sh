#!/bin/bash
set -e

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Source config.env if it exists (sets DATA_DIR, OUTPUT_DIR, HF_HOME, etc.)
if [ -f "$ROOT_DIR/config.env" ]; then
    set -a
    source "$ROOT_DIR/config.env"
    set +a
fi

# If user wants /data/tprimat but repo data is in ./data, create a symlink
# so both prepare/* and train/* see the same files.
if [ "${DATA_DIR:-}" = "/data/tprimat" ] && [ ! -e "/data/tprimat" ] && [ -d "$ROOT_DIR/data" ]; then
    mkdir -p /data
    ln -s "$ROOT_DIR/data" /data/tprimat
fi

./prepare/fetch_deps.py
./prepare/clean_data.py
./prepare/encode_data.py
./prepare/verify_data.py

# ./train/nvd_deep_llama.sh
# sleep 10
# ./train/nvd_deep_qwen.sh
# sleep 10
# ./train/nvd_fsdp_llama.sh
# sleep 10
# ./train/nvd_fsdp_qwen.sh
# sleep 10
# ./train/nvd_mega_llama.sh
# sleep 10
# ./train/nvd_mega_qwen.sh
# sleep 10
./train/nvd_nemo_llama.sh
sleep 10
./train/nvd_nemo_qwen.sh
# sleep 10
# ./train/nvd_tran_llama.sh
# sleep 10
# ./train/nvd_tran_qwen.sh

# ./evaluate/validate_outputs.sh --platform nvd && ./evaluate/compare_nvd.sh

./wrap.sh
