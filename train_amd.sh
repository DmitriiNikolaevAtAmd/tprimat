#!/bin/bash
set -e

export DATA_DIR="./data"

# ./prepare/fetch_deps.py
# ./prepare/clean_data.py
# ./prepare/encode_data.py
# ./prepare/verify_data.py

./train/amd_prim_llama.sh
./train/amd_prim_qwen.sh

./evaluate/compare_amd.sh
