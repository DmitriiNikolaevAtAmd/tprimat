#!/bin/bash
set -e
cd "$(dirname "$0")"

./prepare/fetch_deps.py
./prepare/clean_data.py
# Primus requires nemo format for the indexed dataset
./prepare/encode_data.py --nemo
./prepare/verify_data.py --nemo
./train/amd_prim_llama.sh
./train/amd_prim_qwen.sh
# ./train/amd_deep_llama.sh
# ./train/amd_deep_qwen.sh
# ./train/amd_fsdp_llama.sh
# ./train/amd_fsdp_qwen.sh
# ./train/amd_mega_llama.sh
# ./train/amd_mega_qwen.sh
# ./train/amd_tran_llama.sh
# ./train/amd_tran_qwen.sh
./evaluate/compare_amd.sh
