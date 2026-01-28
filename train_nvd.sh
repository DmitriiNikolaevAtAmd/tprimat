#!/bin/bash
set -e

export DATA_DIR="./data"

./prepare/fetch_deps.py
./prepare/clean_data.py
./prepare/encode_data.py
./prepare/verify_data.py

./train/nvd_deep_llama.sh
sleep 10
./train/nvd_deep_qwen.sh
sleep 10
./train/nvd_fsdp_llama.sh
sleep 10
./train/nvd_fsdp_qwen.sh
sleep 10
./train/nvd_mega_llama.sh
sleep 10
./train/nvd_mega_qwen.sh
sleep 10
./train/nvd_nemo_llama.sh
sleep 10
./train/nvd_nemo_qwen.sh
sleep 10
./train/nvd_tran_llama.sh
sleep 10
./train/nvd_tran_qwen.sh

./evaluate/compare_nvd.sh
