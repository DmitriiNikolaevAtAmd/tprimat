#!/bin/bash
set -e
cd "$(dirname "$0")"

./prepare/fetch_deps.py
./prepare/clean_data.py
./prepare/encode_data.py --mega --nemo
./prepare/verify_data.py --mega --nemo
./train/nvd_nemo_llama.sh
./train/nvd_nemo_qwen.sh
./train/nvd_prim_llama.sh
./train/nvd_prim_qwen.sh
./train/nvd_deep_llama.sh
./train/nvd_deep_qwen.sh
./train/nvd_fsdp_llama.sh
./train/nvd_fsdp_qwen.sh
./train/nvd_mega_llama.sh
./train/nvd_mega_qwen.sh
./train/nvd_tran_llama.sh
./train/nvd_tran_qwen.sh
./evaluate/compare_nvd.sh
