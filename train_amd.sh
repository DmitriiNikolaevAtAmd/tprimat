#!/bin/bash
set -e

# ./train_amd_prim_llama.sh
# ./train_amd_prim_qwen.sh
./train_amd_nemo_llama.sh
./train_amd_nemo_qwen.sh
./train_amd_tran_llama.sh
./train_amd_tran_qwen.sh
./train_amd_deep_llama.sh
./train_amd_deep_qwen.sh
./train_amd_fsdp_llama.sh
./train_amd_fsdp_qwen.sh
./train_amd_mega_llama.sh
./train_amd_mega_qwen.sh
