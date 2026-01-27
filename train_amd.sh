#!/bin/bash
set -e
./train_amd_deep_llama.sh
sleep 10
./train_amd_deep_qwen.sh
sleep 10
./train_amd_fsdp_llama.sh
sleep 10
./train_amd_fsdp_qwen.sh
sleep 10
./train_amd_mega_llama.sh
sleep 10
./train_amd_mega_qwen.sh
sleep 10
./train_amd_nemo_llama.sh
sleep 10
./train_amd_nemo_qwen.sh
sleep 10
./train_amd_tran_llama.sh
sleep 10
./train_amd_tran_qwen.sh
