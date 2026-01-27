#!/bin/bash
set -e
./train_nvd_deep_llama.sh
sleep 10
./train_nvd_deep_qwen.sh
sleep 10
./train_nvd_fsdp_llama.sh
sleep 10
./train_nvd_fsdp_qwen.sh
sleep 10
./train_nvd_mega_llama.sh
sleep 10
./train_nvd_mega_qwen.sh
sleep 10
./train_nvd_nemo_llama.sh
sleep 10
./train_nvd_nemo_qwen.sh
sleep 10
./train_nvd_tran_llama.sh
sleep 10
./train_nvd_tran_qwen.sh
