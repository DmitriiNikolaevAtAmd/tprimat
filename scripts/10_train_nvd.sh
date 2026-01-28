#!/bin/bash
set -e
./11_train_nvd_deep_llama.sh
sleep 10
./11_train_nvd_deep_qwen.sh
sleep 10
./12_train_nvd_fsdp_llama.sh
sleep 10
./12_train_nvd_fsdp_qwen.sh
sleep 10
./13_train_nvd_mega_llama.sh
sleep 10
./13_train_nvd_mega_qwen.sh
sleep 10
./14_train_nvd_nemo_llama.sh
sleep 10
./14_train_nvd_nemo_qwen.sh
sleep 10
./15_train_nvd_tran_llama.sh
sleep 10
./15_train_nvd_tran_qwen.sh
