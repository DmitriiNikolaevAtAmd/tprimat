#!/bin/bash
set -e
./21_train_nvd_deep_llama.sh
sleep 10
./21_train_nvd_deep_qwen.sh
sleep 10
./22_train_nvd_fsdp_llama.sh
sleep 10
./22_train_nvd_fsdp_qwen.sh
sleep 10
./23_train_nvd_mega_llama.sh
sleep 10
./23_train_nvd_mega_qwen.sh
sleep 10
./24_train_nvd_nemo_llama.sh
sleep 10
./24_train_nvd_nemo_qwen.sh
sleep 10
./25_train_nvd_tran_llama.sh
sleep 10
./25_train_nvd_tran_qwen.sh
