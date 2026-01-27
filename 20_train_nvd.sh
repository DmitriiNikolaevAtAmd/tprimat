#!/bin/bash
set -e
./22_train_nvd_deep_llama.sh
sleep 10
./23_train_nvd_deep_qwen.sh
sleep 10
./25_train_nvd_fsdp_llama.sh
sleep 10
./26_train_nvd_fsdp_qwen.sh
sleep 10
./28_train_nvd_mega_llama.sh
sleep 10
./29_train_nvd_mega_qwen.sh
sleep 10
./31_train_nvd_nemo_llama.sh
sleep 10
./32_train_nvd_nemo_qwen.sh
sleep 10
./34_train_nvd_tran_llama.sh
sleep 10
./35_train_nvd_tran_qwen.sh
