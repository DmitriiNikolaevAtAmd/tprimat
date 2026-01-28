#!/bin/bash
set -e
cd "$(dirname "$0")"
./nvd_deep_llama.sh
sleep 10
./nvd_deep_qwen.sh
sleep 10
./nvd_fsdp_llama.sh
sleep 10
./nvd_fsdp_qwen.sh
sleep 10
./nvd_mega_llama.sh
sleep 10
./nvd_mega_qwen.sh
sleep 10
./nvd_nemo_llama.sh
sleep 10
./nvd_nemo_qwen.sh
sleep 10
./nvd_tran_llama.sh
sleep 10
./nvd_tran_qwen.sh
