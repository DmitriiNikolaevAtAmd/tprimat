#!/bin/bash
set -e

mkdir -p output

NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)

torchrun --nproc_per_node=$NUM_GPUS train_all_tran.py llama
sleep 10

torchrun --nproc_per_node=$NUM_GPUS train_all_tran.py qwen
sleep 10

torchrun --nproc_per_node=$NUM_GPUS train_nvd_mega.py llama
sleep 10

torchrun --nproc_per_node=$NUM_GPUS train_nvd_mega.py qwen
sleep 10

deepspeed --num_gpus=$NUM_GPUS train_nvd_deep.py llama
sleep 10

deepspeed --num_gpus=$NUM_GPUS train_nvd_deep.py qwen
sleep 10

python3 -u train_all_nemo.py llama
sleep 10

python3 -u train_all_nemo.py qwen
