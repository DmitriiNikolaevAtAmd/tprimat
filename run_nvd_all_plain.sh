#!/bin/bash
set -e

mkdir -p output

NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)

torchrun --nproc_per_node=$NUM_GPUS run_tran_plain.py llama
sleep 10

torchrun --nproc_per_node=$NUM_GPUS run_tran_plain.py qwen
sleep 10

torchrun --nproc_per_node=$NUM_GPUS run_mega_plain.py llama
sleep 10

torchrun --nproc_per_node=$NUM_GPUS run_mega_plain.py qwen
sleep 10

deepspeed --num_gpus=$NUM_GPUS run_deep_plain.py llama
sleep 10

deepspeed --num_gpus=$NUM_GPUS run_deep_plain.py qwen
sleep 10

python3 -u run_nemo_plain.py llama
sleep 10

python3 -u run_nemo_plain.py qwen
