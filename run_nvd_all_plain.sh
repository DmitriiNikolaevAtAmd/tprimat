#!/bin/bash
set -e

mkdir -p output

NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)

torchrun --nproc_per_node=$NUM_GPUS run_tran_plain.py llama
torchrun --nproc_per_node=$NUM_GPUS run_tran_plain.py qwen
torchrun --nproc_per_node=$NUM_GPUS run_mega_plain.py llama
torchrun --nproc_per_node=$NUM_GPUS run_mega_plain.py qwen
python3 -u run_nemo_plain.py llama
python3 -u run_nemo_plain.py qwen
