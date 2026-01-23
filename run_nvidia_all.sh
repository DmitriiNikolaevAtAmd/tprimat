#!/bin/bash
set -e

mkdir -p output

NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)

torchrun --nproc_per_node=$NUM_GPUS run_hf_standalone.py llama
torchrun --nproc_per_node=$NUM_GPUS run_hf_standalone.py qwen
torchrun --nproc_per_node=$NUM_GPUS run_fsdp_standalone.py llama
torchrun --nproc_per_node=$NUM_GPUS run_fsdp_standalone.py qwen
python3 -u run_nemo_standalone.py llama
python3 -u run_nemo_standalone.py qwen
