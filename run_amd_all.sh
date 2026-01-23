#!/bin/bash
set -e

mkdir -p output

NUM_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())")

HSA_NO_SCRATCH_RECLAIM=1 HSA_ENABLE_SDMA=1 HSA_FORCE_FINE_GRAIN_PCIE=1 \
torchrun --nproc_per_node=$NUM_GPUS run_hf_standalone.py llama

HSA_NO_SCRATCH_RECLAIM=1 HSA_ENABLE_SDMA=1 HSA_FORCE_FINE_GRAIN_PCIE=1 \
torchrun --nproc_per_node=$NUM_GPUS run_hf_standalone.py qwen

HSA_NO_SCRATCH_RECLAIM=1 HSA_ENABLE_SDMA=1 HSA_FORCE_FINE_GRAIN_PCIE=1 \
torchrun --nproc_per_node=$NUM_GPUS run_fsdp_standalone.py llama

HSA_NO_SCRATCH_RECLAIM=1 HSA_ENABLE_SDMA=1 HSA_FORCE_FINE_GRAIN_PCIE=1 \
torchrun --nproc_per_node=$NUM_GPUS run_fsdp_standalone.py qwen

HSA_NO_SCRATCH_RECLAIM=1 HSA_ENABLE_SDMA=1 HSA_FORCE_FINE_GRAIN_PCIE=1 \
python3 -u run_nemo_standalone.py llama

HSA_NO_SCRATCH_RECLAIM=1 HSA_ENABLE_SDMA=1 HSA_FORCE_FINE_GRAIN_PCIE=1 \
python3 -u run_nemo_standalone.py qwen
