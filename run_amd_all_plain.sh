#!/bin/bash
set -e

mkdir -p output

NUM_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())")

HSA_NO_SCRATCH_RECLAIM=1 HSA_ENABLE_SDMA=1 HSA_FORCE_FINE_GRAIN_PCIE=1 \
torchrun --nproc_per_node=$NUM_GPUS run_tran_plain.py llama
sleep 10

HSA_NO_SCRATCH_RECLAIM=1 HSA_ENABLE_SDMA=1 HSA_FORCE_FINE_GRAIN_PCIE=1 \
torchrun --nproc_per_node=$NUM_GPUS run_tran_plain.py qwen
sleep 10

HSA_NO_SCRATCH_RECLAIM=1 HSA_ENABLE_SDMA=1 HSA_FORCE_FINE_GRAIN_PCIE=1 \
torchrun --nproc_per_node=$NUM_GPUS run_mega_plain.py llama
sleep 10

HSA_NO_SCRATCH_RECLAIM=1 HSA_ENABLE_SDMA=1 HSA_FORCE_FINE_GRAIN_PCIE=1 \
torchrun --nproc_per_node=$NUM_GPUS run_mega_plain.py qwen
sleep 10

HSA_NO_SCRATCH_RECLAIM=1 HSA_ENABLE_SDMA=1 HSA_FORCE_FINE_GRAIN_PCIE=1 \
deepspeed --num_gpus=$NUM_GPUS run_deep_plain.py llama
sleep 10

HSA_NO_SCRATCH_RECLAIM=1 HSA_ENABLE_SDMA=1 HSA_FORCE_FINE_GRAIN_PCIE=1 \
deepspeed --num_gpus=$NUM_GPUS run_deep_plain.py qwen
sleep 10

HSA_NO_SCRATCH_RECLAIM=1 HSA_ENABLE_SDMA=1 HSA_FORCE_FINE_GRAIN_PCIE=1 \
python3 -u run_nemo_plain.py llama
sleep 10

HSA_NO_SCRATCH_RECLAIM=1 HSA_ENABLE_SDMA=1 HSA_FORCE_FINE_GRAIN_PCIE=1 \
python3 -u run_nemo_plain.py qwen
