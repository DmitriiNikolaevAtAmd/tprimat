#!/bin/bash
set -e

echo "=========================================="
echo "Running All NVIDIA Standalone Scripts"
echo "=========================================="
echo ""

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Create output directory
mkdir -p output

# Set environment variables for optimal performance
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
export PYTHONHASHSEED="42"

# Detect number of GPUs
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)

echo "=== [1/3] Running HuggingFace Standalone ==="
echo "Output: output/train_hf_llama.json, output/train_hf_qwen.json"
echo ""
torchrun --nproc_per_node=$NUM_GPUS run_hf_standalone.py
echo ""

echo "=== [2/3] Running FSDP Standalone ==="
echo "Output: output/train_fsdp_llama.json, output/train_fsdp_qwen.json"
echo ""
torchrun --nproc_per_node=$NUM_GPUS run_fsdp_standalone.py
echo ""

echo "=== [3/3] Running NeMo Standalone ==="
echo "Output: output/train_nemo_llama.json, output/train_nemo_qwen.json"
echo ""
python3 -u run_nemo_standalone.py
echo ""

echo "=========================================="
echo "All Standalone Scripts Completed!"
echo "=========================================="
echo ""
echo "Results saved to:"
ls -lh output/train_*.json
echo ""
