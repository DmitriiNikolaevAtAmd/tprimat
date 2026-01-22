#!/bin/bash
set -e

echo "=========================================="
echo "Running All AMD/ROCm Standalone Scripts"
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
export HSA_NO_SCRATCH_RECLAIM=1
export HSA_ENABLE_SDMA=1
export HSA_FORCE_FINE_GRAIN_PCIE=1

echo "=== [1/3] Running HuggingFace Standalone ==="
echo "Output: output/train_hf_llama.json, output/train_hf_qwen.json"
echo ""
python3 -u run_hf_standalone.py
echo ""

echo "=== [2/3] Running FSDP Standalone ==="
echo "Output: output/train_fsdp_llama.json, output/train_fsdp_qwen.json"
echo ""
python3 -u run_fsdp_standalone.py
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
