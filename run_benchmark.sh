#!/bin/bash
# Automated benchmarking script for AMD vs NVIDIA comparison

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
MODEL=${1:-"llama"}  # Default to llama if not specified
RUNS=${2:-1}         # Number of runs (default: 1)

echo -e "${BLUE}=================================${NC}"
echo -e "${BLUE}NeMo Benchmark Runner${NC}"
echo -e "${BLUE}=================================${NC}"
echo -e "Model: ${GREEN}${MODEL}${NC}"
echo -e "Runs:  ${GREEN}${RUNS}${NC}"
echo ""

# Detect platform
if ! python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo -e "${RED}❌ No GPU detected!${NC}"
    echo "Please ensure CUDA or ROCm is properly installed."
    exit 1
fi

# Check if it's ROCm or CUDA
if python3 -c "import torch; exit(0 if hasattr(torch.version, 'hip') and torch.version.hip else 1)" 2>/dev/null; then
    PLATFORM="AMD (ROCm)"
    COLOR=$RED
else
    PLATFORM="NVD (CUDA)"
    COLOR=$GREEN
fi

echo -e "Platform: ${COLOR}${PLATFORM}${NC}"
echo ""

# Select training script
case $MODEL in
    llama)
        SCRIPT="pretrain_llama.py"
        MODEL_NAME="Llama 3.1 8B"
        ;;
    qwen)
        SCRIPT="pretrain_qwen.py"
        MODEL_NAME="Qwen 2.5 7B"
        ;;
    mistral)
        SCRIPT="pretrain_mistral.py"
        MODEL_NAME="Mistral 7B"
        ;;
    *)
        echo -e "${RED}❌ Unknown model: ${MODEL}${NC}"
        echo "Available models: llama, qwen, mistral"
        exit 1
        ;;
esac

if [ ! -f "$SCRIPT" ]; then
    echo -e "${RED}❌ Script not found: ${SCRIPT}${NC}"
    exit 1
fi

echo -e "Training: ${GREEN}${MODEL_NAME}${NC}"
echo -e "Script:   ${SCRIPT}"
echo ""

# Create results directory
mkdir -p output

# Run benchmarks
for ((i=1; i<=$RUNS; i++)); do
    echo -e "${BLUE}=================================${NC}"
    echo -e "${BLUE}Run ${i}/${RUNS}${NC}"
    echo -e "${BLUE}=================================${NC}"
    
    # Run training
    python3 "$SCRIPT"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Run ${i} completed successfully${NC}"
    else
        echo -e "${RED}❌ Run ${i} failed${NC}"
        exit 1
    fi
    
    # Cool down between runs
    if [ $i -lt $RUNS ]; then
        echo -e "${YELLOW}Cooling down for 10 seconds...${NC}"
        sleep 10
    fi
    echo ""
done

echo -e "${BLUE}=================================${NC}"
echo -e "${GREEN}✅ All benchmarks completed!${NC}"
echo -e "${BLUE}=================================${NC}"
echo ""

# Show latest results
echo -e "${BLUE}Latest Results:${NC}"

# Determine the expected output file based on platform and model
if python3 -c "import torch; exit(0 if hasattr(torch.version, 'hip') and torch.version.hip else 1)" 2>/dev/null; then
    SOFTWARE_STACK="rocm"
else
    SOFTWARE_STACK="cuda"
fi

RESULT_FILE="output/benchmark_${SOFTWARE_STACK}_${MODEL}.json"

if [ -f "$RESULT_FILE" ]; then
    echo "📄 $RESULT_FILE"
    
    # Extract and display key metrics
    if command -v python3 &> /dev/null; then
        python3 -c "
import json
with open('$RESULT_FILE', 'r') as f:
    data = json.load(f)
print(f\"Platform:        {data['platform'].upper()}\")
print(f\"Device:          {data['gpu_info']['device_name']}\")
print(f\"Avg Step Time:   {data['performance_metrics']['avg_step_time_seconds']:.4f}s\")
if 'tokens_per_second_per_gpu' in data['performance_metrics']:
    print(f\"Tokens/sec/GPU:  {data['performance_metrics']['tokens_per_second_per_gpu']:,.0f}\")
if 'memory_metrics' in data:
    print(f\"Peak Memory:     {data['memory_metrics']['peak_memory_allocated_gb']:.2f} GB\")
" 2>/dev/null
    fi
else
    echo -e "${YELLOW}⚠️  No benchmark results found at ${RESULT_FILE}${NC}"
fi

echo ""
echo -e "${BLUE}Next Steps:${NC}"
echo "1. Run this script on the other platform (AMD/NVD)"
echo "2. Compare results with: python3 compare_results.py"
echo ""

