#!/bin/bash
# Automated benchmarking script for AMD vs NVIDIA comparison
#
# Usage:
#   ./run_benchmark.sh [model] [runs]
#
# Examples:
#   ./run_benchmark.sh all          # Run all models (llama, mistral, qwen)
#   ./run_benchmark.sh llama        # Run only llama
#   ./run_benchmark.sh mistral 3    # Run mistral 3 times
#
# Available models: llama, mistral, qwen, all

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
MODEL=${1:-"llama"}  # Default to llama if not specified
RUNS=${2:-1}         # Number of runs (default: 1)

# Handle "all" option - run all models
if [ "$MODEL" = "all" ]; then
    echo -e "${CYAN}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║${NC}           ${BLUE}TensorPrimat - All Models${NC}                   ${CYAN}║${NC}"
    echo -e "${CYAN}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    
    # Detect platform
    if ! python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
        echo -e "${RED}❌ No GPU detected!${NC}"
        echo "Please ensure CUDA or ROCm is properly installed."
        exit 1
    fi
    
    if python3 -c "import torch; exit(0 if hasattr(torch.version, 'hip') and torch.version.hip else 1)" 2>/dev/null; then
        PLATFORM="AMD (ROCm)"
        SOFTWARE_STACK="rocm"
        COLOR=$RED
    else
        PLATFORM="NVD (CUDA)"
        SOFTWARE_STACK="cuda"
        COLOR=$GREEN
    fi
    
    echo -e "Platform:  ${COLOR}${PLATFORM}${NC}"
    echo -e "Models:    ${GREEN}llama, mistral, qwen${NC}"
    echo -e "Runs each: ${GREEN}${RUNS}${NC}"
    echo ""
    
    # Check if NeMo is available
    if ! python3 -c "import nemo" 2>/dev/null; then
        echo -e "${YELLOW}⚠️  NeMo not detected${NC}"
        echo -e "${BLUE}→ Using Primus log extraction mode${NC}"
        echo ""
        
        # Auto-extract from logs
        ALL_MODELS=("llama" "mistral" "qwen")
        SUCCESSFUL=()
        FAILED=()
        
        for m in "${ALL_MODELS[@]}"; do
            echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
            echo -e "${BLUE}Extracting: ${GREEN}${m}${NC}"
            echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
            
            # Look for log file patterns
            LOG_FILE=""
            for pattern in "training_${m}.log" "*${m}*.log" "primus_${m}.log" "${m}_training.log"; do
                FOUND=$(ls -t $pattern 2>/dev/null | head -1)
                if [ -n "$FOUND" ]; then
                    LOG_FILE="$FOUND"
                    break
                fi
            done
            
            if [ -z "$LOG_FILE" ]; then
                echo -e "${YELLOW}⚠️  No log file found for ${m}${NC}"
                echo "   Searched for: training_${m}.log, *${m}*.log, primus_${m}.log"
                echo ""
                FAILED+=("${m}")
                continue
            fi
            
            echo -e "${GREEN}✓${NC} Found log: ${LOG_FILE}"
            echo -e "${BLUE}→${NC} Extracting metrics..."
            echo ""
            
            # Extract metrics
            if python3 extract_primus_metrics.py \
                --log-file "$LOG_FILE" \
                --model-name "$m" \
                --num-gpus 8 \
                --global-batch-size 128 \
                --sequence-length 2048; then
                SUCCESSFUL+=("${m}")
                echo -e "${GREEN}✅ ${m} extracted successfully${NC}"
            else
                FAILED+=("${m}")
                echo -e "${RED}❌ ${m} extraction failed${NC}"
            fi
            echo ""
        done
        
        # Summary
        echo -e "${CYAN}╔════════════════════════════════════════════════════════════╗${NC}"
        echo -e "${CYAN}║${NC}                  ${BLUE}Extraction Summary${NC}                      ${CYAN}║${NC}"
        echo -e "${CYAN}╚════════════════════════════════════════════════════════════╝${NC}"
        echo ""
        
        if [ ${#SUCCESSFUL[@]} -gt 0 ]; then
            echo -e "${GREEN}✅ Successful (${#SUCCESSFUL[@]}): ${SUCCESSFUL[@]}${NC}"
            for m in "${SUCCESSFUL[@]}"; do
                echo "   📄 output/benchmark_${SOFTWARE_STACK}_${m}.json"
            done
            echo ""
        fi
        
        if [ ${#FAILED[@]} -gt 0 ]; then
            echo -e "${RED}❌ Failed (${#FAILED[@]}): ${FAILED[@]}${NC}"
            echo ""
            echo -e "${YELLOW}To extract manually:${NC}"
            for m in "${FAILED[@]}"; do
                echo "  python3 extract_primus_metrics.py --log-file training_${m}.log --model-name ${m} --num-gpus 8 --global-batch-size 128 --sequence-length 2048"
            done
            echo ""
        fi
        
        echo -e "${BLUE}Next Steps:${NC}"
        echo "  1. Run on the other platform (AMD/NVD)"
        echo "  2. Compare: ${GREEN}python3 compare_results.py${NC}"
        echo ""
        
        [ ${#FAILED[@]} -eq 0 ] && exit 0 || exit 1
    fi
    
    echo -e "${GREEN}✓ NeMo detected${NC}"
    echo ""
    
    # Run each model
    ALL_MODELS=("llama" "mistral" "qwen")
    SUCCESSFUL=()
    FAILED=()
    
    for MODEL_NAME in "${ALL_MODELS[@]}"; do
        echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo -e "${BLUE}Starting: ${GREEN}${MODEL_NAME}${NC}"
        echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo ""
        
        if "$0" "$MODEL_NAME" "$RUNS"; then
            SUCCESSFUL+=("$MODEL_NAME")
            echo -e "${GREEN}✅ ${MODEL_NAME} completed successfully${NC}"
        else
            FAILED+=("$MODEL_NAME")
            echo -e "${RED}❌ ${MODEL_NAME} failed${NC}"
        fi
        echo ""
    done
    
    # Summary
    echo -e "${CYAN}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║${NC}                  ${BLUE}Benchmark Summary${NC}                      ${CYAN}║${NC}"
    echo -e "${CYAN}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    
    if [ ${#SUCCESSFUL[@]} -gt 0 ]; then
        echo -e "${GREEN}✅ Successful (${#SUCCESSFUL[@]}): ${SUCCESSFUL[@]}${NC}"
        for M in "${SUCCESSFUL[@]}"; do
            echo "   📄 output/benchmark_${SOFTWARE_STACK}_${M}.json"
        done
        echo ""
    fi
    
    if [ ${#FAILED[@]} -gt 0 ]; then
        echo -e "${RED}❌ Failed (${#FAILED[@]}): ${FAILED[@]}${NC}"
        echo ""
    fi
    
    echo -e "${BLUE}Next Steps:${NC}"
    echo "  1. Run on the other platform (AMD/NVD)"
    echo "  2. Compare: ${GREEN}python3 compare_results.py${NC}"
    echo ""
    
    [ ${#FAILED[@]} -eq 0 ] && exit 0 || exit 1
fi

# Single model run
echo -e "${BLUE}=================================${NC}"
echo -e "${BLUE}TensorPrimat${NC}"
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
    SOFTWARE_STACK="rocm"
    COLOR=$RED
else
    PLATFORM="NVD (CUDA)"
    SOFTWARE_STACK="cuda"
    COLOR=$GREEN
fi

echo -e "Platform: ${COLOR}${PLATFORM}${NC}"

# Check if NeMo is available
if ! python3 -c "import nemo" 2>/dev/null; then
    echo -e "${YELLOW}⚠️  NeMo not detected${NC}"
    echo -e "${BLUE}→ Using Primus log extraction mode${NC}"
    echo ""
    
    # Look for log file
    LOG_FILE=""
    for pattern in "training_${MODEL}.log" "*${MODEL}*.log" "primus_${MODEL}.log" "${MODEL}_training.log"; do
        FOUND=$(ls -t $pattern 2>/dev/null | head -1)
        if [ -n "$FOUND" ]; then
            LOG_FILE="$FOUND"
            break
        fi
    done
    
    if [ -z "$LOG_FILE" ]; then
        echo -e "${RED}❌ No log file found for ${MODEL}${NC}"
        echo ""
        echo "Searched for:"
        echo "  - training_${MODEL}.log"
        echo "  - *${MODEL}*.log"
        echo "  - primus_${MODEL}.log"
        echo "  - ${MODEL}_training.log"
        echo ""
        echo -e "${YELLOW}To extract metrics:${NC}"
        echo ""
        echo "  1. Run your Primus training and save logs:"
        echo "     primus train ... 2>&1 | tee training_${MODEL}.log"
        echo ""
        echo "  2. Extract metrics:"
        echo "     python3 extract_primus_metrics.py \\"
        echo "       --log-file training_${MODEL}.log \\"
        echo "       --model-name ${MODEL} \\"
        echo "       --num-gpus 8 \\"
        echo "       --global-batch-size 128 \\"
        echo "       --sequence-length 2048"
        echo ""
        exit 1
    fi
    
    echo -e "${GREEN}✓${NC} Found log: ${LOG_FILE}"
    echo -e "${BLUE}→${NC} Extracting metrics..."
    echo ""
    
    # Extract metrics
    if python3 extract_primus_metrics.py \
        --log-file "$LOG_FILE" \
        --model-name "$MODEL" \
        --num-gpus 8 \
        --global-batch-size 128 \
        --sequence-length 2048; then
        echo ""
        echo -e "${GREEN}✅ Metrics extracted successfully${NC}"
        echo "📄 output/benchmark_${SOFTWARE_STACK}_${MODEL}.json"
        echo ""
        echo -e "${BLUE}Next Steps:${NC}"
        echo "  1. Run on the other platform"
        echo "  2. Compare: ${GREEN}python3 compare_results.py${NC}"
        echo ""
        exit 0
    else
        echo ""
        echo -e "${RED}❌ Extraction failed${NC}"
        exit 1
    fi
fi

echo -e "${GREEN}✓ NeMo detected - Running training mode${NC}"
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
        echo "Available models: llama, mistral, qwen, all"
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
echo "   ${CYAN}./run_benchmark.sh ${MODEL}${NC}"
echo "2. Compare results with: ${GREEN}python3 compare_results.py${NC}"
echo ""
echo -e "${BLUE}Tip:${NC} Run ${CYAN}./run_benchmark.sh all${NC} to benchmark all models"
echo ""

