#!/bin/bash
#
# Wrapper script to run Primus training with benchmarking
#
# Usage:
#   ./run_primus_with_benchmark.sh [model_name]
#
# Example:
#   ./run_primus_with_benchmark.sh llama
#
# Or customize parameters:
#   TRAIN_ITERS=20 GLOBAL_BATCH_SIZE=256 ./run_primus_with_benchmark.sh mistral

set -e

# Get model name from first argument or default to llama
MODEL_NAME="${1:-llama}"

# Configuration (edit these to match your setup)
PRIMUS_DIR="${PRIMUS_DIR:-/workspace/primus}"
CONFIG="${CONFIG:-examples/megatron/configs/MI300X/llama3.1_8B-pretrain.yaml}"
TRAIN_ITERS="${TRAIN_ITERS:-10}"
FP8="${FP8:-hybrid}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-128}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-2048}"
NUM_GPUS="${NUM_GPUS:-8}"

# Output locations
BENCHMARK_DIR="./output"
LOG_FILE="${BENCHMARK_DIR}/primus_training_$(date +%Y%m%d_%H%M%S).log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Primus Training with Benchmarking${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "Model:            ${GREEN}${MODEL_NAME}${NC}"
echo -e "Config:           ${GREEN}${CONFIG}${NC}"
echo -e "Train Iters:      ${GREEN}${TRAIN_ITERS}${NC}"
echo -e "FP8:              ${GREEN}${FP8}${NC}"
echo -e "Micro Batch Size: ${GREEN}${MICRO_BATCH_SIZE}${NC}"
echo -e "Global Batch Size:${GREEN}${GLOBAL_BATCH_SIZE}${NC}"
echo -e "Sequence Length:  ${GREEN}${SEQUENCE_LENGTH}${NC}"
echo -e "Num GPUs:         ${GREEN}${NUM_GPUS}${NC}"
echo -e "Log File:         ${GREEN}${LOG_FILE}${NC}"
echo ""

# Create benchmark directory
mkdir -p "${BENCHMARK_DIR}"

# Navigate to Primus directory if specified
if [ -d "${PRIMUS_DIR}" ]; then
    echo -e "${BLUE}→ Changing to Primus directory: ${PRIMUS_DIR}${NC}"
    cd "${PRIMUS_DIR}"
fi

# Run Primus training
echo -e "${BLUE}→ Starting Primus training...${NC}"
echo ""

EXP="${CONFIG}" \
bash ./examples/run_pretrain.sh \
    --train_iters "${TRAIN_ITERS}" \
    --fp8 "${FP8}" \
    --micro_batch_size "${MICRO_BATCH_SIZE}" \
    --global_batch_size "${GLOBAL_BATCH_SIZE}" \
    2>&1 | tee "${LOG_FILE}"

TRAIN_EXIT_CODE=$?

if [ $TRAIN_EXIT_CODE -ne 0 ]; then
    echo -e "${RED}❌ Training failed with exit code ${TRAIN_EXIT_CODE}${NC}"
    exit $TRAIN_EXIT_CODE
fi

echo ""
echo -e "${GREEN}✅ Training completed successfully${NC}"
echo ""

# Extract metrics from log
echo -e "${BLUE}→ Extracting benchmark metrics...${NC}"

python3 /workspace/support/week-02/code/extract_primus_metrics.py \
    --log-file "${LOG_FILE}" \
    --model-name "${MODEL_NAME}" \
    --num-gpus "${NUM_GPUS}" \
    --global-batch-size "${GLOBAL_BATCH_SIZE}" \
    --sequence-length "${SEQUENCE_LENGTH}"

BENCHMARK_JSON="${BENCHMARK_DIR}/benchmark_rocm_${MODEL_NAME}.json"

EXTRACT_EXIT_CODE=$?

if [ $EXTRACT_EXIT_CODE -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✅ Benchmark complete!${NC}"
    echo ""
    echo -e "${BLUE}Files created:${NC}"
    echo -e "  Training log: ${LOG_FILE}"
    echo -e "  Benchmark:    ${BENCHMARK_JSON}"
    echo ""
    echo -e "${BLUE}Next steps:${NC}"
    echo -e "  1. Run on NVIDIA GPU"
    echo -e "  2. Compare: ${GREEN}cd /workspace/support/week-02/code && python3 compare_results.py${NC}"
    echo ""
else
    echo -e "${YELLOW}⚠️  Metric extraction had issues${NC}"
    echo -e "   Check the log file: ${LOG_FILE}"
    echo -e "   You may need to adjust regex patterns in extract_primus_metrics.py"
    echo ""
fi

