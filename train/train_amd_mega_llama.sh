#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TPRIMAT_PATH="$(cd "$SCRIPT_DIR/.." && pwd)"

source "$TPRIMAT_PATH/config.env"

mkdir -p "$TPRIMAT_PATH/output"

export DATA_DIR
export OUTPUT_DIR="${OUTPUT_DIR:-$TPRIMAT_PATH/output}"
export HF_HOME

NUM_GPUS="${NUM_GPUS:-8}"

export TP=${TP:-1}
export PP=${PP:-1}
export DP=${DP:-${NUM_GPUS}}
export GA=${GA:-8}
export MBS=${MBS:-1}
export GBS=$((MBS * DP * GA))

echo "Config: NUM_GPUS=${NUM_GPUS} TP=${TP} PP=${PP} DP=${DP} GA=${GA}"
echo "Batch:  MBS=${MBS} GBS=${GBS} SEQ_LEN=${SEQ_LEN}"

# Performance settings
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONWARNINGS="ignore::UserWarning,ignore::FutureWarning,ignore::DeprecationWarning"
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=error
export HF_HUB_DISABLE_PROGRESS_BARS=1

# AMD-specific tuning
export HSA_NO_SCRATCH_RECLAIM=1
export HSA_ENABLE_SDMA=1
export HSA_FORCE_FINE_GRAIN_PCIE=1
export RCCL_DEBUG=ERROR
export NCCL_DEBUG=ERROR

# Communication tuning
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=1

DATASET="${DATASET:-bc}"
export DATASET
DATA_PREFIX="${DATA_DIR}/${DATASET}-train"

if [ ! -f "${DATA_PREFIX}.bin" ] || [ ! -f "${DATA_PREFIX}.idx" ]; then
    echo "ERROR: Data files not found at ${DATA_PREFIX}.bin/.idx"
    echo "       Run prepare/data.sh first to generate the dataset"
    exit 1
fi

echo ""
echo "=========================================="
echo "Training llama (megatron) on dataset: ${DATASET}"
echo "Stack: Megatron-Core + Lightning"
echo "=========================================="
echo "Dataset: ${DATA_PREFIX} (${DATASET})"

# Start memory monitoring in background (samples every 2 seconds)
MEMORY_LOG="$TPRIMAT_PATH/output/memory_mega_llama_${DATASET}.log"
: > "$MEMORY_LOG"
(
    while true; do
        if command -v rocm-smi &>/dev/null; then
            rocm-smi --showmeminfo vram 2>/dev/null | grep -E "GPU|Used" >> "$MEMORY_LOG"
        elif command -v nvidia-smi &>/dev/null; then
            nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits >> "$MEMORY_LOG"
        fi
        sleep 2
    done
) &
MEMORY_PID=$!
export MEMORY_LOG

python3 -u "$SCRIPT_DIR/train_amd_mega.py" llama

# Stop memory monitoring
kill $MEMORY_PID 2>/dev/null || true
rm -f "$MEMORY_LOG"
