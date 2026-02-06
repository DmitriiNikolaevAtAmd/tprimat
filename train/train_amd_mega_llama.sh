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

# Parallel / batch config
export TP=${TP:-1}
export PP=${PP:-1}
export DP=${DP:-${NUM_GPUS}}
export GA=${GA:-8}
export MBS=${MBS:-1}
export GBS=$((MBS * DP * GA))

echo "Config: NUM_GPUS=${NUM_GPUS} TP=${TP} PP=${PP} DP=${DP} GA=${GA}"
echo "Batch:  MBS=${MBS} GBS=${GBS} SEQ_LEN=${SEQ_LEN}"

# AMD performance settings
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HSA_NO_SCRATCH_RECLAIM=1
export HSA_ENABLE_SDMA=1
export HSA_FORCE_FINE_GRAIN_PCIE=1
export RCCL_DEBUG=ERROR
export NCCL_DEBUG=ERROR

# Communication tuning
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_ASYNC_ERROR_HANDLING=1

cd "$SCRIPT_DIR"

# Kill stale distributed training processes from previous runs
pkill -9 -f "torchrun.*train_amd_mega" 2>/dev/null || true
sleep 1

# Train on all datasets
for DATASET in bc c4; do
    export DATASET
    DATA_PREFIX="${DATA_DIR}/${DATASET}-train"

    # Verify data files exist
    if [ ! -f "${DATA_PREFIX}.bin" ] || [ ! -f "${DATA_PREFIX}.idx" ]; then
        echo "WARNING: Data files not found for dataset '${DATASET}', skipping:"
        echo "  ${DATA_PREFIX}.bin"
        echo "  ${DATA_PREFIX}.idx"
        continue
    fi

    echo ""
    echo "=========================================="
    echo "Training llama on dataset: ${DATASET}"
    echo "=========================================="
    echo "Config: NUM_GPUS=${NUM_GPUS}"
    echo "Dataset: ${DATA_PREFIX} (${DATASET})"

    # Use random port to avoid conflicts with stale processes
    MASTER_PORT=$((30000 + RANDOM % 5000))
    echo "Using MASTER_PORT: $MASTER_PORT"

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

    if [ "$NUM_GPUS" -gt 1 ]; then
        torchrun --nproc_per_node="$NUM_GPUS" \
                 --nnodes=1 \
                 --node_rank=0 \
                 --master_addr=localhost \
                 --master_port="$MASTER_PORT" \
                 train_amd_mega.py llama
    else
        python3 -u train_amd_mega.py llama
    fi

    # Stop memory monitoring
    kill $MEMORY_PID 2>/dev/null || true
    rm -f "$MEMORY_LOG"

    echo "Completed dataset: ${DATASET}"
done

echo ""
echo "All datasets completed for llama (mega)."
