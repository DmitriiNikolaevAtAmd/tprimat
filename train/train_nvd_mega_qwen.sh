#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TPRIMAT_PATH="$(cd "$SCRIPT_DIR/.." && pwd)"

source "$TPRIMAT_PATH/config.env"

mkdir -p "$TPRIMAT_PATH/output"

NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
export OUTPUT_DIR="${OUTPUT_DIR:-$TPRIMAT_PATH/output}"
export DATA_DIR

# Parallel config
export TP=${TP:-1}
export PP=${PP:-1}
export DP=${DP:-8}
export GA=${GA:-8}

# Batch config
export MBS=${MBS:-1}
export GBS=$((MBS * DP * GA))

echo "Config: NUM_GPUS=${NUM_GPUS} TP=${TP} PP=${PP} DP=${DP} GA=${GA}"
echo "Batch: MBS=${MBS} GBS=${GBS} SEQ_LEN=${SEQ_LEN}"

# Performance settings
export PYTORCH_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=ERROR
export PYTHONWARNINGS="ignore::UserWarning,ignore::FutureWarning,ignore::DeprecationWarning"
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=error
export HF_HUB_DISABLE_PROGRESS_BARS=1

# CUDA / NCCL performance tuning
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=1

# Train on all datasets
for DATASET in bc c4; do
    export DATASET
    DATA_PREFIX="${DATA_DIR}/${DATASET}-train"

    # Verify data files exist
    if [ ! -f "${DATA_PREFIX}.bin" ] || [ ! -f "${DATA_PREFIX}.idx" ]; then
        echo "WARNING: Data files not found for dataset '${DATASET}', skipping:"
        echo "  ${DATA_PREFIX}.bin/.idx"
        continue
    fi

    echo ""
    echo "=========================================="
    echo "Training qwen (megatron) on dataset: ${DATASET}"
    echo "=========================================="
    echo "Dataset: ${DATA_PREFIX} (${DATASET})"

    if [ "$NUM_GPUS" -gt 1 ]; then
        MASTER_PORT="${MASTER_PORT:-$(python3 -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()')}"
        torchrun --nproc_per_node="$NUM_GPUS" \
                 --nnodes=1 \
                 --node_rank=0 \
                 --master_addr=localhost \
                 --master_port="$MASTER_PORT" \
                 "$SCRIPT_DIR/train_nvd_mega.py" qwen
    else
        python3 -u "$SCRIPT_DIR/train_nvd_mega.py" qwen
    fi

    echo "Completed dataset: ${DATASET}"
done

echo ""
echo "All datasets completed for qwen (megatron)."
