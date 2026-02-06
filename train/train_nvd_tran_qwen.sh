#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TPRIMAT_PATH="$(cd "$SCRIPT_DIR/.." && pwd)"

source "$TPRIMAT_PATH/config.env"

mkdir -p "$TPRIMAT_PATH/output"

# Data paths - uses DATASET from config.env (bc or c4)
DATASET="${DATASET:-bc}"
DATA_PREFIX="${DATA_DIR}/${DATASET}-train"

# Verify data files exist
if [ ! -f "${DATA_PREFIX}.bin" ] || [ ! -f "${DATA_PREFIX}.idx" ]; then
    echo "ERROR: Data files not found at ${DATA_PREFIX}.bin/.idx"
    echo "       Run prepare/data.sh first to generate the dataset"
    exit 1
fi

NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
export OUTPUT_DIR="${OUTPUT_DIR:-$TPRIMAT_PATH/output}"
export DATA_DIR
export DATASET

echo "Config: NUM_GPUS=${NUM_GPUS} GA=${GA}"
echo "Batch: MBS=${MBS} SEQ_LEN=${SEQ_LEN}"
echo "Dataset: ${DATA_PREFIX} (${DATASET})"

# Performance settings
export PYTORCH_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=ERROR
export PYTHONWARNINGS="ignore::UserWarning,ignore::FutureWarning,ignore::DeprecationWarning"
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=error
export HF_HUB_DISABLE_PROGRESS_BARS=1
export USE_TF=NO
export USE_APEX=NO
export TRANSFORMERS_NO_APEX=1

if [ "$NUM_GPUS" -gt 1 ]; then
    torchrun --nproc_per_node="$NUM_GPUS" \
             --nnodes=1 \
             --node_rank=0 \
             --master_addr=localhost \
             --master_port=29500 \
             "$SCRIPT_DIR/train_nvd_tran.py" qwen
else
    python3 -u "$SCRIPT_DIR/train_nvd_tran.py" qwen
fi
