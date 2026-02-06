#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TPRIMAT_PATH="$(cd "$SCRIPT_DIR/.." && pwd)"

source "$TPRIMAT_PATH/config.env"

mkdir -p "$TPRIMAT_PATH/output"

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

echo "Config: TP=${TP} PP=${PP} DP=${DP} GA=${GA}"
echo "Batch: MBS=${MBS} GBS=${GBS} SEQ_LEN=${SEQ_LEN}"

# Performance settings
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=ERROR
export PYTHONWARNINGS="ignore::UserWarning,ignore::FutureWarning,ignore::DeprecationWarning"
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=error
export HF_HUB_DISABLE_PROGRESS_BARS=1

# CUDA / NCCL performance tuning
export CUDA_DEVICE_MAX_CONNECTIONS=1   # enables compute/communication overlap
export CUDA_MODULE_LOADING=EAGER       # pre-load all CUDA modules at startup
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=1  # reduces NCCL memory fragmentation

# TransformerEngine: enable fused attention kernels
export NVTE_FUSED_ATTN=1
export NVTE_FLASH_ATTN=1

# FP8 hybrid (NVIDIA H100+)
export FP8_HYBRID=${FP8_HYBRID:-true}
export FP8_PARAM=${FP8_PARAM:-true}

# Profiling (from config.env)
export PROFILING=${PROFILING:-false}
export PROFILE_WAIT=${PROFILE_WAIT:-5}
export PROFILE_WARMUP=${PROFILE_WARMUP:-1}
export PROFILE_ACTIVE=${PROFILE_ACTIVE:-2}
export PROFILE_REPEAT=${PROFILE_REPEAT:-1}

# Data paths - uses DATASET from config.env (bc or c4)
DATASET="${DATASET:-bc}"
export DATASET
DATA_PREFIX="${DATA_DIR}/${DATASET}-train"

# Verify data files exist
if [ ! -f "${DATA_PREFIX}.bin" ] || [ ! -f "${DATA_PREFIX}.idx" ]; then
    echo "ERROR: Data files not found at ${DATA_PREFIX}.bin/.idx"
    echo "       Run prepare/data.sh first to generate the dataset"
    exit 1
fi

echo ""
echo "=========================================="
echo "Training llama (nemo) on dataset: ${DATASET}"
echo "=========================================="
echo "Dataset: ${DATA_PREFIX} (${DATASET})"

python3 -u "$SCRIPT_DIR/train_nvd_nemo.py" llama
