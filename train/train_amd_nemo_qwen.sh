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

export OUTPUT_DIR="${OUTPUT_DIR:-$TPRIMAT_PATH/output}"
export DATA_DIR
export DATASET

# Parallel config
export TP=${TP:-1}
export PP=${PP:-1}
export DP=${DP:-8}
export GA=${GA:-64}

# Batch config
export MBS=${MBS:-1}
export GBS=$((MBS * DP * GA))

echo "Config: TP=${TP} PP=${PP} DP=${DP} GA=${GA}"
echo "Batch: MBS=${MBS} GBS=${GBS} SEQ_LEN=${SEQ_LEN}"
echo "Dataset: ${DATA_PREFIX} (${DATASET})"

# AMD ROCm performance settings
export PYTORCH_ALLOC_CONF=expandable_segments:True
export RCCL_DEBUG=ERROR
export NCCL_DEBUG=ERROR
export GLOO_LOG_LEVEL=ERROR
export RCCL_MSCCL_ENABLE=0
export HSA_NO_SCRATCH_RECLAIM=1
export HSA_ENABLE_SDMA=1
export HSA_FORCE_FINE_GRAIN_PCIE=1
export PYTHONWARNINGS="ignore::UserWarning,ignore::FutureWarning,ignore::DeprecationWarning"
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=error
export HF_HUB_DISABLE_PROGRESS_BARS=1

# Profiling (from config.env)
export PROFILING=${PROFILING:-false}
export PROFILE_WAIT=${PROFILE_WAIT:-5}
export PROFILE_WARMUP=${PROFILE_WARMUP:-1}
export PROFILE_ACTIVE=${PROFILE_ACTIVE:-2}
export PROFILE_REPEAT=${PROFILE_REPEAT:-1}

python3 -u "$SCRIPT_DIR/train_amd_nemo.py" qwen
