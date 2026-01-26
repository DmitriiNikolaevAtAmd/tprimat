#!/bin/bash
set -e
TPRIMAT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRIMUS_PATH="${PRIMUS_PATH:-/workspace/Primus}"

export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
export PYTHONHASHSEED="42"
export HSA_NO_SCRATCH_RECLAIM=1
export HSA_ENABLE_SDMA=1
export HSA_FORCE_FINE_GRAIN_PCIE=1
export RCCL_DEBUG=INFO
export NCCL_DEBUG=INFO
mkdir -p "$TPRIMAT_PATH/output"
if [ ! -d "$PRIMUS_PATH" ]; then
    echo "ERROR: Primus directory not found at: $PRIMUS_PATH"
    echo "Please set PRIMUS_PATH environment variable or ensure /workspace/Primus exists"
    exit 1
fi
CONFIG_FILE="examples/megatron/configs/MI300X/qwen2.5_7B-BF16-pretrain.yaml"
if [ ! -f "$PRIMUS_PATH/$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found at: $PRIMUS_PATH/$CONFIG_FILE"
    echo "Available configs:"
    ls -1 "$PRIMUS_PATH/examples/megatron/configs/MI300X/" 2>/dev/null | grep -i qwen || echo "  (none found)"
    exit 1
fi

cd "$PRIMUS_PATH"
export EXP="$CONFIG_FILE"

bash ./examples/train.sh \
    --train_iters 50 \
    --lr 0.0003 \
    --min_lr 0.00003 \
    --lr_warmup_iters 10 \
    --lr_decay_style cosine \
    --lr_decay_iters 50 \
    --weight_decay 0.1 \
    > "$TPRIMAT_PATH/output/training_main_qwen.log" 2>&1

cd "$TPRIMAT_PATH"

python3 extract_metrics.py \
    --log-file "$TPRIMAT_PATH/output/training_main_qwen.log" \
    --model-name "qwen" \
    --output "$TPRIMAT_PATH/output/train_amd_prim_qwen.json" \
    --num-gpus 8 \
    --global-batch-size 128 \
    --sequence-length 2048 \
    --parallel-strategy "minimal_communication"
