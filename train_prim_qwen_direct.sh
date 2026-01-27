#!/bin/bash
# Direct Primus invocation bypassing run_pretrain.sh wrapper
set -e

echo "╔════════════════════════════════════════════════════════════╗"
echo "║     Primus Training: Qwen 2.5 7B (Direct Invocation)     ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

TPRIMAT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRIMUS_PATH="${PRIMUS_PATH:-/workspace/Primus}"
OUTPUT_DIR="${TPRIMAT_PATH}/output"
mkdir -p "$OUTPUT_DIR"

MODEL="qwen"
NUM_GPUS=8
TRAIN_ITERS=10
LR=3.0e-4
MIN_LR=3.0e-5
WARMUP_ITERS=2
DECAY_ITERS=10

echo "  * Primus Path: $PRIMUS_PATH"
echo "  * Output: $OUTPUT_DIR"
echo "  * Training Iterations: $TRAIN_ITERS"
echo "  * LR Warmup: $WARMUP_ITERS, Decay: $DECAY_ITERS"
echo ""

cd "$PRIMUS_PATH"

torchrun \
    --nproc_per_node=$NUM_GPUS \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port=29500 \
    primus/cli/main.py train \
    --config examples/megatron/configs/MI300X/qwen2.5_7B-BF16-pretrain.yaml \
    --train_iters $TRAIN_ITERS \
    --lr $LR \
    --min_lr $MIN_LR \
    --lr_warmup_iters $WARMUP_ITERS \
    --lr_decay_iters $DECAY_ITERS \
    --weight_decay 0.1 \
    2>&1 | tee "$OUTPUT_DIR/training_direct_qwen.log"

echo ""
echo "Done. Log saved to: $OUTPUT_DIR/training_direct_qwen.log"
