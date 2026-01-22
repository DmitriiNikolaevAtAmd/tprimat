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

cd "$PRIMUS_PATH"
export EXP="examples/megatron/configs/MI300X/llama3.1_8B-BF16-pretrain.yaml"

bash ./examples/run_pretrain.sh \
    --train_iters 10 \
    --lr 0.0003 \
    --min_lr 0.00003 \
    --lr_warmup_iters 1 \
    --lr_decay_style cosine \
    --lr_decay_iters 10 \
    --weight_decay 0.1 \
    > "$TPRIMAT_PATH/output/training_main_llama.log" 2>&1

cd "$TPRIMAT_PATH"

python3 extract_metrics.py \
    --log-file "$TPRIMAT_PATH/output/training_main_llama.log" \
    --model-name "llama" \
    --output "$TPRIMAT_PATH/output/train_primus_llama.json" \
    --num-gpus 8 \
    --global-batch-size 128 \
    --sequence-length 2048 \
    --parallel-strategy "minimal_communication"

cd "$PRIMUS_PATH"
export EXP="examples/megatron/configs/MI300X/qwen2.5_7B-BF16-pretrain.yaml"

bash ./examples/run_pretrain.sh \
    --train_iters 10 \
    --lr 0.0003 \
    --min_lr 0.00003 \
    --lr_warmup_iters 1 \
    --lr_decay_style cosine \
    --lr_decay_iters 10 \
    --weight_decay 0.1 \
    > "$TPRIMAT_PATH/output/training_main_qwen.log" 2>&1

cd "$TPRIMAT_PATH"

python3 extract_metrics.py \
    --log-file "$TPRIMAT_PATH/output/training_main_qwen.log" \
    --model-name "qwen" \
    --output "$TPRIMAT_PATH/output/train_primus_qwen.json" \
    --num-gpus 8 \
    --global-batch-size 128 \
    --sequence-length 2048 \
    --parallel-strategy "minimal_communication"
