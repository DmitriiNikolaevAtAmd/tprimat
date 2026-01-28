#!/bin/bash
set -e
cd "$(dirname "$0")"
TPRIMAT_PATH="$(cd .. && pwd)"
set -a
source "$TPRIMAT_PATH/config.env"
set +a

PRIMUS_PATH="${PRIMUS_PATH:-/workspace/Primus}"
OUTPUT_DIR="${OUTPUT_DIR:-$TPRIMAT_PATH/output}"
mkdir -p "$OUTPUT_DIR"

CONFIG_FILE="examples/megatron/configs/MI300X/qwen2.5_7B-BF16-pretrain.yaml"
cd "$PRIMUS_PATH"

PATCHED_CONFIG="$OUTPUT_DIR/qwen2.5_7B-BF16-pretrain.yaml"
cp "$CONFIG_FILE" "$PATCHED_CONFIG"

python3 - <<'PY'
import os, yaml

config = yaml.safe_load(open(os.environ["PATCHED_CONFIG"]))
overrides = config.setdefault("modules", {}).setdefault("pre_trainer", {}).setdefault("overrides", {})

env = os.environ
overrides.update({
    "tensor_model_parallel_size": int(env["TP"]),
    "pipeline_model_parallel_size": int(env["PP"]),
    "global_batch_size": int(env["GBS"]),
    "micro_batch_size": int(env["MBS"]),
    "seq_length": int(env["SEQ_LEN"]),
    "max_position_embeddings": int(env["SEQ_LEN"]),
    "train_iters": int(env["TRAIN_ITERS"]),
    "lr_decay_iters": int(env["LR_DECAY_STEPS"]),
    "lr_warmup_iters": min(int(env["WARMUP_STEPS"]), int(env["TRAIN_ITERS"]) // 2),
    "mock_data": True
})

yaml.dump(config, open(os.environ["PATCHED_CONFIG"], "w"), default_flow_style=False)
PY

export EXP="$PATCHED_CONFIG"
export SKIP_PREPARE=1

TRAIN_SCRIPT="./examples/train.sh"
[ -f "$TRAIN_SCRIPT" ] || TRAIN_SCRIPT="./examples/run_pretrain.sh"

LOG_FILE="$OUTPUT_DIR/training_main_qwen.log"
bash "$TRAIN_SCRIPT" \
    --train_iters "$TRAIN_ITERS" \
    --global_batch_size "$GBS" \
    --micro_batch_size "$MBS" \
    --seq_length "$SEQ_LEN" \
    --tensor_model_parallel_size "$TP" \
    --pipeline_model_parallel_size "$PP" \
    --lr "$LR" \
    --min_lr 0.0 \
    --lr_warmup_iters "$WARMUP_STEPS" \
    --lr_decay_style cosine \
    --lr_decay_iters "$LR_DECAY_STEPS" \
    --weight_decay "$WEIGHT_DECAY" \
    2>&1 | tee "$LOG_FILE"

cd "$TPRIMAT_PATH"
python3 evaluate/extract_prim_metrics.py \
    --log-file "$LOG_FILE" \
    --model-name "qwen" \
    --output "$OUTPUT_DIR/train_amd_prim_qwen.json" \
    --num-gpus $((TP * PP * DP)) \
    --global-batch-size "$GBS" \
    --micro-batch-size "$MBS" \
    --tensor-parallel-size "$TP" \
    --pipeline-parallel-size "$PP" \
    --sequence-length "$SEQ_LEN" \
    --parallel-strategy "TP${TP}_SP"
