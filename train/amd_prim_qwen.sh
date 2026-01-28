#!/bin/bash
set -e
cd "$(dirname "$0")"
TPRIMAT_PATH="$(cd .. && pwd)"
source "$TPRIMAT_PATH/config.env"

PRIMUS_PATH="${PRIMUS_PATH:-/workspace/Primus}"
OUTPUT_DIR="${OUTPUT_DIR:-$TPRIMAT_PATH/output}"

mkdir -p "$OUTPUT_DIR"
if [ ! -d "$PRIMUS_PATH" ]; then
    echo "ERROR: Primus not found at $PRIMUS_PATH"
    exit 1
fi

CONFIG_FILE="examples/megatron/configs/MI300X/qwen2.5_7B-BF16-pretrain.yaml"
if [ ! -f "$PRIMUS_PATH/$CONFIG_FILE" ]; then
    echo "ERROR: Config not found at $PRIMUS_PATH/$CONFIG_FILE"
    exit 1
fi

cd "$PRIMUS_PATH"
PATCHED_CONFIG="$OUTPUT_DIR/qwen2.5_7B-BF16-pretrain.yaml"
cp "$PRIMUS_PATH/$CONFIG_FILE" "$PATCHED_CONFIG"
export PATCHED_CONFIG

if python3 -c "import yaml" 2>/dev/null; then
    python3 - <<'PY'
import os
import yaml

path = os.environ["PATCHED_CONFIG"]
config = yaml.safe_load(open(path))
config["tensor_model_parallel_size"] = int(os.environ["TP"])
config["pipeline_model_parallel_size"] = int(os.environ["PP"])
config["sequence_parallel"] = False
config["global_batch_size"] = int(os.environ["GBS"])
config["micro_batch_size"] = int(os.environ["MBS"])
config["seq_length"] = int(os.environ["SEQ_LEN"])
config["encoder_seq_length"] = int(os.environ["SEQ_LEN"])
config["gradient_accumulation_steps"] = int(os.environ["GRAD_ACCUM"])
config["train_iters"] = int(os.environ["TRAIN_ITERS"])
config["lr_decay_iters"] = int(os.environ["TRAIN_ITERS"])
config["lr_warmup_iters"] = int(os.environ["WARMUP_STEPS"])
with open(path, "w") as f:
    yaml.dump(config, f)
PY
fi

export EXP="$PATCHED_CONFIG"

TRAIN_SCRIPT="./examples/run_pretrain.sh"
if [ ! -f "$TRAIN_SCRIPT" ]; then
    TRAIN_SCRIPT="./examples/train.sh"
fi
if [ ! -f "$TRAIN_SCRIPT" ]; then
    echo "ERROR: No training script found in examples/"
    exit 1
fi

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
    --lr_decay_iters "$TRAIN_ITERS" \
    --weight_decay "$WEIGHT_DECAY" \
    2>&1 | tee "$LOG_FILE"

cd "$TPRIMAT_PATH"
NUM_GPUS=$((TP * PP * DP))
PAR_STR="TP${TP}_SP"
python3 evaluate/extract_prim_metrics.py \
    --log-file "$LOG_FILE" \
    --model-name "qwen" \
    --output "$OUTPUT_DIR/train_amd_prim_qwen.json" \
    --num-gpus "$NUM_GPUS" \
    --global-batch-size "$GBS" \
    --micro-batch-size "$MBS" \
    --tensor-parallel-size "$TP" \
    --pipeline-parallel-size "$PP" \
    --sequence-length "$SEQ_LEN" \
    --parallel-strategy "$PAR_STR"
