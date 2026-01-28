#!/bin/bash
set -e
cd "$(dirname "$0")"
TPRIMAT_PATH="$(cd .. && pwd)"
set -a
source "$TPRIMAT_PATH/config.env"
set +a

PRIMUS_PATH="${PRIMUS_PATH:-/workspace/Primus}"
OUTPUT_DIR="${OUTPUT_DIR:-$TPRIMAT_PATH/output}"

if [ -z "$HF_TOKEN" ] && [ -z "$HUGGINGFACE_HUB_TOKEN" ]; then
    echo "ERROR: HF_TOKEN is required for Llama 3.1."
    exit 1
fi
export HF_TOKEN="${HF_TOKEN:-$HUGGINGFACE_HUB_TOKEN}"
export HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN:-$HF_TOKEN}"

mkdir -p "$OUTPUT_DIR"
if [ ! -d "$PRIMUS_PATH" ]; then
    echo "ERROR: Primus not found at $PRIMUS_PATH"
    exit 1
fi

CONFIG_FILE="examples/megatron/configs/MI300X/llama3.1_8B-BF16-pretrain.yaml"
if [ ! -f "$PRIMUS_PATH/$CONFIG_FILE" ]; then
    echo "ERROR: Config not found at $PRIMUS_PATH/$CONFIG_FILE"
    exit 1
fi

cd "$PRIMUS_PATH"
PATCHED_CONFIG="$OUTPUT_DIR/llama3.1_8B-BF16-pretrain.yaml"
cp "$PRIMUS_PATH/$CONFIG_FILE" "$PATCHED_CONFIG"
export PATCHED_CONFIG

if python3 -c "import yaml" 2>/dev/null; then
    python3 - <<'PY'
import os
import yaml

path = os.environ["PATCHED_CONFIG"]
data_dir = os.environ.get("DATA_DIR", "/data")
config = yaml.safe_load(open(path))

# Get the overrides section
overrides = config.get("modules", {}).get("pre_trainer", {}).get("overrides", {})

overrides["tensor_model_parallel_size"] = int(os.environ["TP"])
overrides["pipeline_model_parallel_size"] = int(os.environ["PP"])
overrides["global_batch_size"] = int(os.environ["GBS"])
overrides["micro_batch_size"] = int(os.environ["MBS"])
overrides["seq_length"] = int(os.environ["SEQ_LEN"])
overrides["max_position_embeddings"] = int(os.environ["SEQ_LEN"])

train_iters = int(os.environ["TRAIN_ITERS"])
warmup_steps = int(os.environ["WARMUP_STEPS"])

# Megatron requires warmup_steps < lr_decay_steps
if warmup_steps >= train_iters:
    warmup_steps = train_iters // 2

overrides["train_iters"] = train_iters
overrides["lr_decay_iters"] = train_iters
overrides["lr_warmup_iters"] = warmup_steps

# Use local data instead of mock data
# Note: Primus/Megatron requires the correct indexed dataset format (mega format)
overrides["mock_data"] = False
data_prefix = f"{data_dir}/allenai-c4-llama-mega"
overrides["train_data_path"] = data_prefix
overrides["valid_data_path"] = data_prefix
overrides["test_data_path"] = data_prefix
# Fallback keys used in some Megatron configs
overrides["data_path"] = data_prefix
overrides["data_prefix"] = data_prefix

print("[config-check] data overrides:", {
    k: overrides.get(k)
    for k in (
        "train_data_path",
        "valid_data_path",
        "test_data_path",
        "data_path",
        "data_prefix",
    )
    if k in overrides
})

with open(path, "w") as f:
    yaml.dump(config, f, default_flow_style=False)
PY
fi

export EXP="$PATCHED_CONFIG"

DATASET_PREFIX="${DATA_DIR}/allenai-c4-llama-mega"
export DATASET_PREFIX
python3 - <<'PY'
import os
import struct
from pathlib import Path

prefix = os.environ["DATASET_PREFIX"]
idx_path = Path(f"{prefix}.idx")
if not idx_path.exists():
    raise SystemExit(f"ERROR: Missing dataset index: {idx_path}")

with open(idx_path, "rb") as f:
    magic = f.read(9)
    if magic != b"MMIDIDX\x00\x00":
        raise SystemExit(f"ERROR: Bad idx magic: {magic}")
    _version = struct.unpack("<Q", f.read(8))[0]
    _dtype = struct.unpack("<B", f.read(1))[0]
    num_seqs = struct.unpack("<Q", f.read(8))[0]
    num_docs = struct.unpack("<Q", f.read(8))[0]
    doc_indices = f.read((num_docs + 1) * 8)
    f.read(num_seqs * 8)          # pointers
    f.read(num_seqs * 4)          # lengths
    last_doc = struct.unpack("<Q", doc_indices[-8:])[0] if doc_indices else 0

print(f"[data-check] {prefix}: num_seqs={num_seqs}, num_docs={num_docs}, doc_indices[-1]={last_doc}")
if last_doc != num_seqs:
    raise SystemExit(
        "ERROR: Mega idx mismatch: document_indices[-1] "
        f"({last_doc}) != num_seqs ({num_seqs}). Re-encode Mega data."
    )
PY

TRAIN_SCRIPT="./examples/run_pretrain.sh"
if [ ! -f "$TRAIN_SCRIPT" ]; then
    TRAIN_SCRIPT="./examples/train.sh"
fi
if [ ! -f "$TRAIN_SCRIPT" ]; then
    echo "ERROR: No training script found in examples/"
    exit 1
fi

LOG_FILE="$OUTPUT_DIR/training_main_llama.log"
# Calculate safe warmup steps
SAFE_WARMUP=$WARMUP_STEPS
if [ "$SAFE_WARMUP" -ge "$TRAIN_ITERS" ]; then
    SAFE_WARMUP=$((TRAIN_ITERS / 2))
    echo "Warning: Capping WARMUP_STEPS to $SAFE_WARMUP because TRAIN_ITERS is $TRAIN_ITERS"
fi

export SAFE_WARMUP

bash "$TRAIN_SCRIPT" \
    --train_iters "$TRAIN_ITERS" \
    --global_batch_size "$GBS" \
    --micro_batch_size "$MBS" \
    --seq_length "$SEQ_LEN" \
    --tensor_model_parallel_size "$TP" \
    --pipeline_model_parallel_size "$PP" \
    --lr "$LR" \
    --min_lr 0.0 \
    --lr_warmup_iters "$SAFE_WARMUP" \
    --lr_decay_style cosine \
    --lr_decay_iters "$TRAIN_ITERS" \
    --weight_decay "$WEIGHT_DECAY" \
    2>&1 | tee "$LOG_FILE"

cd "$TPRIMAT_PATH"
NUM_GPUS=$((TP * PP * DP))
PAR_STR="TP${TP}_SP"
python3 evaluate/extract_prim_metrics.py \
    --log-file "$LOG_FILE" \
    --model-name "llama" \
    --output "$OUTPUT_DIR/train_amd_prim_llama.json" \
    --num-gpus "$NUM_GPUS" \
    --global-batch-size "$GBS" \
    --micro-batch-size "$MBS" \
    --tensor-parallel-size "$TP" \
    --pipeline-parallel-size "$PP" \
    --sequence-length "$SEQ_LEN" \
    --parallel-strategy "$PAR_STR"
