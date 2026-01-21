#!/bin/bash
# Minimal Primus training script

set -e

# === Configuration ===
PRIMUS_PATH="${PRIMUS_PATH:-/workspace/Primus}"
MODEL="${1:-llama}"  # llama or qwen
TRAIN_ITERS="${2:-500}"

# Model configs
if [ "$MODEL" = "llama" ]; then
    CONFIG="examples/megatron/configs/MI300X/llama3.1_8B-BF16-pretrain.yaml"
elif [ "$MODEL" = "qwen" ]; then
    CONFIG="examples/megatron/configs/MI300X/qwen2.5_7B-BF16-pretrain.yaml"
else
    echo "Usage: $0 [llama|qwen] [train_iters]"
    exit 1
fi

# Training params
NUM_GPUS=8
TP=1          # Tensor parallel
PP=1          # Pipeline parallel
GBS=128       # Global batch size
SEQ_LEN=2048
LR=3.0e-4
MIN_LR=3.0e-5
WARMUP=50
WEIGHT_DECAY=0.1

# Output
OUTPUT_DIR="$(pwd)/output"
mkdir -p "$OUTPUT_DIR"
LOG_FILE="$OUTPUT_DIR/training_${MODEL}_minimal.log"

# === Validation ===
if [ ! -d "$PRIMUS_PATH" ]; then
    echo "Error: Primus not found at $PRIMUS_PATH"
    echo "Set PRIMUS_PATH: export PRIMUS_PATH=/path/to/Primus"
    exit 1
fi

if [ ! -f "$PRIMUS_PATH/$CONFIG" ]; then
    echo "Error: Config not found: $PRIMUS_PATH/$CONFIG"
    exit 1
fi

# === Run Training ===
echo "[*] Starting Primus training: $MODEL"
echo "    Config: $CONFIG"
echo "    Steps: $TRAIN_ITERS"
echo "    GPUs: $NUM_GPUS (TP=$TP, PP=$PP)"
echo "    Log: $LOG_FILE"
echo ""

cd "$PRIMUS_PATH"

bash ./examples/run_pretrain.sh \
    --train_iters "$TRAIN_ITERS" \
    --lr "$LR" \
    --min_lr "$MIN_LR" \
    --lr_warmup_iters "$WARMUP" \
    --weight_decay "$WEIGHT_DECAY" \
    2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "[+] Training completed successfully"
else
    echo ""
    echo "[x] Training failed with exit code $EXIT_CODE"
fi

exit $EXIT_CODE
