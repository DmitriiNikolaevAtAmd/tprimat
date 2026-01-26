#!/bin/bash
set -e
TPRIMAT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -f "$TPRIMAT_PATH/config_to_shell.py" ]; then
    eval "$(python3 "$TPRIMAT_PATH/config_to_shell.py")"
else
fi

PRIMUS_PATH="${PRIMUS_PATH:-${CONFIG_PRIMUS_PATH:-/workspace/Primus}}"
MODEL="qwen"
CONFIG_FILE="${CONFIG_QWEN_PRIMUS_CONFIG:-examples/megatron/configs/MI300X/qwen2.5_7B-BF16-pretrain.yaml}"
TRAIN_ITERS="${TRAIN_ITERS:-${CONFIG_TRAIN_ITERS:-50}}"
OUTPUT_DIR="${CONFIG_OUTPUT_DIR:-$TPRIMAT_PATH/output}"
if [[ "$OUTPUT_DIR" != /* ]]; then
    OUTPUT_DIR="$TPRIMAT_PATH/$OUTPUT_DIR"
fi
OUTPUT_DIR="$(cd "$(dirname "$OUTPUT_DIR")" 2>/dev/null && pwd)/$(basename "$OUTPUT_DIR")" || OUTPUT_DIR="$TPRIMAT_PATH/output"
NUM_GPUS="${CONFIG_AMD_NUM_GPUS:-8}"
GLOBAL_BATCH_SIZE="${CONFIG_GLOBAL_BATCH_SIZE:-128}"
SEQ_LENGTH="${CONFIG_SEQ_LENGTH:-2048}"

TP="${CONFIG_QWEN_AMD_TP:-1}"
PP="${CONFIG_QWEN_AMD_PP:-1}"
GACC="${CONFIG_QWEN_AMD_GACC:-16}"


ACT_CHECKPOINT="${CONFIG_AMD_ACT_CHECKPOINT:-false}"
export PYTORCH_ALLOC_CONF='expandable_segments:True'

NLAYERS=32
if [[ "$MODEL" == "qwen" ]]; then NLAYERS=28; fi

LEARNING_RATE="${CONFIG_LEARNING_RATE:-3.0e-4}"
MIN_LEARNING_RATE="${CONFIG_MIN_LEARNING_RATE:-3.0e-5}"
WARMUP_STEPS="${CONFIG_WARMUP_STEPS:-10}"
WEIGHT_DECAY="${CONFIG_WEIGHT_DECAY:-0.1}"
ACT_CHECKPOINT="${CONFIG_AMD_ACT_CHECKPOINT:-false}"
mkdir -p "$OUTPUT_DIR"
export HF_HOME="./cache"
mkdir -p "$HF_HOME"
if [ ! -d "$PRIMUS_PATH" ]; then
    exit 1
fi

if [ ! -f "$PRIMUS_PATH/$CONFIG_FILE" ]; then
    ls -1 "$PRIMUS_PATH/examples/megatron/configs/MI300X/" 2>/dev/null | grep -i qwen || echo "  (none found)"
    exit 1
fi


LOG_FILE="${LOG_FILE:-$(cd "$OUTPUT_DIR" && pwd)/training_main_${MODEL}.log}"
BACKUP_LOG="${BACKUP_LOG:-$(cd "$OUTPUT_DIR" && pwd)/primus_training_${MODEL}.log}"


cd "$PRIMUS_PATH"

PATCHED_CONFIG="$OUTPUT_DIR/$(basename "$CONFIG_FILE")"
cp "$PRIMUS_PATH/$CONFIG_FILE" "$PATCHED_CONFIG"

if python3 -c "import yaml" 2>/dev/null; then
    python3 -c "
import yaml
with open('$PATCHED_CONFIG', 'r') as f:
    config = yaml.safe_load(f)
config['tensor_model_parallel_size'] = $TP
config['pipeline_model_parallel_size'] = $PP
if 'gradient_accumulation_steps' in config:
    config['gradient_accumulation_steps'] = $GACC

if '${ACT_CHECKPOINT}' == 'true':
    config['recompute_activations'] = True
    config['recompute_granularity'] = 'full'
    config['recompute_method'] = 'uniform'
    config['recompute_num_layers'] = $NLAYERS
    
config['use_distributed_optimizer'] = True
config['use_flash_attn'] = True
config['use_fused_rmsnorm'] = True
config['fp32_residual_connection'] = False

config['log_memory_usage'] = True
config['log_interval'] = 1

config['fp8'] = 'hybrid'
config['fp8_param'] = True

with open('$PATCHED_CONFIG', 'w') as f:
    yaml.dump(config, f)
"
else
    sed "s/tensor_model_parallel_size:.*/tensor_model_parallel_size: $TP/" "$PATCHED_CONFIG" > "$PATCHED_CONFIG.tmp" && mv "$PATCHED_CONFIG.tmp" "$PATCHED_CONFIG"
    sed "s/pipeline_model_parallel_size:.*/pipeline_model_parallel_size: $PP/" "$PATCHED_CONFIG" > "$PATCHED_CONFIG.tmp" && mv "$PATCHED_CONFIG.tmp" "$PATCHED_CONFIG"
    sed "s/gradient_accumulation_steps:.*/gradient_accumulation_steps: $GACC/" "$PATCHED_CONFIG" > "$PATCHED_CONFIG.tmp" && mv "$PATCHED_CONFIG.tmp" "$PATCHED_CONFIG"
fi

export EXP="$PATCHED_CONFIG"


bash ./examples/train.sh \
    --train_iters $TRAIN_ITERS \
    --lr $LEARNING_RATE \
    --min_lr $MIN_LEARNING_RATE \
    --lr_warmup_iters $WARMUP_STEPS \
    --lr_decay_style cosine \
    --lr_decay_iters $TRAIN_ITERS \
    --weight_decay $WEIGHT_DECAY \
    2>&1 | tee "$LOG_FILE" "$BACKUP_LOG" > /dev/null
EXIT_CODE=${PIPESTATUS[0]}



if [ $EXIT_CODE -eq 0 ]; then
    
    
    cd "$TPRIMAT_PATH"
    
    PARALLEL_STRATEGY="${PARALLEL:-unknown}"
    
    python3 extract_metrics.py \
        --log-file "$LOG_FILE" \
        --model-name "$MODEL" \
        --output "$OUTPUT_DIR/train_amd_prim_${MODEL}.json" \
        --num-gpus "$NUM_GPUS" \
        --global-batch-size "$GLOBAL_BATCH_SIZE" \
        --sequence-length "$SEQ_LENGTH" \
        --parallel-strategy "$PARALLEL_STRATEGY"
    
    EXTRACT_EXIT=$?
    if [ $EXTRACT_EXIT -eq 0 ]; then
        
        


    else
        EXIT_CODE=$EXTRACT_EXIT
    fi
else
fi


exit $EXIT_CODE
