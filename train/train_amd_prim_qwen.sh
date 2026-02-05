#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TPRIMAT_PATH="$(cd "$SCRIPT_DIR/.." && pwd)"
PRIMUS_PATH="${PRIMUS_PATH:-/workspace/Primus}"

source "$TPRIMAT_PATH/config.env"

mkdir -p "$TPRIMAT_PATH/output"

# Data paths (disabled - using default Primus data source)
# DATA_PREFIX="${DATA_DIR}/allenai-c4-qwen-mega"
# TOKENIZER_PATH="${DATA_DIR}/qwen-qwen25-7b"

# Verify data files exist (disabled)
# if [ ! -f "${DATA_PREFIX}.bin" ] || [ ! -f "${DATA_PREFIX}.idx" ]; then
#     echo "ERROR: Data files not found at ${DATA_PREFIX}.bin/.idx"
#     echo "       Run prepare/prepare.sh first to generate the dataset"
#     exit 1
# fi
# if [ ! -d "${TOKENIZER_PATH}" ]; then
#     echo "ERROR: Tokenizer not found at ${TOKENIZER_PATH}"
#     exit 1
# fi

# Training batch config (from config.env: TP, PP, DP, GA, MBS, SEQ_LEN, etc.)
NUM_GPUS=$((TP * PP * DP))
GBS=$((MBS * NUM_GPUS * GA))
LR_DECAY_ITERS=$TRAIN_ITERS

echo "Config: TP=${TP} PP=${PP} DP=${DP} GA=${GA}"
echo "Batch: MBS=${MBS} GBS=${GBS} SEQ_LEN=${SEQ_LEN}"

# Critical AMD performance settings
export RCCL_DEBUG=ERROR
export NCCL_DEBUG=ERROR
export GLOO_LOG_LEVEL=ERROR
export RCCL_MSCCL_ENABLE=0
export HSA_NO_SCRATCH_RECLAIM=1
export HSA_ENABLE_SDMA=1
export HSA_FORCE_FINE_GRAIN_PCIE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Disable extra logging/profiling noise
export PYTHONWARNINGS="ignore::UserWarning,ignore::FutureWarning,ignore::DeprecationWarning"
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=error
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TORCH_CPP_LOG_LEVEL=ERROR
export TORCH_SHOW_CPP_STACKTRACES=0

CONFIG_FILE="examples/megatron/configs/MI300X/qwen2.5_7B-BF16-pretrain.yaml"
cd "$PRIMUS_PATH"

PATCHED_CONFIG="$TPRIMAT_PATH/output/qwen2.5_7B-BF16-pretrain.yaml"
cp "$PRIMUS_PATH/$CONFIG_FILE" "$PATCHED_CONFIG"

export PATCHED_CONFIG TP PP GBS MBS SEQ_LEN GA TRAIN_ITERS WARMUP_STEPS LR WEIGHT_DECAY
if python3 -c "import yaml" 2>/dev/null; then
    python3 << 'PYTHON_EOF'
import os
import yaml

patched_config = os.environ['PATCHED_CONFIG']
tp = int(os.environ['TP'])
pp = int(os.environ['PP'])
gbs = int(os.environ['GBS'])
mbs = int(os.environ['MBS'])
seq_len = int(os.environ['SEQ_LEN'])
grad_accum = int(os.environ['GA'])
train_iters = int(os.environ['TRAIN_ITERS'])
warmup_steps = int(os.environ['WARMUP_STEPS'])

with open(patched_config, 'r') as f:
    config = yaml.safe_load(f)

config['tensor_model_parallel_size'] = tp
config['pipeline_model_parallel_size'] = pp
config['sequence_parallel'] = False
config['global_batch_size'] = gbs
config['micro_batch_size'] = mbs
config['seq_length'] = seq_len
config['encoder_seq_length'] = seq_len
config['gradient_accumulation_steps'] = grad_accum
config['use_distributed_optimizer'] = True
config['use_flash_attn'] = True
config['use_fused_rmsnorm'] = True
config['fp32_residual_connection'] = False
config['train_iters'] = train_iters
config['lr_decay_iters'] = train_iters
config['lr_warmup_iters'] = warmup_steps

# External data configuration (disabled - using default Primus data source)
# config['data_path'] = data_prefix
# config['tokenizer_type'] = 'HuggingFaceTokenizer'
# config['tokenizer_model'] = tokenizer_path
# config['split'] = '100,0,0'

config['disable_tensorboard'] = True
config['disable_wandb'] = True
config['disable_mlflow'] = True
config['log_interval'] = 0
config['log_timers_to_tensorboard'] = False
config['log_throughput'] = False
config['log_memory_to_tensorboard'] = False
config['log_learning_rate_to_tensorboard'] = False
config['log_loss_scale_to_tensorboard'] = False
config['profile'] = False
config['use_pytorch_profiler'] = False
config['torch_profiler_with_stack'] = False
config['torch_profiler_record_shapes'] = False
config['torch_profiler_use_gzip'] = False

with open(patched_config, 'w') as f:
    yaml.dump(config, f)
PYTHON_EOF
    echo "Patched config written to: $PATCHED_CONFIG"
else
    echo "WARNING: pyyaml not available, using unpatched config"
fi

export EXP="$PATCHED_CONFIG"

TRAIN_SCRIPT="./examples/run_pretrain.sh"
if [ ! -f "$TRAIN_SCRIPT" ]; then
    TRAIN_SCRIPT="./examples/train.sh"
fi

if [ ! -f "$TRAIN_SCRIPT" ]; then
    echo "ERROR: Neither run_pretrain.sh nor train.sh found in examples/"
    exit 1
fi

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
    2>&1 | tee "$TPRIMAT_PATH/output/training_main_qwen.log"

# External data CLI args (disabled)
#     --data_path "$DATA_PREFIX" \
#     --tokenizer_type HuggingFaceTokenizer \
#     --tokenizer_model "$TOKENIZER_PATH" \
#     --split 100,0,0 \

cd "$TPRIMAT_PATH"

python3 evaluate/extract_prim_metrics.py \
    --log-file "$TPRIMAT_PATH/output/training_main_qwen.log" \
    --model-name "qwen" \
    --output "$TPRIMAT_PATH/output/train_amd_prim_qwen.json" \
    --num-gpus "$NUM_GPUS" \
    --global-batch-size "$GBS" \
    --micro-batch-size "$MBS" \
    --tensor-parallel-size "$TP" \
    --pipeline-parallel-size "$PP" \
    --sequence-length "$SEQ_LEN" \
    --parallel-strategy "TP${TP}_PP${PP}_DP${DP}"
