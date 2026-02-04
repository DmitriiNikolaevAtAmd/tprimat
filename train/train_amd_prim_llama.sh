#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TPRIMAT_PATH="$(cd "$SCRIPT_DIR/.." && pwd)"
PRIMUS_PATH="${PRIMUS_PATH:-/workspace/Primus}"

source "$TPRIMAT_PATH/config.env"

mkdir -p "$TPRIMAT_PATH/output"

# Data paths
DATA_PREFIX="${DATA_DIR}/allenai-c4-llama-mega"
TOKENIZER_PATH="${DATA_DIR}/meta-llama-llama-31-8b"

# Parallel config - llama AMD (identical_config 01)
TP=4
PP=1
DP=2
GRAD_ACCUM=64

# Training batch config
NUM_GPUS=$((TP * PP * DP))  # 8
MBS=1
GBS=$((MBS * NUM_GPUS * GRAD_ACCUM))  # 512

# Training schedule
TRAIN_ITERS=500
LR_WARMUP_ITERS=50
LR_DECAY_ITERS=$TRAIN_ITERS

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

CONFIG_FILE="examples/megatron/configs/MI300X/llama3.1_8B-BF16-pretrain.yaml"
cd "$PRIMUS_PATH"

PATCHED_CONFIG="$TPRIMAT_PATH/output/llama3.1_8B-BF16-pretrain.yaml"
cp "$PRIMUS_PATH/$CONFIG_FILE" "$PATCHED_CONFIG"

if python3 -c "import yaml" 2>/dev/null; then
    python3 -c "
import yaml
with open('$PATCHED_CONFIG', 'r') as f:
    config = yaml.safe_load(f)

config['tensor_model_parallel_size'] = 4
config['pipeline_model_parallel_size'] = 1
config['sequence_parallel'] = False
config['global_batch_size'] = 512
config['micro_batch_size'] = 1
config['seq_length'] = 2048
config['encoder_seq_length'] = 2048
config['gradient_accumulation_steps'] = 64
config['use_distributed_optimizer'] = True
config['use_flash_attn'] = True
config['use_fused_rmsnorm'] = True
config['fp32_residual_connection'] = False
config['train_iters'] = 500
config['lr_decay_iters'] = 500
config['lr_warmup_iters'] = 50

config['data_path'] = '$DATA_PREFIX'
config['tokenizer_type'] = 'HuggingFaceTokenizer'
config['tokenizer_model'] = '$TOKENIZER_PATH'
config['split'] = '100,0,0'

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

with open('$PATCHED_CONFIG', 'w') as f:
    yaml.dump(config, f)
"
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
    --train_iters 500 \
    --global_batch_size 512 \
    --micro_batch_size 1 \
    --seq_length 2048 \
    --tensor_model_parallel_size 4 \
    --pipeline_model_parallel_size 1 \
    --lr 3.0e-4 \
    --min_lr 0.0 \
    --lr_warmup_iters 50 \
    --lr_decay_style cosine \
    --lr_decay_iters 500 \
    --weight_decay 0.1 \
    --data_path "$DATA_PREFIX" \
    --tokenizer_type HuggingFaceTokenizer \
    --tokenizer_model "$TOKENIZER_PATH" \
    --split 100,0,0 \
    2>&1 | tee "$TPRIMAT_PATH/output/training_main_llama.log"

cd "$TPRIMAT_PATH"

python3 evaluate/extract_prim_metrics.py \
    --log-file "$TPRIMAT_PATH/output/training_main_llama.log" \
    --model-name "llama" \
    --output "$TPRIMAT_PATH/output/train_amd_prim_llama.json" \
    --num-gpus 8 \
    --global-batch-size 512 \
    --micro-batch-size 1 \
    --tensor-parallel-size 4 \
    --pipeline-parallel-size 1 \
    --sequence-length 2048 \
    --parallel-strategy "TP4_PP1_DP2"
