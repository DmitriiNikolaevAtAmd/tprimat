#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TPRIMAT_PATH="$(cd "$SCRIPT_DIR/.." && pwd)"
ROOT_DIR="$TPRIMAT_PATH"
PRIMUS_PATH="${PRIMUS_PATH:-/workspace/Primus}"

# Source config.env if it exists
if [ -f "$TPRIMAT_PATH/config.env" ]; then
    set -a
    source "$TPRIMAT_PATH/config.env"
    set +a
fi

mkdir -p "$TPRIMAT_PATH/output"

# Training batch config
NUM_GPUS="${NUM_GPUS:-8}"
GBS="${GBS:-64}"
MBS="${MBS:-1}"
GRAD_ACCUM=$((GBS / (MBS * NUM_GPUS)))

# Training schedule
TRAIN_ITERS="${TRAIN_ITERS:-500}"
LR_WARMUP_ITERS="${LR_WARMUP_ITERS:-50}"
LR_DECAY_ITERS="${LR_DECAY_ITERS:-$TRAIN_ITERS}"

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

config['tensor_model_parallel_size'] = 2
config['pipeline_model_parallel_size'] = 1
config['sequence_parallel'] = True
config['global_batch_size'] = int('$GBS')
config['micro_batch_size'] = int('$MBS')
config['seq_length'] = 2048
config['encoder_seq_length'] = 2048
# grad_accum = GBS / (MBS * num_gpus)
config['gradient_accumulation_steps'] = int('$GRAD_ACCUM')
config['use_distributed_optimizer'] = True
config['use_flash_attn'] = True
config['use_fused_rmsnorm'] = True
config['fp32_residual_connection'] = False
# Training schedule
config['train_iters'] = int('$TRAIN_ITERS')
config['lr_decay_iters'] = int('$LR_DECAY_ITERS')
config['lr_warmup_iters'] = int('$LR_WARMUP_ITERS')

# Disable logging/profiling
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
    --train_iters "$TRAIN_ITERS" \
    --global_batch_size "$GBS" \
    --micro_batch_size "$MBS" \
    --seq_length 2048 \
    --tensor_model_parallel_size 2 \
    --pipeline_model_parallel_size 1 \
    --lr 3.0e-4 \
    --min_lr 0.0 \
    --lr_warmup_iters "$LR_WARMUP_ITERS" \
    --lr_decay_style cosine \
    --lr_decay_iters "$LR_DECAY_ITERS" \
    --weight_decay 0.1 \
    2>&1 | tee "$TPRIMAT_PATH/output/training_main_llama.log"

cd "$TPRIMAT_PATH"

python3 evaluate/extract_prim_metrics.py \
    --log-file "$TPRIMAT_PATH/output/training_main_llama.log" \
    --model-name "llama" \
    --output "$TPRIMAT_PATH/output/train_amd_prim_llama.json" \
    --num-gpus "$NUM_GPUS" \
    --global-batch-size "$GBS" \
    --micro-batch-size "$MBS" \
    --tensor-parallel-size 2 \
    --pipeline-parallel-size 1 \
    --sequence-length 2048 \
    --parallel-strategy "TP1_SP"