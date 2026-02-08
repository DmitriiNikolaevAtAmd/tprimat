#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TPRIMAT_PATH="$(cd "$SCRIPT_DIR/.." && pwd)"
PRIMUS_PATH="${PRIMUS_PATH:-/workspace/Primus}"

source "$TPRIMAT_PATH/config.env"

mkdir -p "$TPRIMAT_PATH/output"

TOKENIZER_PATH="${DATA_DIR}/tokenizers/llama"
TOKENIZER_HF="meta-llama/Llama-3.1-8B"

if [ -d "${TOKENIZER_PATH}" ]; then
    TOKENIZER_MODEL="${TOKENIZER_PATH}"
    echo "Using local tokenizer: ${TOKENIZER_MODEL}"
else
    TOKENIZER_MODEL="${TOKENIZER_HF}"
    echo "Using HuggingFace tokenizer: ${TOKENIZER_MODEL}"
fi

NUM_GPUS=$((TP * PP * DP))
GBS=$((MBS * DP * GA))
LR_DECAY_ITERS=$TRAIN_ITERS

echo "Config: TP=${TP} PP=${PP} DP=${DP} GA=${GA}"
echo "Batch: MBS=${MBS} GBS=${GBS} SL=${SL}"
echo "Seed: ${SEED}"

export RCCL_DEBUG=ERROR
export NCCL_DEBUG=ERROR
export GLOO_LOG_LEVEL=ERROR
export RCCL_MSCCL_ENABLE=0
export HSA_NO_SCRATCH_RECLAIM=1
export HSA_ENABLE_SDMA=1
export HSA_FORCE_FINE_GRAIN_PCIE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Maximum performance: ROCm-optimized GEMM and HIP kernels
export TORCH_BLAS_PREFER_HIPBLASLT=1
export HIPBLASLT_FORCE_REDUCE_SCATTERING=1
export GPU_MAX_HW_QUEUES=2
export ROCM_FORCE_HIGH_PERF=1

export PYTHONWARNINGS="ignore::UserWarning,ignore::FutureWarning,ignore::DeprecationWarning"
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=error
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TORCH_CPP_LOG_LEVEL=ERROR
export TORCH_SHOW_CPP_STACKTRACES=0

python3 -c "import site; open(site.getsitepackages()[0] + '/primus.pth', 'w').write('$PRIMUS_PATH')" 2>/dev/null || true
export PYTHONPATH="$PRIMUS_PATH:${PYTHONPATH:-}"

CONFIG_FILE="examples/megatron/configs/MI300X/llama3.1_8B-BF16-pretrain.yaml"

TRAIN_SCRIPT="./examples/run_pretrain.sh"
if [ ! -f "$PRIMUS_PATH/$TRAIN_SCRIPT" ]; then
    TRAIN_SCRIPT="./examples/train.sh"
fi

if [ ! -f "$PRIMUS_PATH/$TRAIN_SCRIPT" ]; then
    echo "ERROR: Neither run_pretrain.sh nor train.sh found in $PRIMUS_PATH/examples/"
    exit 1
fi

DATASET="${DATASET:-bc}"
DATA_PREFIX="${DATA_DIR}/${DATASET}-train"

if [ ! -f "${DATA_PREFIX}.bin" ] || [ ! -f "${DATA_PREFIX}.idx" ]; then
    echo "ERROR: Data files not found at ${DATA_PREFIX}.bin/.idx"
    echo "       Run scripts/prepare.sh first to generate the dataset"
    exit 1
fi

echo ""
echo "=========================================="
echo "Training llama (prim) on dataset: ${DATASET}"
echo "=========================================="
echo "Dataset: ${DATA_PREFIX} (${DATASET})"

cd "$PRIMUS_PATH"

PATCHED_CONFIG="$TPRIMAT_PATH/output/llama3.1_8B-BF16-pretrain.yaml"
cp "$PRIMUS_PATH/$CONFIG_FILE" "$PATCHED_CONFIG"

export PATCHED_CONFIG TP PP GBS MBS SL GA TRAIN_ITERS WARMUP_STEPS LR WEIGHT_DECAY BETA1 BETA2 DATA_PREFIX TOKENIZER_MODEL SEED
if python3 -c "import yaml" 2>/dev/null; then
    python3 << 'PYTHON_EOF'
import os
import yaml

patched_config = os.environ['PATCHED_CONFIG']
tp = int(os.environ['TP'])
pp = int(os.environ['PP'])
gbs = int(os.environ['GBS'])
mbs = int(os.environ['MBS'])
seq_len = int(os.environ['SL'])
grad_accum = int(os.environ['GA'])
train_iters = int(os.environ['TRAIN_ITERS'])
warmup_steps = int(os.environ['WARMUP_STEPS'])
beta1 = float(os.environ.get('BETA1', '0.9'))
beta2 = float(os.environ.get('BETA2', '0.95'))

with open(patched_config, 'r') as f:
    config = yaml.safe_load(f)

config['tensor_model_parallel_size'] = tp
config['pipeline_model_parallel_size'] = pp
config['sequence_parallel'] = (tp > 1)
config['global_batch_size'] = gbs
config['micro_batch_size'] = mbs
config['seq_length'] = seq_len
config['encoder_seq_length'] = seq_len
config['gradient_accumulation_steps'] = grad_accum
config['use_distributed_optimizer'] = True
config['use_flash_attn'] = True        # Maximum performance: ROCm Flash Attention
config['use_fused_rmsnorm'] = True      # Maximum performance: AMD fused RMSNorm kernels
config['fp32_residual_connection'] = False
config['train_iters'] = train_iters
config['lr_decay_iters'] = train_iters
config['lr_warmup_iters'] = warmup_steps

config['seed'] = int(os.environ.get('SEED', '42'))
config['init_method_std'] = 0.02
config['adam_beta1'] = beta1
config['adam_beta2'] = beta2

if 'modules' in config and 'pre_trainer' in config['modules']:
    overrides = config['modules']['pre_trainer'].get('overrides', {})
    overrides['init_method_std'] = 0.02
    overrides['global_batch_size'] = gbs
    overrides['micro_batch_size'] = mbs
    overrides['seq_length'] = seq_len
    overrides['lr_warmup_iters'] = warmup_steps
    overrides['train_iters'] = train_iters
    config['modules']['pre_trainer']['overrides'] = overrides

data_prefix = os.environ['DATA_PREFIX']
tokenizer_model = os.environ['TOKENIZER_MODEL']
config['data_path'] = data_prefix
config['tokenizer_type'] = 'HuggingFaceTokenizer'
config['tokenizer_model'] = tokenizer_model
config['split'] = '100,0,0'
config['eval_iters'] = 0
config['eval_interval'] = train_iters + 1

config['log_interval'] = 1
config['log_memory_to_tensorboard'] = True
config['log_throughput'] = True

config['disable_tensorboard'] = True
config['disable_wandb'] = True
config['disable_mlflow'] = True
config['log_timers_to_tensorboard'] = False
config['log_learning_rate_to_tensorboard'] = False
config['log_loss_scale_to_tensorboard'] = False

config['profile'] = False
config['use_pytorch_profiler'] = False
config['torch_profiler_with_stack'] = False
config['torch_profiler_record_shapes'] = False
config['torch_profiler_use_gzip'] = False

# Disable gradient checkpointing/recompute for fair comparison
config['recompute_granularity'] = None
config['recompute_method'] = None
config['recompute_num_layers'] = None

with open(patched_config, 'w') as f:
    yaml.dump(config, f)
PYTHON_EOF
    echo "Patched config written to: $PATCHED_CONFIG"
else
    echo "WARNING: pyyaml not available, using unpatched config"
fi

export EXP="$PATCHED_CONFIG"

MEMORY_LOG="$TPRIMAT_PATH/output/memory_llama_${DATASET}.log"
(
    while true; do
        if command -v rocm-smi &>/dev/null; then
            rocm-smi --showmeminfo vram 2>/dev/null | grep -E "GPU|Used" >> "$MEMORY_LOG"
        elif command -v nvidia-smi &>/dev/null; then
            nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits >> "$MEMORY_LOG"
        fi
        sleep 2
    done
) &
MEMORY_PID=$!

LOG_FILE="$TPRIMAT_PATH/output/training_main_llama_${DATASET}.log"
: > "$LOG_FILE"
tail -f "$LOG_FILE" &
TAIL_PID=$!

bash "$TRAIN_SCRIPT" \
    --train_iters "$TRAIN_ITERS" \
    --global_batch_size "$GBS" \
    --micro_batch_size "$MBS" \
    --seq_length "$SL" \
    --tensor_model_parallel_size "$TP" \
    --pipeline_model_parallel_size "$PP" \
    --lr "$LR" \
    --min_lr 0.0 \
    --lr_warmup_iters "$WARMUP_STEPS" \
    --lr_decay_style cosine \
    --lr_decay_iters "$TRAIN_ITERS" \
    --weight_decay "$WEIGHT_DECAY" \
    --data_path "$DATA_PREFIX" \
    --tokenizer_type HuggingFaceTokenizer \
    --tokenizer_model "$TOKENIZER_MODEL" \
    --split 100,0,0 \
    --seed "$SEED" \
    >> "$LOG_FILE" 2>&1

kill $TAIL_PID 2>/dev/null || true
wait $TAIL_PID 2>/dev/null || true

kill $MEMORY_PID 2>/dev/null || true

cd "$TPRIMAT_PATH"

MEMORY_ARG=""
if [ -f "$MEMORY_LOG" ]; then
    MEMORY_ARG="--memory-log $MEMORY_LOG"
fi

python3 evaluate/extract.py \
    --log-file "$TPRIMAT_PATH/output/training_main_llama_${DATASET}.log" \
    --model-name "llama" \
    --dataset "$DATASET" \
    --output "$TPRIMAT_PATH/output/train_amd_prim_llama_${DATASET}.json" \
    --num-gpus "$NUM_GPUS" \
    --global-batch-size "$GBS" \
    --micro-batch-size "$MBS" \
    --tensor-parallel-size "$TP" \
    --pipeline-parallel-size "$PP" \
    --sequence-length "$SL" \
    --parallel-strategy "TP${TP}_PP${PP}_DP${DP}" \
    $MEMORY_ARG

rm -f "$MEMORY_LOG"
