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

# Critical AMD performance settings
export RCCL_DEBUG=ERROR
export NCCL_DEBUG=ERROR
export RCCL_MSCCL_ENABLE=0
export HSA_NO_SCRATCH_RECLAIM=1
export HSA_ENABLE_SDMA=1
export HSA_FORCE_FINE_GRAIN_PCIE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_DEVICE_MAX_CONNECTIONS=1

CONFIG_FILE="examples/megatron/configs/MI300X/qwen2.5_7B-BF16-pretrain.yaml"
cd "$PRIMUS_PATH"

PATCHED_CONFIG="$TPRIMAT_PATH/output/qwen2.5_7B-BF16-pretrain.yaml"
cp "$PRIMUS_PATH/$CONFIG_FILE" "$PATCHED_CONFIG"

if python3 -c "import yaml" 2>/dev/null; then
    python3 -c "
import yaml
with open('$PATCHED_CONFIG', 'r') as f:
    config = yaml.safe_load(f)

config['tensor_model_parallel_size'] = 1
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
config['train_iters'] = 50
config['lr_decay_iters'] = 50
config['lr_warmup_iters'] = 10
config['mock_data'] = True

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
    --train_iters 50 \
    --global_batch_size "$GBS" \
    --micro_batch_size "$MBS" \
    --seq_length 2048 \
    --tensor_model_parallel_size 1 \
    --pipeline_model_parallel_size 1 \
    --lr 3.0e-4 \
    --min_lr 0.0 \
    --lr_warmup_iters 10 \
    --lr_decay_style cosine \
    --lr_decay_iters 50 \
    --weight_decay 0.1 \
    2>&1 | tee "$TPRIMAT_PATH/output/training_main_qwen.log"

cd "$TPRIMAT_PATH"

python3 evaluate/extract_prim_metrics.py \
    --log-file "$TPRIMAT_PATH/output/training_main_qwen.log" \
    --model-name "qwen" \
    --output "$TPRIMAT_PATH/output/train_amd_prim_qwen.json" \
    --num-gpus "$NUM_GPUS" \
    --global-batch-size "$GBS" \
    --micro-batch-size "$MBS" \
    --tensor-parallel-size 1 \
    --pipeline-parallel-size 1 \
    --sequence-length 2048 \
    --parallel-strategy "TP1_SP"