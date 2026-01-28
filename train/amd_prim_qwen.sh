#!/bin/bash
set -e
cd "$(dirname "$0")"
TPRIMAT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PRIMUS_PATH="${PRIMUS_PATH:-/workspace/Primus}"

export PYTORCH_ALLOC_CONF='expandable_segments:True'
export PYTHONHASHSEED="42"
export HSA_NO_SCRATCH_RECLAIM=1
export HSA_ENABLE_SDMA=1
export HSA_FORCE_FINE_GRAIN_PCIE=1
export PYTHONWARNINGS="ignore::UserWarning,ignore::FutureWarning,ignore::DeprecationWarning"
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=error
export HF_HUB_DISABLE_PROGRESS_BARS=1
export RCCL_DEBUG=WARN
export NCCL_DEBUG=WARN
export GLOO_LOG_LEVEL=WARN
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-ens51np0}"
export RCCL_MSCCL_ENABLE=0

TP="${TP:-1}"
PP="${PP:-1}"
DP="${DP:-8}"
MBS="${MBS:-1}"
GBS="${GBS:-64}"
SEQ_LEN="${SEQ_LEN:-2048}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
TRAIN_ITERS="${TRAIN_ITERS:-500}"
LR="${LR:-0.0003}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
WARMUP_STEPS="${WARMUP_STEPS:-50}"
NUM_GPUS="$((TP * PP * DP))"

mkdir -p "$TPRIMAT_PATH/output"
if [ ! -d "$PRIMUS_PATH" ]; then
    echo "ERROR: Primus directory not found at: $PRIMUS_PATH"
    echo "Please set PRIMUS_PATH environment variable or ensure /workspace/Primus exists"
    exit 1
fi
CONFIG_FILE="examples/megatron/configs/MI300X/qwen2.5_7B-BF16-pretrain.yaml"
if [ ! -f "$PRIMUS_PATH/$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found at: $PRIMUS_PATH/$CONFIG_FILE"
    echo "Available configs:"
    ls -1 "$PRIMUS_PATH/examples/megatron/configs/MI300X/" 2>/dev/null | grep -i qwen || echo "  (none found)"
    exit 1
fi

cd "$PRIMUS_PATH"

PATCHED_CONFIG="$TPRIMAT_PATH/output/qwen2.5_7B-BF16-pretrain.yaml"
cp "$PRIMUS_PATH/$CONFIG_FILE" "$PATCHED_CONFIG"

if python3 -c "import yaml" 2>/dev/null; then
    python3 -c "
import yaml
with open('$PATCHED_CONFIG', 'r') as f:
    config = yaml.safe_load(f)

config['tensor_model_parallel_size'] = int('$TP')
config['pipeline_model_parallel_size'] = int('$PP')
config['sequence_parallel'] = False
config['global_batch_size'] = int('$GBS')
config['micro_batch_size'] = int('$MBS')
config['seq_length'] = int('$SEQ_LEN')
config['encoder_seq_length'] = int('$SEQ_LEN')
config['gradient_accumulation_steps'] = int('$GRAD_ACCUM')
config['use_distributed_optimizer'] = True
config['use_flash_attn'] = True
config['use_fused_rmsnorm'] = True
config['fp32_residual_connection'] = False
config['train_iters'] = int('$TRAIN_ITERS')
config['lr_decay_iters'] = int('$TRAIN_ITERS')
config['lr_warmup_iters'] = int('$WARMUP_STEPS')

with open('$PATCHED_CONFIG', 'w') as f:
    yaml.dump(config, f)
"
    echo "Config patched: TP=$TP, PP=$PP, DP=$DP, micro_batch=$MBS, global_batch=$GBS, seq_len=$SEQ_LEN"
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

filter_noise() {
    grep -v -E "(^\[Primus CLI\]|^\[Primus\] sys\.path|^Supported flash-attn versions|^\[aiter\]|^fused_indices_to_multihot|^\[PrimusPatch\]|^\[Gloo\] Rank|waiting for baton release)"
}

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
    2>&1 | tee "$TPRIMAT_PATH/output/training_main_qwen_raw.log" | filter_noise | tee "$TPRIMAT_PATH/output/training_main_qwen.log"

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
