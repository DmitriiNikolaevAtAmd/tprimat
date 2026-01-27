#!/bin/bash
set -e
TPRIMAT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRIMUS_PATH="${PRIMUS_PATH:-/workspace/Primus}"

export PYTORCH_ALLOC_CONF='expandable_segments:True'
export PYTHONHASHSEED="42"
export HSA_NO_SCRATCH_RECLAIM=1
export HSA_ENABLE_SDMA=1
export HSA_FORCE_FINE_GRAIN_PCIE=1
# Disable debug logging for better performance
export RCCL_DEBUG=WARN
export NCCL_DEBUG=WARN

# RCCL network interface
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-ens51np0}"
export RCCL_MSCCL_ENABLE=0

# HuggingFace authentication for gated models (Llama 3.1)
echo "[*] Checking HuggingFace authentication..."
if [ -n "$HF_TOKEN" ]; then
    export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
    echo "    HF_TOKEN is set"
elif [ -n "$HUGGINGFACE_HUB_TOKEN" ]; then
    export HF_TOKEN="$HUGGINGFACE_HUB_TOKEN"
    echo "    HUGGINGFACE_HUB_TOKEN is set"
else
    echo "    ERROR: HF_TOKEN not set!"
    echo ""
    echo "    Llama 3.1 is a gated model. You need to:"
    echo "    1. Get a token from https://huggingface.co/settings/tokens"
    echo "    2. Accept the license at https://huggingface.co/meta-llama/Llama-3.1-8B"
    echo "    3. Set: export HF_TOKEN='hf_xxxxxxxxxxxxx'"
    echo ""
    echo "    Or use Qwen instead (no auth needed): ./train_amd_prim_qwen.sh"
    exit 1
fi

mkdir -p "$TPRIMAT_PATH/output"
if [ ! -d "$PRIMUS_PATH" ]; then
    echo "ERROR: Primus directory not found at: $PRIMUS_PATH"
    echo "Please set PRIMUS_PATH environment variable or ensure /workspace/Primus exists"
    exit 1
fi
CONFIG_FILE="examples/megatron/configs/MI300X/llama3.1_8B-BF16-pretrain.yaml"
if [ ! -f "$PRIMUS_PATH/$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found at: $PRIMUS_PATH/$CONFIG_FILE"
    echo "Available configs:"
    ls -1 "$PRIMUS_PATH/examples/megatron/configs/MI300X/" 2>/dev/null | grep -i llama || echo "  (none found)"
    exit 1
fi

cd "$PRIMUS_PATH"

# Patch config with minimal_communication strategy (identical to NVIDIA)
# TP=1, PP=1, DP=4, GradAccum=32 - eliminates model parallelism overhead
PATCHED_CONFIG="$TPRIMAT_PATH/output/llama3.1_8B-BF16-pretrain.yaml"
cp "$PRIMUS_PATH/$CONFIG_FILE" "$PATCHED_CONFIG"

if python3 -c "import yaml" 2>/dev/null; then
    python3 -c "
import yaml
with open('$PATCHED_CONFIG', 'r') as f:
    config = yaml.safe_load(f)

# Minimal communication strategy: same as NVIDIA baseline
config['tensor_model_parallel_size'] = 1
config['pipeline_model_parallel_size'] = 1
config['gradient_accumulation_steps'] = 32

# Enable performance optimizations
config['use_distributed_optimizer'] = True
config['use_flash_attn'] = True
config['use_fused_rmsnorm'] = True
config['fp32_residual_connection'] = False

# Ensure training parameters
config['train_iters'] = 50
config['lr_decay_iters'] = 50
config['lr_warmup_iters'] = 10

with open('$PATCHED_CONFIG', 'w') as f:
    yaml.dump(config, f)
"
    echo "Config patched with minimal_communication: TP=1, DP=4, GradAccum=32 (matches NVIDIA)"
else
    echo "WARNING: pyyaml not available, using unpatched config"
fi

export EXP="$PATCHED_CONFIG"

# Note: examples/train.sh may not exist; use run_pretrain.sh instead
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
    --lr 0.0003 \
    --min_lr 0.00003 \
    --lr_warmup_iters 10 \
    --lr_decay_style cosine \
    --lr_decay_iters 50 \
    --weight_decay 0.1 \
    2>&1 | tee "$TPRIMAT_PATH/output/training_main_llama.log"

cd "$TPRIMAT_PATH"

python3 extract_metrics.py \
    --log-file "$TPRIMAT_PATH/output/training_main_llama.log" \
    --model-name "llama" \
    --output "$TPRIMAT_PATH/output/train_amd_prim_llama.json" \
    --num-gpus 8 \
    --global-batch-size 128 \
    --sequence-length 2048 \
    --parallel-strategy "minimal_communication"
