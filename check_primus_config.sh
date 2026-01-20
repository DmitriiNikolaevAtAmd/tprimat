#!/bin/bash
# Script to check Primus configuration for benchmark comparison

echo "╔════════════════════════════════════════════════════════════╗"
echo "║         Primus Configuration Checker                      ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

PRIMUS_PATH="/workspace/Primus"
CONFIG_DIR="$PRIMUS_PATH/examples/megatron/configs/MI300X"

if [ ! -d "$PRIMUS_PATH" ]; then
    echo "[!] Primus not found at $PRIMUS_PATH"
    echo "   Please update PRIMUS_PATH in this script"
    exit 1
fi

echo "  * Searching for Llama 3.1 8B configuration..."
echo ""

# Find Llama config files
LLAMA_CONFIGS=$(find "$CONFIG_DIR" -name "*llama*8b*.yaml" -o -name "*llama*8B*.yaml" 2>/dev/null)

if [ -z "$LLAMA_CONFIGS" ]; then
    echo "  x No Llama 8B config found in $CONFIG_DIR"
    echo ""
    echo "Available configs:"
    ls -1 "$CONFIG_DIR" 2>/dev/null || echo "Directory not accessible"
    exit 1
fi

echo "  + Found configuration files:"
echo "$LLAMA_CONFIGS"
echo ""
echo "════════════════════════════════════════════════════════════"

for CONFIG in $LLAMA_CONFIGS; do
    echo ""
    echo "  * File: $(basename $CONFIG)"
    echo "────────────────────────────────────────────────────────────"
    
    # Extract key parameters
    echo ""
    echo "  * Critical Parameters:"
    echo ""
    
    # Tensor Parallelism
    TP=$(grep -E "tensor_model_parallel_size|tp_size" "$CONFIG" | head -1)
    echo "  Tensor Parallel (TP): $TP"
    
    # Pipeline Parallelism
    PP=$(grep -E "pipeline_model_parallel_size|pp_size" "$CONFIG" | head -1)
    echo "  Pipeline Parallel (PP): $PP"
    
    # Precision
    PRECISION=$(grep -E "precision|fp16|bf16|fp8" "$CONFIG" | head -1)
    echo "  Precision: $PRECISION"
    
    # Batch sizes
    GBS=$(grep -E "global_batch_size" "$CONFIG" | head -1)
    echo "  Global Batch Size: $GBS"
    
    MBS=$(grep -E "micro_batch_size" "$CONFIG" | head -1)
    echo "  Micro Batch Size: $MBS"
    
    # Sequence length
    SEQ=$(grep -E "seq_length|max_position_embeddings" "$CONFIG" | head -1)
    echo "  Sequence Length: $SEQ"
    
    echo ""
    echo "  * Memory & Optimization:"
    echo ""
    
    # Activation checkpointing
    CHECKPOINT=$(grep -E "activations_checkpoint|recompute" "$CONFIG" | head -1)
    echo "  Activation Checkpoint: $CHECKPOINT"
    
    # Flash attention
    FLASH=$(grep -E "flash_attention|use_flash_attn" "$CONFIG" | head -1)
    echo "  Flash Attention: $FLASH"
    
    echo ""
    echo "════════════════════════════════════════════════════════════"
done

echo ""
echo "  * Comparison with NeMo (NVIDIA H100):"
echo "────────────────────────────────────────────────────────────"
echo "  NVIDIA TP: 4 (model split across 4 GPUs)"
echo "  NVIDIA PP: 1"
echo "  NVIDIA Precision: FP8"
echo "  NVIDIA GBS: 128"
echo "  NVIDIA MBS: 1"
echo "  NVIDIA Seq: 2048"
echo ""
echo "❓ Questions to verify optimal AMD config:"
echo "  1. Is TP=1? (Leverages MI300X's 192GB memory)"
echo "  2. Is precision BF16? (Native to MI300X)"
echo "  3. Is GBS=128? (Matches NVIDIA)"
echo "  4. Is MBS optimized for throughput?"
echo ""
