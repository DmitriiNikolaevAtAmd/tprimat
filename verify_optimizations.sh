#!/bin/bash
# Quick verification script to check if optimizations are properly applied

echo "╔════════════════════════════════════════════════════════════╗"
echo "║     Verifying Primus/AMD Performance Optimizations        ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ERRORS=0

# Check 1: Config.yaml minimal_communication settings
echo "[1] Checking config.yaml minimal_communication strategy..."

# Extract the AMD section under minimal_communication for llama
AMD_SECTION=$(sed -n '/minimal_communication:/,/balanced:/p' "$SCRIPT_DIR/config.yaml" | sed -n '/llama:/,/qwen:/p' | sed -n '/amd:/,/qwen:/p')

if echo "$AMD_SECTION" | grep -q "gradient_accumulation_steps: 32"; then
    echo "    ✓ AMD gradient_accumulation_steps set to 32 (matches NVIDIA)"
else
    echo "    ✗ AMD gradient_accumulation_steps NOT set to 32"
    ERRORS=$((ERRORS + 1))
fi

if echo "$AMD_SECTION" | grep -q "tensor_model_parallel_size: 1"; then
    echo "    ✓ AMD tensor_model_parallel_size set to 1 (no tensor parallelism)"
else
    echo "    ✗ AMD tensor_model_parallel_size NOT set to 1"
    ERRORS=$((ERRORS + 1))
fi

if echo "$AMD_SECTION" | grep -q "data_parallel_size: 4"; then
    echo "    ✓ AMD data_parallel_size set to 4 (matches NVIDIA)"
else
    echo "    ✗ AMD data_parallel_size NOT set to 4"
    ERRORS=$((ERRORS + 1))
fi

echo ""

# Check 2: Dockerfile debug logging and network settings
echo "[2] Checking amd.Dockerfile settings..."
if grep -q "RCCL_DEBUG=WARN" "$SCRIPT_DIR/amd.Dockerfile"; then
    echo "    ✓ RCCL_DEBUG set to WARN"
else
    echo "    ✗ RCCL_DEBUG still at INFO (causes overhead)"
    ERRORS=$((ERRORS + 1))
fi

if grep -q "NCCL_DEBUG=WARN" "$SCRIPT_DIR/amd.Dockerfile"; then
    echo "    ✓ NCCL_DEBUG set to WARN"
else
    echo "    ✗ NCCL_DEBUG still at INFO (causes overhead)"
    ERRORS=$((ERRORS + 1))
fi

if grep -q 'NCCL_SOCKET_IFNAME=eth0' "$SCRIPT_DIR/amd.Dockerfile"; then
    echo "    ✗ NCCL_SOCKET_IFNAME hardcoded to eth0 (causes 'no socket interface' error)"
    ERRORS=$((ERRORS + 1))
else
    echo "    ✓ NCCL_SOCKET_IFNAME not hardcoded (auto-detect in training script)"
fi

echo ""

# Check 3: Training scripts
echo "[3] Checking training scripts..."
if grep -q "RCCL_DEBUG=WARN" "$SCRIPT_DIR/train_amd_prim_llama.sh"; then
    echo "    ✓ train_amd_prim_llama.sh: debug logging disabled"
else
    echo "    ✗ train_amd_prim_llama.sh: debug logging still enabled"
    ERRORS=$((ERRORS + 1))
fi

if grep -q "minimal_communication" "$SCRIPT_DIR/train_amd_prim_llama.sh" && grep -q "gradient_accumulation_steps.*32" "$SCRIPT_DIR/train_amd_prim_llama.sh"; then
    echo "    ✓ train_amd_prim_llama.sh: applies minimal_communication (TP=1, DP=4, GradAccum=32)"
else
    echo "    ✗ train_amd_prim_llama.sh: missing minimal_communication config"
    ERRORS=$((ERRORS + 1))
fi

if [ -f "$SCRIPT_DIR/train_amd_prim_qwen.sh" ]; then
    if grep -q "RCCL_DEBUG=WARN" "$SCRIPT_DIR/train_amd_prim_qwen.sh"; then
        echo "    ✓ train_amd_prim_qwen.sh: debug logging disabled"
    else
        echo "    ✗ train_amd_prim_qwen.sh: debug logging still enabled"
        ERRORS=$((ERRORS + 1))
    fi
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ $ERRORS -eq 0 ]; then
    echo "✓ All optimizations verified successfully!"
    echo ""
    echo "Next steps:"
    echo "  1. Rebuild Docker image: docker build -f amd.Dockerfile -t tprimat-amd:optimized ."
    echo "  2. Run training: ./train_amd_prim_llama.sh"
    echo "  3. Compare results with NVIDIA baseline"
    echo ""
    echo "Expected improvement: 2-4x speedup (15s → 4-7s per iteration)

Strategy: minimal_communication (TP=1, DP=4, GradAccum=32)
- Identical settings to NVIDIA baseline (apples-to-apples comparison)
- Uses 4 GPUs with zero tensor/pipeline parallelism overhead
- Entire model on each GPU (leverages MI300X's 192GB memory)"
else
    echo "✗ Found $ERRORS issue(s). Please review PERFORMANCE_OPTIMIZATIONS.md"
    exit 1
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
