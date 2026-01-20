#!/bin/bash
# Run all Primus models (Llama, Qwen) in sequence

set -e

echo "╔════════════════════════════════════════════════════════════╗"
echo "║        Primus Training: All Models                        ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODELS=("llama" "qwen")
SUCCESS=()
FAILED=()

echo "[*] Training Plan:"
echo "  1. Llama 3.1 8B"
echo "  2. Qwen 2.5 7B"
echo ""

for MODEL in "${MODELS[@]}"; do
    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "Starting: $MODEL"
    echo "════════════════════════════════════════════════════════════"
    echo ""
    
    if "$SCRIPT_DIR/run_primus_${MODEL}.sh"; then
        SUCCESS+=("$MODEL")
        echo "[OK] $MODEL completed successfully"
    else
        FAILED+=("$MODEL")
        echo "[X] $MODEL failed (continuing with next model...)"
    fi
    
    # Cooldown between models
    if [ "$MODEL" != "${MODELS[-1]}" ]; then
        echo ""
        echo "Cooling down for 5 seconds..."
        sleep 5
    fi
done

echo ""
echo "════════════════════════════════════════════════════════════"
echo "SUMMARY"
echo "════════════════════════════════════════════════════════════"
echo ""

if [ ${#SUCCESS[@]} -gt 0 ]; then
    echo "[OK] Successful (${#SUCCESS[@]}): ${SUCCESS[*]}"
    echo ""
    echo "Results saved to:"
    # Use OUTPUT_DIR or default to output
    RESULTS_DIR="${OUTPUT_DIR:-output}"
    for MODEL in "${SUCCESS[@]}"; do
        echo "  [.] ${RESULTS_DIR}/benchmark_rocm_${MODEL}.json"
    done
    echo ""
fi

if [ ${#FAILED[@]} -gt 0 ]; then
    echo "[X] Failed (${#FAILED[@]}): ${FAILED[*]}"
    echo ""
fi

if [ ${#SUCCESS[@]} -gt 0 ]; then
    echo "════════════════════════════════════════════════════════════"
    echo "[=>] Next Steps:"
    echo "════════════════════════════════════════════════════════════"
    echo ""
    echo "1. Run on NVIDIA system (if not done):"
    echo "   ./benchmark.py"
    echo ""
    echo "2. Compare results:"
    echo "   python3 compare_results.py"
    echo ""
    echo "3. View enhanced metrics:"
    echo "   python3 compare_with_enhanced_metrics.py"
    echo ""
fi
