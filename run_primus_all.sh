#!/bin/bash
# Run all Primus models (Llama, Mixtral, Qwen) in sequence

set -e

echo "╔════════════════════════════════════════════════════════════╗"
echo "║        Primus Training: All Models                        ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODELS=("llama" "mixtral" "qwen")
SUCCESS=()
FAILED=()

echo "📋 Training Plan:"
echo "  1. Llama 3.1 8B"
echo "  2. Mixtral 8x7B"
echo "  3. Qwen 2.5 7B"
echo ""
echo "Press Enter to continue or Ctrl+C to cancel..."
read

for MODEL in "${MODELS[@]}"; do
    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "Starting: $MODEL"
    echo "════════════════════════════════════════════════════════════"
    echo ""
    
    if "$SCRIPT_DIR/run_primus_${MODEL}.sh"; then
        SUCCESS+=("$MODEL")
        echo "✅ $MODEL completed successfully"
    else
        FAILED+=("$MODEL")
        echo "❌ $MODEL failed"
        
        # Ask if user wants to continue
        echo ""
        read -p "Continue with next model? (Y/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]] && [[ ! -z $REPLY ]]; then
            echo "Stopping..."
            break
        fi
    fi
    
    # Cooldown between models
    if [ "$MODEL" != "${MODELS[-1]}" ]; then
        echo ""
        echo "Cooling down for 30 seconds..."
        sleep 30
    fi
done

echo ""
echo "════════════════════════════════════════════════════════════"
echo "SUMMARY"
echo "════════════════════════════════════════════════════════════"
echo ""

if [ ${#SUCCESS[@]} -gt 0 ]; then
    echo "✅ Successful (${#SUCCESS[@]}): ${SUCCESS[*]}"
    echo ""
    echo "Results saved to:"
    for MODEL in "${SUCCESS[@]}"; do
        echo "  📄 output/benchmark_rocm_${MODEL}.json"
    done
    echo ""
fi

if [ ${#FAILED[@]} -gt 0 ]; then
    echo "❌ Failed (${#FAILED[@]}): ${FAILED[*]}"
    echo ""
fi

if [ ${#SUCCESS[@]} -gt 0 ]; then
    echo "════════════════════════════════════════════════════════════"
    echo "🎯 Next Steps:"
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
