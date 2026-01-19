#!/bin/bash
# Run all Primus models (Llama, Qwen) in sequence
#
# Usage:
#   ./run_primus_all.sh                           # Default: truly_identical
#   ./run_primus_all.sh --parallel balanced       # Use balanced strategy
#   ./run_primus_all.sh --parallel maximum_performance

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-output}"
PARALLEL="${PARALLEL:-truly_identical}"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --parallel)
            PARALLEL="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--parallel STRATEGY] [--output-dir DIR]"
            echo "Strategies: truly_identical, maximum_performance, memory_optimized,"
            echo "            minimal_communication, balanced"
            exit 1
            ;;
    esac
done

export PARALLEL
export OUTPUT_DIR

echo "╔════════════════════════════════════════════════════════════╗"
echo "║           Primus Training: All Models                      ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "Strategy: $PARALLEL"
echo "Output: $OUTPUT_DIR"
echo ""

# Run Llama
echo "═══════════════════════════════════════════════════════════"
echo "Running Llama..."
echo "═══════════════════════════════════════════════════════════"
"$SCRIPT_DIR/run_primus_llama.sh"

echo ""
echo "Cooling down for 30 seconds..."
sleep 30

# Run Qwen
echo "═══════════════════════════════════════════════════════════"
echo "Running Qwen..."
echo "═══════════════════════════════════════════════════════════"
"$SCRIPT_DIR/run_primus_qwen.sh"

echo ""
echo "✅ All Primus models completed!"
