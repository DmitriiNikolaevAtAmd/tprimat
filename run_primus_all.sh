#!/bin/bash
# Run all Primus models (Llama, Qwen) in sequence with progress tracking
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

# Check if alive-progress is available for Python-based progress
if python3 -c "from alive_progress import alive_bar" 2>/dev/null; then
    # Use Python wrapper with alive-progress
    python3 "$SCRIPT_DIR/run_primus_pipeline.py" \
        --parallel "$PARALLEL" \
        --output-dir "$OUTPUT_DIR"
else
    # Fallback to shell-only version
    "$SCRIPT_DIR/run_primus_model.sh"
fi
