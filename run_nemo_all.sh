#!/bin/bash
# Run NeMo benchmark for all models
#
# Usage:
#   ./run_nemo_all.sh                           # Default: truly_identical
#   ./run_nemo_all.sh --parallel balanced       # Use balanced strategy
#   ./run_nemo_all.sh --parallel maximum_performance

PARALLEL="${PARALLEL:-truly_identical}"
OUTPUT_DIR="${OUTPUT_DIR:-output}"

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

python3 benchmark.py --parallel "$PARALLEL" --output-dir "$OUTPUT_DIR"
