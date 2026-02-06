#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/output}"
mkdir -p "$OUTPUT_DIR"

# Usage: ./compare.sh [--dataset bc|c4]
# Examples:
#   ./compare.sh                  # Compare all results, generate per-dataset plots
#   ./compare.sh --dataset bc     # Compare only BookCorpus results
#   ./compare.sh --dataset c4     # Compare only C4 results

python3 "$SCRIPT_DIR/compare.py" --results-dir "$OUTPUT_DIR" "$@"
