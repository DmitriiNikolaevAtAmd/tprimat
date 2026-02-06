#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/output}"
mkdir -p "$OUTPUT_DIR"

# Usage: ./compare_nvd.sh [--dataset bc|c4]
# Examples:
#   ./compare_nvd.sh              # Compare all NeMo results
#   ./compare_nvd.sh --dataset bc # Compare only BookCorpus results
#   ./compare_nvd.sh --dataset c4 # Compare only C4 results

python3 "$SCRIPT_DIR/compare_nvd.py" --results-dir "$OUTPUT_DIR" "$@"
