#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/output}"
mkdir -p "$OUTPUT_DIR"
python3 "$SCRIPT_DIR/compare_amd.py" --results-dir "$OUTPUT_DIR" "$@"
