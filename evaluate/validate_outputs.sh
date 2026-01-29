#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/output}"
python3 "$SCRIPT_DIR/validate_outputs.py" --results-dir "$OUTPUT_DIR" "$@"
