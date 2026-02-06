#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TPRIMAT_PATH="$(cd "$SCRIPT_DIR/.." && pwd)"

source "$TPRIMAT_PATH/config.env"

export OUTPUT_DIR="${OUTPUT_DIR:-$TPRIMAT_PATH/output}"

python3 "$SCRIPT_DIR/compare.py" --results-dir "$OUTPUT_DIR"
