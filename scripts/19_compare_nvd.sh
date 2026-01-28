#!/bin/bash
set -e
cd "$(dirname "$0")"
OUTPUT_DIR="${OUTPUT_DIR:-./output}"
mkdir -p "$OUTPUT_DIR"
python3 19_compare_nvd.py --results-dir "$OUTPUT_DIR" "$@"
