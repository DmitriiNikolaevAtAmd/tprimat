#!/bin/bash
set -e
cd "$(dirname "$0")"
OUTPUT_DIR="${OUTPUT_DIR:-./output}"
mkdir -p "$OUTPUT_DIR"
python3 29_compare_amd.py --results-dir "$OUTPUT_DIR" "$@"
