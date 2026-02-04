#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

source "$ROOT_DIR/config.env"

export DATA_DIR
export OUTPUT_DIR
export HF_HOME

if [ "${DATA_DIR:-}" = "/data/tprimat" ] && [ ! -e "/data/tprimat" ] && [ -d "$ROOT_DIR/data" ]; then
    mkdir -p /data
    ln -s "$ROOT_DIR/data" /data/tprimat
fi

mkdir -p "$OUTPUT_DIR"

"$SCRIPT_DIR/train_amd_prim_llama.sh"
"$SCRIPT_DIR/train_amd_prim_qwen.sh"

"$ROOT_DIR/evaluate/validate_outputs.sh" --platform amd
"$ROOT_DIR/evaluate/compare_amd.sh"
