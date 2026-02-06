#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "$SCRIPT_DIR/config.env"

export DATA_DIR
export OUTPUT_DIR
export HF_HOME

if [ "${DATA_DIR:-}" = "/data/tprimat" ] && [ ! -e "/data/tprimat" ] && [ -d "$SCRIPT_DIR/data" ]; then
    mkdir -p /data
    ln -s "$SCRIPT_DIR/data" /data/tprimat
fi

mkdir -p "$OUTPUT_DIR"

"$SCRIPT_DIR/train/train_nvd_nemo_llama.sh"
"$SCRIPT_DIR/train/train_nvd_nemo_qwen.sh"

"$SCRIPT_DIR/evaluate/validate_outputs.sh" --platform nvd
"$SCRIPT_DIR/evaluate/compare_nvd.sh"
