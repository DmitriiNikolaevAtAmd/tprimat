#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"$SCRIPT_DIR/train_prim_llama.sh"
"$SCRIPT_DIR/train_prim_qwen.sh"
