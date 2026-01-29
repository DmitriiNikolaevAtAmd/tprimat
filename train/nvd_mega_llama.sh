#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
export OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/output}"
export DATA_DIR="${DATA_DIR:-$ROOT_DIR/data}"
export HF_HOME="${HF_HOME:-$ROOT_DIR/cache}"
mkdir -p "$OUTPUT_DIR" "$HF_HOME"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=INFO

if [ "$NUM_GPUS" -gt 1 ]; then
    MASTER_PORT="${MASTER_PORT:-$(python3 - <<'PY'
import socket
sock = socket.socket()
sock.bind(("", 0))
port = sock.getsockname()[1]
sock.close()
print(port)
PY
)}"
    torchrun --nproc_per_node="$NUM_GPUS" \
             --nnodes=1 \
             --node_rank=0 \
             --master_addr=localhost \
             --master_port="$MASTER_PORT" \
             "$SCRIPT_DIR/nvd_mega.py" llama
else
    python3 -u "$SCRIPT_DIR/nvd_mega.py" llama
fi
