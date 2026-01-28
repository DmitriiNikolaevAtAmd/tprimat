#!/bin/bash
set -e
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
OUTPUT_DIR="./output"
mkdir -p "$OUTPUT_DIR"
export HF_HOME="./cache"
mkdir -p "$HF_HOME"
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
             13_train_nvd_mega.py llama
else
    python3 -u 13_train_nvd_mega.py llama
fi
