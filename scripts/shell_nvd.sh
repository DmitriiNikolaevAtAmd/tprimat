#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

docker run --gpus all -it --rm \
    --name primat \
    --shm-size=64g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v "$(pwd)":/workspace/code \
    -v /data:/data \
    -w /workspace/code \
    --env-file config.env \
    --env-file secrets.env \
    -e CUDA_LAUNCH_BLOCKING=1 \
    primat:nvd "$@"
