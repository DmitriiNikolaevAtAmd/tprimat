#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

docker build -t primat:amd -f amd.Dockerfile .

docker run --rm \
    --network=host \
    --device=/dev/kfd --device=/dev/dri \
    --group-add video \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --shm-size=64g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v "$(pwd)":/workspace/code \
    -v /data:/data \
    -w /workspace/code \
    --env-file config.env \
    --env-file secrets.env \
    primat:amd bash scripts/train_amd.sh
