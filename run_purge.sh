#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if command -v nvidia-smi &>/dev/null; then
    IMAGE="primat:nvd"
    DOCKER_ARGS="--gpus all"
elif [ -e /dev/kfd ]; then
    IMAGE="primat:amd"
    DOCKER_ARGS="--device=/dev/kfd --device=/dev/dri --group-add video"
else
    echo "ERROR: No GPU runtime detected (need nvidia-smi or /dev/kfd)"
    exit 1
fi

docker run --rm \
    $DOCKER_ARGS \
    -v "$(pwd)":/workspace/code \
    -v /data:/data \
    -w /workspace/code \
    "$IMAGE" bash purge.sh "$@"
