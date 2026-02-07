#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

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
    --network=host \
    -v "$(pwd)":/workspace/code \
    -v /data:/data \
    -w /workspace/code \
    --env-file config.env \
    --env-file secrets.env \
    "$IMAGE" bash prepare/data.sh "$@"
