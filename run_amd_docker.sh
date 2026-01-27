#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="${AMD_IMAGE:-tprimat-amd:latest}"

[ -f "$SCRIPT_DIR/secrets.env" ] && source "$SCRIPT_DIR/secrets.env"

if ! docker image inspect "$IMAGE_NAME" &>/dev/null; then
    docker build -q -f "$SCRIPT_DIR/amd.Dockerfile" -t "$IMAGE_NAME" "$SCRIPT_DIR"
fi

docker run -it --rm \
    --network=host \
    --device=/dev/kfd --device=/dev/dri \
    --group-add video \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --shm-size=64g \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    -e HUGGINGFACE_HUB_TOKEN="${HF_TOKEN:-}" \
    -v "$SCRIPT_DIR:/workspace/tprimat" \
    -v "${DATA_DIR:-/data}:/data" \
    -v "${HF_CACHE:-$HOME/.cache/huggingface}:/workspace/cache/huggingface" \
    "$IMAGE_NAME" \
    fish
