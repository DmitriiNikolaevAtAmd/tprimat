#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="${AMD_IMAGE:-tprimat-amd:latest}"

[ -f "$SCRIPT_DIR/secrets.env" ] && source "$SCRIPT_DIR/secrets.env"

TRAIN_ITERS="${TRAIN_ITERS:-10}"
WARMUP_STEPS="${WARMUP_STEPS:-2}"
DATA_SAMPLES="${DATA_SAMPLES:-100}"

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
    -e TRAIN_ITERS="${TRAIN_ITERS}" \
    -e WARMUP_STEPS="${WARMUP_STEPS}" \
    -e DATA_SAMPLES="${DATA_SAMPLES}" \
    -v "$SCRIPT_DIR:/workspace/tprimat" \
    -v "${DATA_DIR:-/data}:/data" \
    -v "${HF_CACHE:-$HOME/.cache/huggingface}:/workspace/cache/huggingface" \
    "$IMAGE_NAME" \
    fish
