#!/bin/bash
if [ -f secrets.env ]; then
    source secrets.env
fi

TRAIN_ITERS="${TRAIN_ITERS:-10}"
WARMUP_STEPS="${WARMUP_STEPS:-2}"
DATA_SAMPLES="${DATA_SAMPLES:-100}"

docker run --gpus all -it --rm \
    --name primat \
    --shm-size=64g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v $(pwd):/workspace/tprimat \
    -v /data:/data \
    -w /workspace/tprimat \
    -e HF_TOKEN="${HF_TOKEN}" \
    -e TRAIN_ITERS="${TRAIN_ITERS}" \
    -e WARMUP_STEPS="${WARMUP_STEPS}" \
    -e DATA_SAMPLES="${DATA_SAMPLES}" \
    primat:nvd "$@"
