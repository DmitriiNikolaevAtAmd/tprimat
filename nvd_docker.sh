#!/bin/bash
if [ -f secrets.env ]; then
    source secrets.env
fi

docker run --gpus all -it --rm \
    --name primat \
    --shm-size=64g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v $(pwd):/workspace/tprimat \
    -v /data:/data \
    -w /workspace/tprimat \
    -e HF_TOKEN="${HF_TOKEN}" \
    primat:nvd "$@"
