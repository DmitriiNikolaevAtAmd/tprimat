#!/bin/bash
docker run --gpus all -it --rm \
    --name primat \
    --shm-size=64g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v $(pwd):/workspace/code \
    -v /data:/data \
    -w /workspace/code \
    --env-file config.env \
    --env-file secrets.env \
    -e CUDA_LAUNCH_BLOCKING=1 \
    primat:nvd "$@"
