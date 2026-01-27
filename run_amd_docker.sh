#!/bin/bash

# Load secrets if available
if [ -f secrets.env ]; then
    source secrets.env
fi

docker run -it --rm \
    --device /dev/dri --device /dev/kfd --device /dev/infiniband \
    --network host --ipc host --group-add video \
    --cap-add SYS_PTRACE --security-opt seccomp=unconfined --privileged \
    -v $HOME:$HOME -v .:/workspace/tprimat \
    -w /workspace/tprimat --shm-size 128G --name primat \
    -e HF_TOKEN="${HF_TOKEN}" \
    -e HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN:-$HF_TOKEN}" \
    primat:latest "$@"
