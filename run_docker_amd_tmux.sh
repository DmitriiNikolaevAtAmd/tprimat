#!/bin/bash
# Run AMD/ROCm Docker container with tmux session

# Load secrets if available
if [ -f secrets.env ]; then
    source secrets.env
fi

echo "Starting container with tmux..."
echo ""
echo "Inside the container, run:"
echo "  tmux new -s benchmark"
echo "  ./run_primus_all.sh"
echo ""
echo "To detach from tmux: Press Ctrl+B, then D"
echo "To reattach later: docker exec -it primat-amd tmux attach -t benchmark"
echo ""

docker run --gpus all -it \
    --name primat-amd \
    --device=/dev/kfd \
    --device=/dev/dri \
    --shm-size=64g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v $(pwd):/workspace/tprimat \
    -v /data:/data \
    -w /workspace/tprimat \
    -e HF_TOKEN="${HF_TOKEN}" \
    primat:amd /bin/bash
