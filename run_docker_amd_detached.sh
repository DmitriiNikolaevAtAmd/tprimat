#!/bin/bash
# Run AMD/ROCm Docker container in detached mode with logging

# Load secrets if available
if [ -f secrets.env ]; then
    source secrets.env
fi

# Create logs directory
mkdir -p logs

# Get timestamp for log file
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/benchmark_${TIMESTAMP}.log"

echo "Starting container in detached mode..."
echo "Logs will be written to: $LOG_FILE"

docker run --gpus all -d \
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
    primat:amd \
    /bin/bash -c "cd /workspace/tprimat && ./run_primus_all.sh 2>&1 | tee $LOG_FILE"

echo ""
echo "Container started! Monitor progress with:"
echo "  docker logs -f primat-amd"
echo ""
echo "Or attach to the container:"
echo "  docker exec -it primat-amd /bin/bash"
echo ""
echo "Check status:"
echo "  docker ps -a | grep primat-amd"
echo ""
echo "Stop container:"
echo "  docker stop primat-amd && docker rm primat-amd"
