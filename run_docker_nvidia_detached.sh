#!/bin/bash
# Run NVIDIA/NeMo Docker container in detached mode with logging

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
    --name primat \
    --shm-size=64g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v $(pwd):/workspace/tprimat \
    -v /data:/data \
    -w /workspace/tprimat \
    -e HF_TOKEN="${HF_TOKEN}" \
    primat:nvidia \
    /bin/bash -c "cd /workspace/tprimat && ./run_nemo_all.sh 2>&1 | tee $LOG_FILE"

echo ""
echo "Container started! Monitor progress with:"
echo "  docker logs -f primat"
echo ""
echo "Or attach to the container:"
echo "  docker exec -it primat /bin/bash"
echo ""
echo "Check status:"
echo "  docker ps -a | grep primat"
echo ""
echo "Stop container:"
echo "  docker stop primat && docker rm primat"
