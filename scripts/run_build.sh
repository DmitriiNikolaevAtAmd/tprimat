#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

if command -v nvidia-smi &>/dev/null; then
    echo "Detected NVIDIA GPU — building primat:nvd"
    docker build -t primat:nvd -f nvd.Dockerfile .
elif [ -e /dev/kfd ]; then
    echo "Detected AMD GPU — building primat:amd"
    docker build -t primat:amd -f amd.Dockerfile .
else
    echo "ERROR: No GPU runtime detected (need nvidia-smi or /dev/kfd)"
    exit 1
fi
