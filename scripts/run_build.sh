#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if command -v nvidia-smi &>/dev/null; then
    echo "Detected NVIDIA GPU"
    "$SCRIPT_DIR/build_nvd.sh"
elif [ -e /dev/kfd ]; then
    echo "Detected AMD GPU"
    "$SCRIPT_DIR/build_amd.sh"
else
    echo "ERROR: No GPU runtime detected (need nvidia-smi or /dev/kfd)"
    exit 1
fi
