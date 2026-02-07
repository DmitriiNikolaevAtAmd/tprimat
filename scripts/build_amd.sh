#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

echo "Building tprimat:amd ..."
docker build -f amd.Dockerfile -t tprimat:amd .
echo "Done: tprimat:amd"
