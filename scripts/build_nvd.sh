#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

echo "Building tprimat:nvd ..."
docker build -f nvd.Dockerfile -t tprimat:nvd .
echo "Done: tprimat:nvd"
