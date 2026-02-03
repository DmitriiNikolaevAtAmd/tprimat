#!/bin/bash
set -e

DATA_DIR="${DATA_DIR:-/data/tprimat}"
WORKERS="${WORKERS:-$(nproc 2>/dev/null || echo 4)}"

bash "$ROOT_DIR/prepare/prepare.sh" \
    --all \
    --data-dir "$DATA_DIR" \
    --workers "$WORKERS"
