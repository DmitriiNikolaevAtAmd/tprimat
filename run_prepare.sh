#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$SCRIPT_DIR"

if [ -f "$ROOT_DIR/config.env" ]; then
    set -a
    source "$ROOT_DIR/config.env"
    set +a
fi

DATA_DIR="${DATA_DIR:-/data/tprimat}"
WORKERS="${WORKERS:-$(nproc 2>/dev/null || echo 4)}"

bash "$ROOT_DIR/prepare/prepare.sh" \
    --all \
    --data-dir "$DATA_DIR" \
    --workers "$WORKERS"
