#!/bin/bash

set -e

if [ -d "output" ]; then
    find output -type d -empty -delete 2>/dev/null
else
    exit 1
fi

tar -czf output.nvd.tar.gz \
    output/train_tran_*.json \
    output/train_mega_*.json \
    output/train_deep_*.json \
    output/train_nemo_*.json \
    2>/dev/null || true

tar -czf output.amd.tar.gz \
    output/train_prim_*.json \
    2>/dev/null || true

if [ ! -f "output.nvd.tar.gz" ] && [ ! -f "output.amd.tar.gz" ]; then
    exit 1
fi
