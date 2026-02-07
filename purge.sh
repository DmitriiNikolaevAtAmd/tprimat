#!/bin/bash
# Usage: ./purge.sh [--keep-data]

rm -rf cache __pycache__ torchelastic_* *.out *.err *.zip

if [ "$1" != "--keep-data" ]; then
    rm -rf /data/tprimat
fi