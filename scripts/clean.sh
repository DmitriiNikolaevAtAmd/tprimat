#!/bin/bash
# Usage: ./scripts/clean.sh [--with-data]

rm -rf cache __pycache__ torchelastic_* *.out *.err *.zip output

if [ "$1" == "--with-data" ]; then
    echo "Cleaning the data at /data/tprimat ..."
    rm -rf /data/tprimat
fi
