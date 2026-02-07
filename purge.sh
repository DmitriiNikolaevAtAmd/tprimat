#!/bin/bash
# Usage: ./purge.sh [--with-data]

rm -rf cache __pycache__ torchelastic_* *.out *.err *.zip

if [ "$1" == "--with-data" ]; then
    echo "Purging data at /data/tprimat ..."
    rm -rf /data/tprimat
fi