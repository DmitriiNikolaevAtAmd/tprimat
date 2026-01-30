#!/bin/bash
# Usage: ./purge.sh [--keep-data]

rm -rf cache output __pycache__ torchelastic_* *.out *.err *.zip

if [ "$1" != "--keep-data" ]; then
    rm -rf /data/tprimat
    echo "Purged everything including data"
else
    echo "Purged everything except data"
fi
