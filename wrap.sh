#!/bin/bash
set -e

if [ ! -d "output" ]; then
    exit 1
fi

find output -type f -name "*.log" -delete
find output -type d -empty -delete

if [ -z "$(find output -type f 2>/dev/null)" ]; then
    exit 1
fi

zip -q -r output.zip output

exit 0
