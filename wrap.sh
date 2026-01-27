#!/bin/bash

set -e

if [ ! -d "output" ]; then
    exit 1
fi

# Check if any JSON files exist
JSON_COUNT=$(ls output/*.json 2>/dev/null | wc -l)
if [ "$JSON_COUNT" -eq 0 ]; then
    exit 1
fi

# Archive all results (with output folder structure preserved)
zip -q -r output.zip output/*.json

exit 0
