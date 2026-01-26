#!/bin/bash

set -e

if [ ! -d "output" ]; then
    echo "Error: output directory does not exist"
    exit 1
fi

# Archive NVIDIA results (without directory structure)
NVD_FILES=$(cd output && ls train_tran_*.json train_mega_*.json train_deep_*.json train_nemo_*.json 2>/dev/null || true)
if [ -n "$NVD_FILES" ]; then
    cd output
    zip -q ../nvd-output.zip $NVD_FILES
    cd ..
    echo "✓ Created nvd-output.zip"
fi

# Archive AMD results (without directory structure)
AMD_FILES=$(cd output && ls train_prim_*.json 2>/dev/null || true)
if [ -n "$AMD_FILES" ]; then
    cd output
    zip -q ../amd-output.zip $AMD_FILES
    cd ..
    echo "✓ Created amd-output.zip"
fi

if [ ! -f "nvd-output.zip" ] && [ ! -f "amd-output.zip" ]; then
    echo ""
    echo "Error: No JSON files found to archive"
    exit 1
fi

echo ""
echo "Summary:"
if [ -f "nvd-output.zip" ]; then
    FILE_COUNT=$(unzip -l nvd-output.zip | tail -1 | awk '{print $2}')
    echo "  nvd-output.zip - $FILE_COUNT file(s)"
fi
if [ -f "amd-output.zip" ]; then
    FILE_COUNT=$(unzip -l amd-output.zip | tail -1 | awk '{print $2}')
    echo "  amd-output.zip - $FILE_COUNT file(s)"
fi

exit 0
