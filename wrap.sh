#!/bin/bash

set -e

if [ ! -d "output" ]; then
    echo "Error: output directory does not exist"
    exit 1
fi

# Check if any JSON files exist
JSON_COUNT=$(ls output/*.json 2>/dev/null | wc -l)
if [ "$JSON_COUNT" -eq 0 ]; then
    echo ""
    echo "Error: No JSON files found in output/"
    exit 1
fi

# Archive all results (with output folder structure preserved)
echo "Archiving output directory..."
zip -q -r output.zip output/*.json

echo ""
echo "✓ Created output.zip"
echo ""
echo "Summary:"
FILE_COUNT=$(unzip -l output.zip | grep -c "output/.*\.json$" || echo "0")
echo "  output.zip - $FILE_COUNT file(s) in output/"
echo ""
echo "Archive contents:"
unzip -l output.zip | grep "output/" | awk '{print "  " $4}'

exit 0
