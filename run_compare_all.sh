#!/bin/bash
# sequentially build all charts for output-00 to output-04

set -e

# list of directories to process
# Finds all output-XX directories, even if they are nested under 'output/'
OUTPUT_DIRS=($(find . -maxdepth 2 -type d -name "output-*" | sort))

if [ ${#OUTPUT_DIRS[@]} -eq 0 ]; then
    echo "[X] No output-XX directories found!"
    exit 1
fi

echo "╔════════════════════════════════════════════════════════════╗"
echo "║        Building Sequentially All Comparison Charts        ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo "Found ${#OUTPUT_DIRS[@]} directories: ${OUTPUT_DIRS[*]}"
echo ""

for DIR in "${OUTPUT_DIRS[@]}"; do
    if [ -d "$DIR" ]; then
        echo "════════════════════════════════════════════════════════════"
        echo "Processing: $DIR"
        echo "════════════════════════════════════════════════════════════"
        
        # Determine index (XX) from directory name like output-XX or output/output-XX
        DIR_NAME=$(basename "$DIR")
        INDEX="${DIR_NAME#output-}"
        
        # Place the generated chart in the 'output/' directory
        OUTPUT_FILE="output/compare-${INDEX}.png"
        
        # Ensure the output parent directory exists
        mkdir -p "output"
        
        echo "[#] Generating chart: $OUTPUT_FILE from $DIR"
        python3 compare.py --results-dir "$DIR" --output "$OUTPUT_FILE"
        echo "[OK] Created $OUTPUT_FILE"

        echo ""
    else
        echo "[!] Directory $DIR not found, skipping..."
    fi
done

echo "════════════════════════════════════════════════════════════"
echo "SUMMARY"
echo "════════════════════════════════════════════════════════════"
ls -lh output/compare-*.png 2>/dev/null || echo "No charts generated in output/."
echo ""
echo "Done."
