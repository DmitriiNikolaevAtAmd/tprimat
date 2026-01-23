#!/bin/bash

# Remove output and cache directories
rm -rf output/ __pycache__/

# Alternative: Keep output files but remove empty directories
# find output -type d -empty -delete 2>/dev/null
