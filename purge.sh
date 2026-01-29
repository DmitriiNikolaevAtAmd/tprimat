#!/bin/bash
# WARNING: This will delete local output and cache directories
# It will NOT delete /data/tprimat (external mounted volume)
rm -rf cache output __pycache__ torchelastic_* *.out *.err *.zip
echo "Cleaned up local cache and output directories"
echo "Note: /data/tprimat is preserved (external volume)"