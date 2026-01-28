#!/bin/bash
set -e
cd "$(dirname "$0")"

python3 01_fetch_deps.py
python3 02_clean_data.py
python3 03_encode_data.py
python3 04_verify_data.py
