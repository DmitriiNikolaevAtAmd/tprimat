#!/bin/bash
set -e
cd "$(dirname "$0")"

python3 fetch_deps.py
python3 clean_data.py
python3 encode_data.py
python3 verify_data.py
