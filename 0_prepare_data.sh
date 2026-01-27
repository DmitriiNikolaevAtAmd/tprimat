#!/bin/bash
set -e

python3 1_fetch_deps.py
python3 2_clean_data.py
python3 3_encode_data.py
python3 4_verify_data.py
