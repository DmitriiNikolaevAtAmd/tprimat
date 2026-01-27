#!/bin/bash
set -e

python3 11_fetch_deps.py
python3 12_clean_data.py
python3 13_encode_data.py
python3 14_verify_data.py
