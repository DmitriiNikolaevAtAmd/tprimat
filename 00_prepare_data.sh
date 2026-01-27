#!/bin/bash
set -e

python3 00_fetch_deps.py
python3 01_clean_data.py
python3 02_encode_data.py
python3 03_verify_data.py
