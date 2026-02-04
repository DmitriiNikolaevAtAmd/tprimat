#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../config.env"

export DATA_DIR
export DATA_SAMPLES

python3 "${SCRIPT_DIR}/fetch_deps.py" \
    --samples "${DATA_SAMPLES}" \
    --output "${DATA_DIR}/allenai-c4-raw.jsonl" \
    --bookcorpus-output "${DATA_DIR}/bookcorpus-raw.jsonl"

python3 "${SCRIPT_DIR}/clean_data.py" \
    --input "${DATA_DIR}/allenai-c4-raw.jsonl" \
    --output "${DATA_DIR}/allenai-c4.jsonl"

python3 "${SCRIPT_DIR}/clean_data.py" \
    --input "${DATA_DIR}/bookcorpus-raw.jsonl" \
    --output "${DATA_DIR}/bookcorpus.jsonl"

python3 "${SCRIPT_DIR}/encode_data.py" \
    --input "${DATA_DIR}/allenai-c4.jsonl" \
    --output-dir "${DATA_DIR}" \
    --seq-length "${SEQ_LEN}"

python3 "${SCRIPT_DIR}/verify_data.py" \
    --input-dir "${DATA_DIR}"
