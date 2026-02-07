#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../config.env"

export DATA_DIR
export DATA_SAMPLES
export TRAIN_SPLIT
export SL

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

# Encode C4 dataset with train/test split
echo "Encoding C4 dataset..."
python3 "${SCRIPT_DIR}/encode_data.py" \
    --input "${DATA_DIR}/allenai-c4.jsonl" \
    --output-dir "${DATA_DIR}" \
    --output-name "c4" \
    --seq-length "${SL}" \
    --max-samples "${DATA_SAMPLES}" \
    --train-split "${TRAIN_SPLIT}"

# Encode BookCorpus dataset with train/test split
echo "Encoding BookCorpus dataset..."
python3 "${SCRIPT_DIR}/encode_data.py" \
    --input "${DATA_DIR}/bookcorpus.jsonl" \
    --output-dir "${DATA_DIR}" \
    --output-name "bc" \
    --seq-length "${SL}" \
    --max-samples "${DATA_SAMPLES}" \
    --train-split "${TRAIN_SPLIT}"

# Verify both datasets
python3 "${SCRIPT_DIR}/verify_data.py" \
    --input-dir "${DATA_DIR}"
