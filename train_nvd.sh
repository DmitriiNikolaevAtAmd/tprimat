#!/bin/bash
set -e
./prepare/prepare_data.sh
./train/train_nvd.sh
./evaluate/compare_nvd.sh
