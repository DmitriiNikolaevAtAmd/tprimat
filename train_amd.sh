#!/bin/bash
set -e
./prepare/prepare_data.sh
./train/train_amd.sh
./evaluate/compare_amd.sh
