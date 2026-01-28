#!/bin/bash
set -e
./scripts/00_prepare_data.sh
./scripts/10_train_nvd.sh
./scripts/19_compare_nvd.sh
