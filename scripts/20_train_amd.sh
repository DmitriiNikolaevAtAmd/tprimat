#!/bin/bash
set -e
cd "$(dirname "$0")"

./25_train_amd_prim_llama.sh
./25_train_amd_prim_qwen.sh
