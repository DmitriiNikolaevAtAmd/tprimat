#!/bin/bash
set -e
cd "$(dirname "$0")"

./amd_prim_llama.sh
./amd_prim_qwen.sh
