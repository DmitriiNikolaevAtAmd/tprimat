#!/bin/bash
python3 benchmark.py --parallel "maximum_performance" --output-dir all-output/output-00
sleep 10
python3 benchmark.py --parallel "identical_config" --output-dir all-output/output-01
sleep 10
python3 benchmark.py --parallel "memory_optimized" --output-dir all-output/output-02
sleep 10
python3 benchmark.py --parallel "minimal_communication" --output-dir all-output/output-03
sleep 10
python3 benchmark.py --parallel "balanced" --output-dir all-output/output-04
