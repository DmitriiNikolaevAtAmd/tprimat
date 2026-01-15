#!/bin/bash
python3 benchmark.py --parallel "maximum_performance" --output-dir output-00
sleep 10
python3 benchmark.py --parallel "identical_config" --output-dir output-01
sleep 10
python3 benchmark.py --parallel "memory_optimized" --output-dir output-02
sleep 10
python3 benchmark.py --parallel "minimal_communication" --output-dir output-03
sleep 10
python3 benchmark.py --parallel "balanced" --output-dir output-04
