#!/bin/bash
OUTPUT_DIR=output/output-00 PARALLEL=maximum_performance ./run_primus_model.sh
sleep 10
OUTPUT_DIR=output/output-01 PARALLEL=identical_config ./run_primus_model.sh
sleep 10
OUTPUT_DIR=output/output-02 PARALLEL=memory_optimized ./run_primus_model.sh
sleep 10
OUTPUT_DIR=output/output-03 PARALLEL=minimal_communication ./run_primus_model.sh
sleep 10
OUTPUT_DIR=output/output-04 PARALLEL=balanced ./run_primus_model.sh