#!/bin/bash
OUTPUT_DIR=output-00 TPRIMAT_PARALLEL=maximum_performance ./run_primus_model.sh
sleep 10
OUTPUT_DIR=output-01 TPRIMAT_PARALLEL=identical_config ./run_primus_model.sh
sleep 10
OUTPUT_DIR=output-02 TPRIMAT_PARALLEL=memory_optimized ./run_primus_model.sh
sleep 10
OUTPUT_DIR=output-03 TPRIMAT_PARALLEL=minimal_communication ./run_primus_model.sh
sleep 10
OUTPUT_DIR=output-04 TPRIMAT_PARALLEL=balanced ./run_primus_model.sh