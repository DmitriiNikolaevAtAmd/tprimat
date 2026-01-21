#!/bin/bash
# Run Primus training for Llama 3.1 8B and capture logs for benchmarking

set -e

echo "╔════════════════════════════════════════════════════════════╗"
echo "║        Primus Training: Llama 3.1 8B                      ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Get script directory
TPRIMAT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load configuration from config.yaml
echo "[*] Loading configuration from config.yaml..."
if [ -f "$TPRIMAT_PATH/config_to_shell.py" ]; then
    eval "$(python3 "$TPRIMAT_PATH/config_to_shell.py")"
    echo "[+] Configuration loaded"
else
    echo "[!] config_to_shell.py not found, using defaults"
fi
echo ""

# Configuration (with fallbacks to environment or defaults from config.yaml)
PRIMUS_PATH="${PRIMUS_PATH:-${CONFIG_PRIMUS_PATH:-/workspace/Primus}}"
MODEL="llama"
CONFIG_FILE="${CONFIG_LLAMA_PRIMUS_CONFIG:-examples/megatron/configs/MI300X/llama3.1_8B-BF16-pretrain.yaml}"
TRAIN_ITERS="${TRAIN_ITERS:-${CONFIG_TRAIN_ITERS:-10}}"
OUTPUT_DIR="${CONFIG_OUTPUT_DIR:-$TPRIMAT_PATH/output}"
# Ensure OUTPUT_DIR is absolute and normalized
if [[ "$OUTPUT_DIR" != /* ]]; then
    OUTPUT_DIR="$TPRIMAT_PATH/$OUTPUT_DIR"
fi
# Normalize the path to remove ./ and ../ components
OUTPUT_DIR="$(cd "$(dirname "$OUTPUT_DIR")" 2>/dev/null && pwd)/$(basename "$OUTPUT_DIR")" || OUTPUT_DIR="$TPRIMAT_PATH/output"
NUM_GPUS="${CONFIG_AMD_NUM_GPUS:-8}"
GLOBAL_BATCH_SIZE="${CONFIG_GLOBAL_BATCH_SIZE:-128}"
SEQ_LENGTH="${CONFIG_SEQ_LENGTH:-2048}"

# Parallelism parameters for AMD
TP="${CONFIG_LLAMA_AMD_TP:-1}"
PP="${CONFIG_LLAMA_AMD_PP:-1}"
GACC="${CONFIG_LLAMA_AMD_GACC:-16}"

# Profiling parameters
PROF_ENABLED="${CONFIG_PROF_ENABLED:-false}"
PROF_WAIT="${CONFIG_PROF_WAIT:-1}"
PROF_WARMUP="${CONFIG_PROF_WARMUP:-1}"
PROF_ACTIVE="${CONFIG_PROF_ACTIVE:-5}"
PROF_START=$((PROF_WAIT + PROF_WARMUP))
PROF_STOP=$((PROF_START + PROF_ACTIVE))

# Optimization parameters
ACT_CHECKPOINT="${CONFIG_AMD_ACT_CHECKPOINT:-false}"
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'

# Determine number of layers for activation checkpointing
NLAYERS=32
if [[ "$MODEL" == "qwen" ]]; then NLAYERS=28; fi

# Optimizer parameters from config.yaml
LEARNING_RATE="${CONFIG_LEARNING_RATE:-3.0e-4}"
MIN_LEARNING_RATE="${CONFIG_MIN_LEARNING_RATE:-3.0e-5}"
WARMUP_STEPS="${CONFIG_WARMUP_STEPS:-10}"
WEIGHT_DECAY="${CONFIG_WEIGHT_DECAY:-0.1}"

# Cleanup function to remove temporary artifacts (DISABLED - logs are kept)
cleanup() {
    echo ""
    echo "  * Cleanup disabled - all logs and artifacts are preserved"
    # Uncomment below to enable cleanup:
    # rm -f "$OUTPUT_DIR"/training_main_*.log
    # rm -f "$OUTPUT_DIR"/primus_training_*.log
    # rm -f "$OUTPUT_DIR"/*.yaml
    # find "$OUTPUT_DIR" -name "*.json" -not -name "benchmark_*" -not -name "config.json" -mmin -5 -delete 2>/dev/null || true
}

# Note: Pre-existing log deletion is disabled to preserve all training history
# Uncomment the lines below if you want to remove old logs before each run:
# rm -f "$OUTPUT_DIR"/training_main_${MODEL}.log
# rm -f "$OUTPUT_DIR"/primus_training_${MODEL}.log

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Check if Primus exists
if [ ! -d "$PRIMUS_PATH" ]; then
    echo "  x Primus not found at $PRIMUS_PATH"
    echo ""
    echo "Please set PRIMUS_PATH environment variable:"
    echo "  export PRIMUS_PATH=/path/to/Primus"
    echo ""
    exit 1
fi

# Check if config exists
if [ ! -f "$PRIMUS_PATH/$CONFIG_FILE" ]; then
    echo "  x Config file not found: $PRIMUS_PATH/$CONFIG_FILE"
    echo ""
    echo "Available configs:"
    ls -1 "$PRIMUS_PATH/examples/megatron/configs/MI300X/" 2>/dev/null | grep -i llama || echo "  (none found)"
    echo ""
    exit 1
fi

echo "  * Primus Path: $PRIMUS_PATH"
echo "  * Config: $CONFIG_FILE"
echo "  * Training Iterations: $TRAIN_ITERS"
echo "  * Output: $OUTPUT_DIR"
echo "  * Num GPUs: $NUM_GPUS"
echo "  * Global Batch Size: $GLOBAL_BATCH_SIZE"
echo "  * Sequence Length: $SEQ_LENGTH"
echo "  * Parallelism: TP=$TP, PP=$PP, GradAccum=$GACC"
echo "  * Learning Rate: $LEARNING_RATE"
echo "  * Min Learning Rate: $MIN_LEARNING_RATE"
echo "  * Warmup Steps: $WARMUP_STEPS"
echo ""

# Log file configuration (use absolute paths to avoid issues when changing directories)
# To customize log file name/path, set LOG_FILE and BACKUP_LOG environment variables before running:
#   export LOG_FILE="/custom/path/my_training.log"
#   export BACKUP_LOG="/custom/path/my_training_backup.log"
LOG_FILE="${LOG_FILE:-$(cd "$OUTPUT_DIR" && pwd)/training_main_${MODEL}.log}"
BACKUP_LOG="${BACKUP_LOG:-$(cd "$OUTPUT_DIR" && pwd)/primus_training_${MODEL}.log}"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  * Starting training..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Log files:"
echo "  Primary: $LOG_FILE"
echo "  Backup:  $BACKUP_LOG"
echo ""

# Change to Primus directory
cd "$PRIMUS_PATH"

# Export config
# Create a patched config in the output directory to apply parallelism settings
PATCHED_CONFIG="$OUTPUT_DIR/$(basename "$CONFIG_FILE")"
cp "$PRIMUS_PATH/$CONFIG_FILE" "$PATCHED_CONFIG"

echo "  * Patching config with:"
echo "    - Parallelism: TP=$TP, PP=$PP, GradAccum=$GACC"
echo "    - Memory logging: enabled"
echo "    - FP8 precision: hybrid mode"
echo "    - Activation checkpointing: $ACT_CHECKPOINT"
# Use python to patch YAML reliably if possible, otherwise sed
if python3 -c "import yaml" 2>/dev/null; then
    python3 -c "
import yaml
with open('$PATCHED_CONFIG', 'r') as f:
    config = yaml.safe_load(f)
config['tensor_model_parallel_size'] = $TP
config['pipeline_model_parallel_size'] = $PP
if 'gradient_accumulation_steps' in config:
    config['gradient_accumulation_steps'] = $GACC

# Add profiling settings to YAML
if '${PROF_ENABLED}' == 'true':
    config['profile'] = True
    config['profile_step_start'] = $PROF_START
    config['profile_step_stop'] = $PROF_STOP
    config['profile_export_path'] = '$OUTPUT_DIR'

# Add memory optimizations
if '${ACT_CHECKPOINT}' == 'true':
    config['recompute_activations'] = True
    config['recompute_granularity'] = 'full'
    config['recompute_method'] = 'uniform'
    config['recompute_num_layers'] = $NLAYERS
    
config['use_distributed_optimizer'] = True
config['use_flash_attn'] = True
config['use_fused_rmsnorm'] = True
config['fp32_residual_connection'] = False

# Enable memory logging (for benchmarking)
config['log_memory_usage'] = True
config['log_interval'] = 1  # Log every step

# Enable FP8 precision (from config.yaml platform_optimizations.amd)
config['fp8'] = 'hybrid'
config['fp8_param'] = True

with open('$PATCHED_CONFIG', 'w') as f:
    yaml.dump(config, f)
"
else
    # Fallback to sed if pyyaml is not available
    # Use a more portable sed approach for different OSes
    sed "s/tensor_model_parallel_size:.*/tensor_model_parallel_size: $TP/" "$PATCHED_CONFIG" > "$PATCHED_CONFIG.tmp" && mv "$PATCHED_CONFIG.tmp" "$PATCHED_CONFIG"
    sed "s/pipeline_model_parallel_size:.*/pipeline_model_parallel_size: $PP/" "$PATCHED_CONFIG" > "$PATCHED_CONFIG.tmp" && mv "$PATCHED_CONFIG.tmp" "$PATCHED_CONFIG"
    sed "s/gradient_accumulation_steps:.*/gradient_accumulation_steps: $GACC/" "$PATCHED_CONFIG" > "$PATCHED_CONFIG.tmp" && mv "$PATCHED_CONFIG.tmp" "$PATCHED_CONFIG"
    
    if [ "$PROF_ENABLED" = "true" ]; then
        echo "profile: true" >> "$PATCHED_CONFIG"
        echo "profile_step_start: $PROF_START" >> "$PATCHED_CONFIG"
        echo "profile_step_stop: $PROF_STOP" >> "$PATCHED_CONFIG"
        echo "profile_export_path: $OUTPUT_DIR" >> "$PATCHED_CONFIG"
        # Disable heavy profiling features for AMD TP=1 to avoid OOM
        echo "profile_record_shapes: false" >> "$PATCHED_CONFIG"
        echo "profile_memory: false" >> "$PATCHED_CONFIG"
    fi
fi

export EXP="$PATCHED_CONFIG"

# Run training and capture logs
echo "Running: bash ./examples/run_pretrain.sh --train_iters $TRAIN_ITERS --lr $LEARNING_RATE --min_lr $MIN_LEARNING_RATE --lr_warmup_iters $WARMUP_STEPS --lr_decay_style cosine --lr_decay_iters $TRAIN_ITERS --weight_decay $WEIGHT_DECAY"

bash ./examples/run_pretrain.sh \
    --train_iters $TRAIN_ITERS \
    --lr $LEARNING_RATE \
    --min_lr $MIN_LEARNING_RATE \
    --lr_warmup_iters $WARMUP_STEPS \
    --lr_decay_style cosine \
    --lr_decay_iters $TRAIN_ITERS \
    --weight_decay $WEIGHT_DECAY \
    2>&1 | tee "$LOG_FILE" "$BACKUP_LOG" > /dev/null
EXIT_CODE=${PIPESTATUS[0]}

# Cleanup and rename profile traces for AMD
if [ "$PROF_ENABLED" = "true" ]; then
    echo "  * Cleaning up profile traces..."
    # Primus/Megatron usually saves traces with rank/timestamp in the name
    # We want to match the NVIDIA layout: profile_rocm_llama_PARALLEL.pt.trace.json
    STRATEGY="${PARALLEL:-unknown}"
    TARGET_NAME="profile_rocm_${MODEL}_${STRATEGY}.pt.trace.json"
    
    # Find the largest json file in output dir created in the last minute (likely the trace)
    # or look for files containing 'trace' and 'json'
    LATEST_TRACE=$(find "$OUTPUT_DIR" -name "*.json" -not -name "benchmark_*" -not -name "config.json" -newer "$LOG_FILE" | head -1)
    if [ -n "$LATEST_TRACE" ]; then
        mv "$LATEST_TRACE" "$OUTPUT_DIR/$TARGET_NAME"
        echo "[+] Profile renamed to: $TARGET_NAME"
    fi
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ $EXIT_CODE -eq 0 ]; then
    echo "  + Training completed successfully!"
    echo ""
    echo "Log saved to:"
    echo "  $LOG_FILE"
    echo "  $BACKUP_LOG"
    echo ""
    
    # Automatically extract metrics
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  * Extracting metrics..."
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    
    cd "$TPRIMAT_PATH"
    
    # Get parallel strategy from environment (if set)
    PARALLEL_STRATEGY="${PARALLEL:-unknown}"
    
    python3 extract_primus_metrics.py \
        --log-file "$LOG_FILE" \
        --model-name "$MODEL" \
        --output "$OUTPUT_DIR/benchmark_rocm_${MODEL}.json" \
        --num-gpus "$NUM_GPUS" \
        --global-batch-size "$GLOBAL_BATCH_SIZE" \
        --sequence-length "$SEQ_LENGTH" \
        --parallel-strategy "$PARALLEL_STRATEGY"
    
    EXTRACT_EXIT=$?
    if [ $EXTRACT_EXIT -eq 0 ]; then
        echo ""
        echo "  + Metrics extracted successfully!"
        echo ""
        echo "Results saved to: $OUTPUT_DIR/benchmark_rocm_${MODEL}.json"
        echo ""
        
        # Note: Cleanup is disabled - all logs are preserved
        # Uncomment the line below to enable cleanup after successful extraction:
        # cleanup
        
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"


        echo "  * Next Steps:"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo ""
        echo "1. Run on NVIDIA system (if not done):"
        echo "   ./benchmark.py --model $MODEL"
        echo ""
        echo "2. Compare results:"
        echo "   python3 compare_results.py"
        echo ""
        echo "3. View enhanced metrics:"
        echo "   python3 compare_with_enhanced_metrics.py"
        echo ""
    else
        echo "  x Metric extraction failed"
        echo "   Check the log file manually: $LOG_FILE"
        EXIT_CODE=$EXTRACT_EXIT
    fi
else
    echo "  x Training failed with exit code $EXIT_CODE"
    echo ""
    echo "Check the log file for errors:"
    echo "  $LOG_FILE"
    echo ""
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Exit with the training exit code to properly propagate success/failure
exit $EXIT_CODE
