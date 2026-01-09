#!/bin/bash
# Run both "Maximum Performance" and "Identical Config" comparisons on AMD

set -e

echo "╔════════════════════════════════════════════════════════════╗"
echo "║     AMD Dual Benchmark: Max Performance + Fair Config     ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Configuration
PRIMUS_PATH="${PRIMUS_PATH:-/workspace/Primus}"
TPRIMAT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="$TPRIMAT_PATH/output"
TRAIN_ITERS="${TRAIN_ITERS:-10}"

# Model configurations
declare -A MODEL_CONFIGS
MODEL_CONFIGS[llama]="examples/megatron/configs/MI300X/llama3.1_8B-pretrain.yaml"
MODEL_CONFIGS[qwen]="examples/megatron/configs/MI300X/qwen2.5_7B-pretrain.yaml"

# Check if Primus exists
if [ ! -d "$PRIMUS_PATH" ]; then
    echo "❌ Primus not found at $PRIMUS_PATH"
    echo "   Set PRIMUS_PATH environment variable to correct location"
    exit 1
fi

echo "📂 Primus Path: $PRIMUS_PATH"
echo "📂 Output Path: $OUTPUT_DIR"
echo ""

# Function to run Primus training
run_primus_training() {
    local model=$1
    local config=$2
    local suffix=$3
    local log_file="$OUTPUT_DIR/primus_training_${model}_${suffix}.log"
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🚀 Running $model ($suffix configuration)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    cd "$PRIMUS_PATH"
    export EXP="$config"
    
    echo "Config: $EXP"
    echo "Log: $log_file"
    echo ""
    
    bash ./examples/run_pretrain.sh --train_iters $TRAIN_ITERS 2>&1 | tee "$log_file"
    
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo "✅ Training completed"
    else
        echo "❌ Training failed"
        return 1
    fi
    
    cd "$TPRIMAT_PATH"
}

# Function to extract metrics
extract_metrics() {
    local model=$1
    local log_file=$2
    local suffix=$3
    
    echo ""
    echo "📊 Extracting metrics from $model ($suffix)..."
    
    python3 extract_primus_metrics.py \
        --log-file "$log_file" \
        --model-name "${model}_${suffix}" \
        --num-gpus 8 \
        --global-batch-size 128 \
        --sequence-length 2048
    
    if [ $? -eq 0 ]; then
        echo "✅ Metrics extracted"
    else
        echo "❌ Metric extraction failed"
        return 1
    fi
}

echo "════════════════════════════════════════════════════════════"
echo "PHASE 1: Maximum Performance (Platform-Optimized Configs)"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "This run uses AMD-optimized configuration:"
echo "  • Likely TP=1 (full model per GPU)"
echo "  • Leverages MI300X's 192GB memory"
echo "  • Minimal inter-GPU communication"
echo ""

# Run max performance benchmarks
for model in llama qwen; do
    config="${MODEL_CONFIGS[$model]}"
    
    if [ -z "$config" ]; then
        echo "⚠️  No config found for $model, skipping..."
        continue
    fi
    
    # Check if config file exists
    if [ ! -f "$PRIMUS_PATH/$config" ]; then
        echo "⚠️  Config file not found: $PRIMUS_PATH/$config"
        echo "   Skipping $model..."
        continue
    fi
    
    # Run training with max performance config
    run_primus_training "$model" "$config" "max_perf"
    
    # Extract metrics
    log_file="$OUTPUT_DIR/primus_training_${model}_max_perf.log"
    if [ -f "$log_file" ]; then
        extract_metrics "$model" "$log_file" "max_perf"
    fi
    
    echo ""
done

echo ""
echo "════════════════════════════════════════════════════════════"
echo "PHASE 2: Identical Configuration (Fair Hardware Comparison)"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "⚠️  MANUAL STEP REQUIRED!"
echo ""
echo "To run fair comparison, you need to:"
echo "1. Create modified config files with TP=4 (matching NVIDIA)"
echo "2. Set precision to match NVIDIA (FP8 or BF16)"
echo "3. Use same micro_batch_size=1"
echo ""
echo "Example config modifications needed:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  tensor_model_parallel_size: 4  # Was likely 1"
echo "  pipeline_model_parallel_size: 1"
echo "  micro_batch_size: 1"
echo "  global_batch_size: 128"
echo "  precision: bf16  # or fp8 if supported"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Create these configs at:"
echo "  $PRIMUS_PATH/examples/megatron/configs/MI300X/"
echo ""
echo "Then run this script again with FAIR_CONFIG=1:"
echo "  FAIR_CONFIG=1 ./run_amd_dual_comparison.sh"
echo ""

if [ "${FAIR_CONFIG:-0}" == "1" ]; then
    echo "Running fair comparison (TP=4)..."
    echo ""
    
    # TODO: Add your fair comparison config paths here
    # MODEL_CONFIGS_FAIR[llama]="examples/megatron/configs/MI300X/llama3.1_8B-pretrain-tp4.yaml"
    
    echo "⚠️  Please update this script with your TP=4 config file paths"
    echo ""
fi

echo "════════════════════════════════════════════════════════════"
echo "✅ Benchmarking Complete!"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "Results saved to: $OUTPUT_DIR/"
echo ""
echo "📊 To compare with NVIDIA results:"
echo "   cd $TPRIMAT_PATH"
echo "   python3 compare_results.py"
echo ""
echo "📝 Files generated:"
ls -lh "$OUTPUT_DIR"/benchmark_*.json 2>/dev/null || echo "   (Run extract_primus_metrics.py if logs exist)"
echo ""
