#!/bin/bash
# Run all 5 parallelism configurations for benchmarking
# Usage: ./run_all_configs.sh [llama|qwen|both]

set -e

MODEL=${1:-both}  # Default to both models

# Configuration names mapping to directory suffixes
declare -A CONFIGS
CONFIGS["00"]="maximum_performance"
CONFIGS["01"]="identical_config"
CONFIGS["02"]="memory_optimized"
CONFIGS["03"]="minimal_communication"
CONFIGS["04"]="balanced"

TPRIMAT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$TPRIMAT_PATH"

echo "╔════════════════════════════════════════════════════════════╗"
echo "║          TPrimat: Run All Configurations                  ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "Model(s): $MODEL"
echo ""

# Function to run a single configuration
run_config() {
    local config_num=$1
    local config_name=$2
    local model=$3
    
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🚀 Configuration $config_num: $config_name"
    echo "   Model: $model"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # Create output directory for this configuration
    output_dir="output-${config_num}"
    mkdir -p "$output_dir"
    
    # Set environment variables
    export TPRIMAT_METHODOLOGY="$config_name"
    export OUTPUT_DIR="$output_dir"
    
    # Update config.yaml to use the correct output directory
    # (This is a temporary workaround; ideally config_loader should support this)
    
    # Run the training
    if [ "$model" == "llama" ]; then
        echo "▶ Running Llama 3.1 8B with $config_name..."
        python3 pretrain_llama.py 2>&1 | tee "$output_dir/training_llama.log"
    elif [ "$model" == "qwen" ]; then
        echo "▶ Running Qwen 2.5 7B with $config_name..."
        python3 pretrain_qwen.py 2>&1 | tee "$output_dir/training_qwen.log"
    fi
    
    echo "✅ Configuration $config_num ($config_name) completed for $model"
}

# Function to generate comparison for a configuration
generate_comparison() {
    local config_num=$1
    local config_name=$2
    
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📊 Generating comparison for configuration $config_num: $config_name"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    output_dir="output-${config_num}"
    
    # Generate comparison plot from this output directory
    python3 compare.py --results-dir "$output_dir"
    
    # Move the comparison plot to all_outputs with config number
    if [ -f "compare.png" ]; then
        mv compare.png "all_outputs/compare-${config_num}.png"
        echo "✅ Comparison saved to: all_outputs/compare-${config_num}.png"
    else
        echo "⚠️  No comparison plot generated"
    fi
}

# Create all_outputs directory for storing all comparison plots
mkdir -p all_outputs

# Run all configurations
for config_num in 00 01 02 03 04; do
    config_name="${CONFIGS[$config_num]}"
    
    if [ "$MODEL" == "llama" ] || [ "$MODEL" == "both" ]; then
        run_config "$config_num" "$config_name" "llama"
    fi
    
    if [ "$MODEL" == "qwen" ] || [ "$MODEL" == "both" ]; then
        run_config "$config_num" "$config_name" "qwen"
    fi
    
    # Generate comparison after both models are done (if running both)
    if [ "$MODEL" == "both" ]; then
        generate_comparison "$config_num" "$config_name"
    fi
done

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║                  ALL BENCHMARKS COMPLETE                  ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "📁 Results saved to:"
for config_num in 00 01 02 03 04; do
    echo "   - output-${config_num}/ (${CONFIGS[$config_num]})"
done
echo ""
echo "📊 Comparison plots:"
ls -1 all_outputs/compare-*.png 2>/dev/null || echo "   (No comparison plots generated)"
echo ""
echo "🔍 To view results:"
echo "   1. Check benchmark JSON files in each output-XX/ directory"
echo "   2. View comparison plots in all_outputs/"
echo "   3. Run: python3 compare.py (for latest config)"
echo ""
