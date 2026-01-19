#!/bin/bash
# Generate comparison plots for all configurations
# Usage: ./compare_all_configs.sh [output_directory]

set -e

OUTPUT_BASE=${1:-"all_outputs"}
mkdir -p "$OUTPUT_BASE"

echo "╔════════════════════════════════════════════════════════════╗"
echo "║         Generate Comparison Plots for All Configs         ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Configuration names
declare -A CONFIGS
CONFIGS["00"]="maximum_performance"
CONFIGS["01"]="truly_identical"
CONFIGS["02"]="memory_optimized"
CONFIGS["03"]="minimal_communication"
CONFIGS["04"]="balanced"

# Check which configurations exist
available_configs=()
for config_num in 00 01 02 03 04; do
    if [ -d "output-${config_num}" ]; then
        available_configs+=("$config_num")
    fi
done

if [ ${#available_configs[@]} -eq 0 ]; then
    echo "❌ No configuration directories found!"
    echo "   Expected: output-00/, output-01/, output-02/, output-03/, output-04/"
    echo ""
    echo "   Run benchmarks first using:"
    echo "   ./run_all_configs.sh"
    exit 1
fi

echo "📂 Found ${#available_configs[@]} configuration(s): ${available_configs[@]}"
echo ""

# Generate comparison plot for each configuration
for config_num in "${available_configs[@]}"; do
    config_name="${CONFIGS[$config_num]}"
    config_dir="output-${config_num}"
    output_file="${OUTPUT_BASE}/compare-${config_num}.png"
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📊 Configuration ${config_num}: ${config_name}"
    echo "   Input:  ${config_dir}/"
    echo "   Output: ${output_file}"
    
    # Check if benchmark files exist
    benchmark_count=$(find "$config_dir" -name "benchmark_*.json" 2>/dev/null | wc -l)
    if [ "$benchmark_count" -eq 0 ]; then
        echo "   ⚠️  No benchmark files found in ${config_dir}/, skipping..."
        continue
    fi
    
    echo "   Found: ${benchmark_count} benchmark file(s)"
    
    # Generate comparison plot
    python3 compare.py --results-dir "$config_dir"
    
    # Move to output directory
    if [ -f "compare.png" ]; then
        mv compare.png "$output_file"
        echo "   ✅ Generated: ${output_file}"
    else
        echo "   ❌ Failed to generate plot"
    fi
    echo ""
done

echo "╔════════════════════════════════════════════════════════════╗"
echo "║                    PLOTS GENERATED                         ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "📊 Comparison plots saved to: ${OUTPUT_BASE}/"
ls -lh "${OUTPUT_BASE}"/compare-*.png 2>/dev/null || echo "   (No plots generated)"
echo ""

# Create a summary file
summary_file="${OUTPUT_BASE}/configurations_summary.txt"
echo "Configuration Summary" > "$summary_file"
echo "====================" >> "$summary_file"
echo "" >> "$summary_file"

for config_num in "${available_configs[@]}"; do
    config_name="${CONFIGS[$config_num]}"
    echo "Config ${config_num}: ${config_name}" >> "$summary_file"
    
    # Extract parallelism info from config.yaml for this configuration
    case "$config_name" in
        "maximum_performance")
            echo "  Llama (NVIDIA): TP=4, PP=1, DP=2" >> "$summary_file"
            echo "  Llama (AMD):    TP=1, PP=1, DP=8" >> "$summary_file"
            echo "  Qwen (NVIDIA):  TP=4, PP=2, DP=1" >> "$summary_file"
            echo "  Qwen (AMD):     TP=1, PP=1, DP=8" >> "$summary_file"
            ;;
        "truly_identical")
            echo "  Llama: TP=4, PP=1, DP=2 (both platforms and models)" >> "$summary_file"
            echo "  Qwen:  TP=4, PP=1, DP=2 (both platforms and models)" >> "$summary_file"
            ;;
        "memory_optimized")
            echo "  Llama (NVIDIA): TP=8, PP=1, DP=1" >> "$summary_file"
            echo "  Llama (AMD):    TP=2, PP=1, DP=4" >> "$summary_file"
            echo "  Qwen (NVIDIA):  TP=8, PP=1, DP=1" >> "$summary_file"
            echo "  Qwen (AMD):     TP=2, PP=1, DP=4" >> "$summary_file"
            ;;
        "minimal_communication")
            echo "  All: TP=1, PP=1, DP=8 (both platforms)" >> "$summary_file"
            ;;
        "balanced")
            echo "  Llama: TP=2, PP=1, DP=4 (both platforms)" >> "$summary_file"
            echo "  Qwen:  TP=2, PP=2, DP=2 (both platforms)" >> "$summary_file"
            ;;
    esac
    echo "" >> "$summary_file"
done

echo "📄 Configuration summary saved to: ${summary_file}"
cat "$summary_file"
echo ""

echo "🔍 To view the comparison plots:"
echo "   open ${OUTPUT_BASE}/compare-*.png"
echo ""
echo "   Or on Linux:"
echo "   xdg-open ${OUTPUT_BASE}/compare-00.png"
echo ""
