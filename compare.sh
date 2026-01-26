#!/bin/bash
set -e

OUTPUT_BASE=${1:-"output"}
mkdir -p "$OUTPUT_BASE"

declare -A CONFIGS
CONFIGS["00"]="maximum_performance"
CONFIGS["01"]="truly_identical"
CONFIGS["02"]="memory_optimized"
CONFIGS["03"]="minimal_communication"
CONFIGS["04"]="balanced"

OUTPUT_DIRS=($(find . -maxdepth 2 -type d -name "output-*" | sort))

if [ ${#OUTPUT_DIRS[@]} -eq 0 ]; then
    exit 1
fi

for DIR in "${OUTPUT_DIRS[@]}"; do
    if [ -d "$DIR" ]; then
        DIR_NAME=$(basename "$DIR")
        INDEX="${DIR_NAME#output-}"
        OUTPUT_FILE="${OUTPUT_BASE}/compare-${INDEX}.png"
        
        json_count=$(find "$DIR" -name "train_*.json" 2>/dev/null | wc -l)
        
        if [ "$json_count" -eq 0 ]; then
            echo "  ⚠ Skipping $DIR (no train_*.json files found)"
            continue
        fi
        
        echo "  📊 Comparing results from $DIR ($json_count JSON file(s))"
        python3 compare.py --results-dir "$DIR" --output "$OUTPUT_FILE"
    fi
done

summary_file="${OUTPUT_BASE}/configurations_summary.txt"
{
echo "Configuration Summary"
echo "===================="
echo ""

for DIR in "${OUTPUT_DIRS[@]}"; do
    DIR_NAME=$(basename "$DIR")
    INDEX="${DIR_NAME#output-}"
    config_name="${CONFIGS[$INDEX]}"
    
    if [ -n "$config_name" ]; then
        echo "Config ${INDEX}: ${config_name}"
        
        case "$config_name" in
            "maximum_performance")
                echo "  Llama (NVIDIA): TP=4, PP=1, DP=2"
                echo "  Llama (AMD):    TP=1, PP=1, DP=8"
                echo "  Qwen (NVIDIA):  TP=4, PP=2, DP=1"
                echo "  Qwen (AMD):     TP=1, PP=1, DP=8"
                ;;
            "truly_identical")
                echo "  Llama: TP=4, PP=1, DP=2 (both platforms and models)"
                echo "  Qwen:  TP=4, PP=1, DP=2 (both platforms and models)"
                ;;
            "memory_optimized")
                echo "  Llama (NVIDIA): TP=8, PP=1, DP=1"
                echo "  Llama (AMD):    TP=2, PP=1, DP=4"
                echo "  Qwen (NVIDIA):  TP=8, PP=1, DP=1"
                echo "  Qwen (AMD):     TP=2, PP=1, DP=4"
                ;;
            "minimal_communication")
                echo "  All: TP=1, PP=1, DP=8 (both platforms)"
                ;;
            "balanced")
                echo "  Llama: TP=2, PP=1, DP=4 (both platforms)"
                echo "  Qwen:  TP=2, PP=2, DP=2 (both platforms)"
                ;;
        esac
        echo ""
    fi
done
} > "$summary_file"
