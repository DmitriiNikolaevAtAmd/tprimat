#!/bin/bash
# Run Primus training for Llama 3.1 8B and capture logs for benchmarking

set -e

echo "╔════════════════════════════════════════════════════════════╗"
echo "║        Primus Training: Llama 3.1 8B                      ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Configuration
PRIMUS_PATH="${PRIMUS_PATH:-/workspace/Primus}"
TPRIMAT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="llama"
CONFIG_FILE="examples/megatron/configs/MI300X/llama3.1_8B-pretrain.yaml"
TRAIN_ITERS="${TRAIN_ITERS:-10}"
OUTPUT_DIR="$TPRIMAT_PATH/output"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Check if Primus exists
if [ ! -d "$PRIMUS_PATH" ]; then
    echo "❌ Primus not found at $PRIMUS_PATH"
    echo ""
    echo "Please set PRIMUS_PATH environment variable:"
    echo "  export PRIMUS_PATH=/path/to/Primus"
    echo ""
    exit 1
fi

# Check if config exists
if [ ! -f "$PRIMUS_PATH/$CONFIG_FILE" ]; then
    echo "❌ Config file not found: $PRIMUS_PATH/$CONFIG_FILE"
    echo ""
    echo "Available configs:"
    ls -1 "$PRIMUS_PATH/examples/megatron/configs/MI300X/" 2>/dev/null | grep -i llama || echo "  (none found)"
    echo ""
    exit 1
fi

echo "📂 Primus Path: $PRIMUS_PATH"
echo "📄 Config: $CONFIG_FILE"
echo "📊 Training Iterations: $TRAIN_ITERS"
echo "📁 Output: $OUTPUT_DIR"
echo ""

# Log file
LOG_FILE="$OUTPUT_DIR/training_${MODEL}.log"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_LOG="$OUTPUT_DIR/primus_training_${MODEL}_${TIMESTAMP}.log"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 Starting training..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Log files:"
echo "  Primary: $LOG_FILE"
echo "  Backup:  $BACKUP_LOG"
echo ""

# Change to Primus directory
cd "$PRIMUS_PATH"

# Export config
export EXP="$CONFIG_FILE"

# Run training and capture logs
echo "Running: bash ./examples/run_pretrain.sh --train_iters $TRAIN_ITERS"
echo ""

bash ./examples/run_pretrain.sh --train_iters $TRAIN_ITERS 2>&1 | tee "$LOG_FILE" "$BACKUP_LOG"

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Training completed successfully!"
    echo ""
    echo "Log saved to:"
    echo "  $LOG_FILE"
    echo "  $BACKUP_LOG"
    echo ""
    
    # Automatically extract metrics
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📊 Extracting metrics..."
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    
    cd "$TPRIMAT_PATH"
    
    python3 extract_primus_metrics.py \
        --log-file "$LOG_FILE" \
        --model-name "$MODEL" \
        --num-gpus 8 \
        --global-batch-size 128 \
        --sequence-length 2048
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ Metrics extracted successfully!"
        echo ""
        echo "Results saved to: $OUTPUT_DIR/benchmark_rocm_${MODEL}.json"
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "🎯 Next Steps:"
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
        echo "❌ Metric extraction failed"
        echo "   Check the log file manually: $LOG_FILE"
    fi
else
    echo "❌ Training failed with exit code $EXIT_CODE"
    echo ""
    echo "Check the log file for errors:"
    echo "  $LOG_FILE"
    echo ""
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
