#!/bin/bash
# Clean checkpoint artifacts from training runs, keeping only logs

set -e

echo "╔════════════════════════════════════════════════════════════╗"
echo "║        Clean Training Checkpoints & Artifacts            ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="$SCRIPT_DIR/output"

# Check if Primus path is set
PRIMUS_PATH="${PRIMUS_PATH:-/workspace/Primus}"

echo "🔍 Searching for checkpoint artifacts..."
echo ""

# Function to safely remove directories
remove_dir() {
    local dir=$1
    if [ -d "$dir" ]; then
        echo "  Removing: $dir"
        rm -rf "$dir"
    fi
}

# Function to safely remove files
remove_files() {
    local pattern=$1
    local found=$(find . -name "$pattern" 2>/dev/null)
    if [ -n "$found" ]; then
        echo "  Found files matching: $pattern"
        find . -name "$pattern" -type f -delete
    fi
}

CLEANED=0

# Clean Primus checkpoints
if [ -d "$PRIMUS_PATH" ]; then
    cd "$PRIMUS_PATH"
    echo "📁 Cleaning Primus directory: $PRIMUS_PATH"
    
    # Common checkpoint directories
    for ckpt_dir in checkpoints ckpt checkpoint checkpoint_* ckpt_* experiments/checkpoints; do
        if [ -d "$ckpt_dir" ]; then
            remove_dir "$ckpt_dir"
            CLEANED=$((CLEANED + 1))
        fi
    done
    
    # Note: TensorBoard and profiling artifacts are preserved
    
    echo ""
fi

# Clean tprimat output directory (except logs and JSON results)
if [ -d "$OUTPUT_DIR" ]; then
    cd "$OUTPUT_DIR"
    echo "📁 Cleaning tprimat output: $OUTPUT_DIR"
    
    # Remove checkpoint directories but keep logs
    for ckpt_dir in checkpoints checkpoint checkpoint_*; do
        if [ -d "$ckpt_dir" ]; then
            remove_dir "$ckpt_dir"
            CLEANED=$((CLEANED + 1))
        fi
    done
    
    # Note: TensorBoard and profiling artifacts are preserved
    
    echo ""
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ $CLEANED -gt 0 ]; then
    echo "✅ Cleaned $CLEANED checkpoint/artifact locations"
    echo ""
    echo "📄 Preserved:"
    echo "  - Training logs (*.log)"
    echo "  - Benchmark results (*.json)"
    echo "  - Analysis plots (*.png)"
    echo "  - TensorBoard logs"
    echo "  - Profiling data"
else
    echo "✨ No checkpoints or artifacts found - already clean!"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
