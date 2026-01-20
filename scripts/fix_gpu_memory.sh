#!/bin/bash
# Fix GPU memory issues before running benchmarks

echo "╔════════════════════════════════════════════════════════════╗"
echo "║              GPU Memory Cleanup & Check                   ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Function to check GPU memory
check_gpu_memory() {
    echo "  * Current GPU Memory Status:"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=index,name,memory.used,memory.free,memory.total --format=csv,noheader,nounits | \
        while IFS=, read -r idx name used free total; do
            used_pct=$(awk "BEGIN {printf \"%.1f\", ($used/$total)*100}")
            echo "GPU $idx: $name"
            echo "  Used: ${used} MiB / ${total} MiB ($used_pct%)"
            echo "  Free: ${free} MiB"
            
            if (( $(echo "$used > 1000" | bc -l) )); then
                echo "  [!] WARNING: GPU has significant memory in use!"
            fi
            echo ""
        done
    else
        echo "  x nvidia-smi not found. Are you on a CUDA system?"
        exit 1
    fi
}

# Function to find and kill lingering Python/training processes
kill_lingering_processes() {
    echo ""
    echo "  * Checking for lingering training processes..."
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # Find Python processes using GPU
    PROCESSES=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null)
    
    if [ -z "$PROCESSES" ]; then
        echo "  + No GPU processes found"
    else
        echo "Found GPU processes:"
        echo "$PROCESSES"
        echo ""
        
        read -p "Kill these processes? (y/N): " -n 1 -r
        echo
        
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            nvidia-smi --query-compute-apps=pid --format=csv,noheader | while read pid; do
                if [ -n "$pid" ]; then
                    echo "Killing process $pid..."
                    kill -9 $pid 2>/dev/null || sudo kill -9 $pid 2>/dev/null
                fi
            done
            
            sleep 2
            echo "  + Processes killed"
        else
            echo "Skipping process cleanup"
        fi
    fi
}

# Function to clear PyTorch cache
clear_pytorch_cache() {
    echo ""
    echo "  * Clearing PyTorch cache..."
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    python3 << 'EOF'
import torch
if torch.cuda.is_available():
    print(f"CUDA available: {torch.cuda.device_count()} GPUs")
    torch.cuda.empty_cache()
    print("  + PyTorch cache cleared")
    
    # Print memory stats for each GPU
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1e9
        reserved = torch.cuda.memory_reserved(i) / 1e9
        print(f"GPU {i}: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB")
else:
    print("  x CUDA not available")
EOF
}

# Main execution
echo "Step 1: Check current GPU memory"
check_gpu_memory

echo ""
echo "Step 2: Find and kill lingering processes"
kill_lingering_processes

echo ""
echo "Step 3: Clear PyTorch cache"
clear_pytorch_cache

echo ""
echo "Step 4: Final memory check"
check_gpu_memory

echo "════════════════════════════════════════════════════════════"
echo "  * Recommendations:"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "If memory issues persist:"
echo ""
echo "1. Set PyTorch memory allocator:"
echo "   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
echo ""
echo "2. Reduce batch size in pretrain scripts:"
echo "   recipe.data.global_batch_size = 64  # Instead of 128"
echo ""
echo "3. Check if other users are using GPUs:"
echo "   nvidia-smi"
echo ""
echo "4. Reboot the system (if you have access):"
echo "   sudo reboot"
echo ""
echo "5. Use fewer GPUs:"
echo "   # In pretrain script, change num_gpus_per_node=4 instead of 8"
echo ""
