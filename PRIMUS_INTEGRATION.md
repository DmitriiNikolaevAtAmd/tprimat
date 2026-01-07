# Primus Benchmark Integration Guide

## Overview

This guide shows how to integrate the AMD vs NVIDIA benchmarking system with your **Primus** training setup.

## Your Current Setup

```bash
EXP=examples/megatron/configs/MI300X/llama3.1_8B-pretrain.yaml \
bash ./examples/run_pretrain.sh \
    --train_iters 10 \
    --fp8 hybrid \
    --micro_batch_size 1 \
    --global_batch_size 128
```

## Integration Methods

### Method 1: Modify YAML Config (Recommended)

Edit your config file: `examples/megatron/configs/MI300X/llama3.1_8B-pretrain.yaml`

Add the benchmark callback:

```yaml
trainer:
  # ... existing trainer config ...
  callbacks:
    - class_path: benchmark_utils.BenchmarkCallback
      init_args:
        output_dir: ./benchmark_results
        platform: auto  # Auto-detects ROCm
```

Or if Primus uses a different callback format:

```yaml
trainer:
  callbacks:
    - type: custom
      module: benchmark_utils
      class: BenchmarkCallback
      args:
        output_dir: ./benchmark_results
        platform: auto
```

### Method 2: Wrapper Script

Create a wrapper script `run_primus_benchmark.sh`:

```bash
#!/bin/bash

# Location of benchmark utilities
export PYTHONPATH="/workspace/support/week-02/code:$PYTHONPATH"

# Your Primus command
EXP=examples/megatron/configs/MI300X/llama3.1_8B-pretrain.yaml \
bash ./examples/run_pretrain.sh \
    --train_iters 10 \
    --fp8 hybrid \
    --micro_batch_size 1 \
    --global_batch_size 128 \
    --benchmark_output ./benchmark_results

# After training completes, show results
echo ""
echo "========================================"
echo "Benchmark Results"
echo "========================================"
python3 -c "
import json, glob
files = glob.glob('./benchmark_results/benchmark_rocm_*.json')
if files:
    latest = max(files)
    with open(latest) as f:
        data = json.load(f)
    print(f\"Platform: {data['platform'].upper()}\")
    print(f\"Device: {data['gpu_info']['device_name']}\")
    print(f\"GPUs: {data['gpu_info']['device_count']}\")
    if data['performance_metrics'].get('tokens_per_second_per_gpu'):
        print(f\"Throughput: {data['performance_metrics']['tokens_per_second']:,.0f} tokens/sec\")
        print(f\"Per-GPU: {data['performance_metrics']['tokens_per_second_per_gpu']:,.0f} tokens/sec/GPU\")
    print(f\"File: {latest}\")
"
```

Make it executable:
```bash
chmod +x run_primus_benchmark.sh
```

### Method 3: Modify Primus Run Script

If you can modify `examples/run_pretrain.sh`, add near the trainer initialization:

```python
# Add benchmark callback
import sys
sys.path.insert(0, '/workspace/support/week-02/code')
from benchmark_utils import BenchmarkCallback

benchmark_callback = BenchmarkCallback(
    output_dir="./benchmark_results",
    platform="auto"
)

# Add to trainer callbacks
if not hasattr(trainer, 'callbacks'):
    trainer.callbacks = []
trainer.callbacks.append(benchmark_callback)
```

### Method 4: Post-Processing (If Callbacks Don't Work)

If you can't modify the training run, collect data afterward:

```bash
# After training completes, extract metrics from Primus logs
python3 extract_primus_metrics.py \
    --log-file primus_output.log \
    --output benchmark_results/benchmark_rocm_manual.json \
    --num-gpus 8 \
    --global-batch-size 128 \
    --sequence-length 2048
```

Create `extract_primus_metrics.py`:

```python
#!/usr/bin/env python3
"""Extract metrics from Primus training logs."""
import argparse
import json
import re
from datetime import datetime
from pathlib import Path

def extract_metrics_from_log(log_file, num_gpus, global_batch_size, seq_length):
    """Extract timing and performance metrics from Primus log."""
    step_times = []
    
    with open(log_file, 'r') as f:
        for line in f:
            # Look for step timing patterns
            # Adjust regex based on your Primus log format
            match = re.search(r'step=(\d+).*time=([0-9.]+)', line)
            if match:
                step_times.append(float(match.group(2)))
    
    if not step_times:
        print("⚠️  No timing data found in log file")
        return None
    
    # Calculate metrics (skip first step as warmup)
    step_times_no_warmup = step_times[1:] if len(step_times) > 1 else step_times
    avg_step_time = sum(step_times_no_warmup) / len(step_times_no_warmup)
    
    # Calculate token-based throughput
    tokens_per_step = global_batch_size * seq_length
    tokens_per_second = tokens_per_step / avg_step_time
    tokens_per_second_per_gpu = tokens_per_second / num_gpus
    
    results = {
        "platform": "rocm",
        "gpu_info": {
            "device_count": num_gpus,
            "device_name": "AMD GPU",  # Update with actual from log if available
        },
        "timestamp": datetime.now().isoformat(),
        "training_config": {
            "global_batch_size": global_batch_size,
            "sequence_length": seq_length,
            "num_gpus": num_gpus,
        },
        "performance_metrics": {
            "total_steps": len(step_times),
            "avg_step_time_seconds": avg_step_time,
            "tokens_per_second": tokens_per_second,
            "tokens_per_second_per_gpu": tokens_per_second_per_gpu,
        },
        "raw_step_times": step_times,
        "source": "primus_log_extraction"
    }
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Extract metrics from Primus logs')
    parser.add_argument('--log-file', required=True, help='Primus log file')
    parser.add_argument('--output', required=True, help='Output JSON file')
    parser.add_argument('--num-gpus', type=int, required=True, help='Number of GPUs')
    parser.add_argument('--global-batch-size', type=int, required=True)
    parser.add_argument('--sequence-length', type=int, default=2048)
    
    args = parser.parse_args()
    
    results = extract_metrics_from_log(
        args.log_file,
        args.num_gpus,
        args.global_batch_size,
        args.sequence_length
    )
    
    if results:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✅ Metrics saved to: {args.output}")
        print(f"Tokens/sec/GPU: {results['performance_metrics']['tokens_per_second_per_gpu']:,.0f}")
    else:
        print("❌ Failed to extract metrics")

if __name__ == "__main__":
    main()
```

## Configuration Parameters

Your Primus command shows:
- `--train_iters 10` → 10 training steps
- `--fp8 hybrid` → FP8 hybrid precision
- `--micro_batch_size 1` → Micro batch size per GPU
- `--global_batch_size 128` → Total batch size across all GPUs

The benchmark system needs:
- **num_gpus**: Number of GPUs (auto-detected from environment)
- **global_batch_size**: 128 (from your command)
- **sequence_length**: Usually 2048 (check your YAML config)

## Expected Output

After running with benchmarking enabled:

```
============================================================
BENCHMARK COMPLETE - Platform: ROCM
============================================================
GPUs: 8
Total Steps: 10
Total Time: 12.45s
Avg Step Time: 1.245s

Throughput Metrics:
  Total Throughput: 104,857 tokens/sec
  Per-GPU Throughput: 13,107 tokens/sec/GPU
  (Global batch size: 128, Sequence length: 2048)

Memory Usage:
  Avg Memory: 48.23GB
  Peak Memory: 51.67GB

Results saved to: ./benchmark_results/benchmark_rocm_20260105_143022.json
============================================================
```

## File Locations

```
/workspace/support/week-02/code/
├── benchmark_utils.py          ← Core benchmarking framework
├── primus_benchmark.py         ← Primus integration helper
├── extract_primus_metrics.py   ← Log extraction tool
└── benchmark_results/          ← Output directory
    └── benchmark_rocm_*.json   ← Your results
```

## Comparing with NVIDIA Results

After running on both AMD (Primus) and NVIDIA:

```bash
cd /workspace/support/week-02/code

# Ensure you have both:
# benchmark_results/benchmark_rocm_*.json  (AMD/Primus)
# benchmark_results/benchmark_cuda_*.json  (NVIDIA)

python3 compare_results.py
```

Output:
```
============================================================
AMD vs NVIDIA GPU COMPARISON
============================================================

NVIDIA (NVIDIA H100 SXM5 80GB):
  GPUs:            8
  Avg Step Time:   1.245s
  Throughput:      110,104 tokens/sec (total)
  Tokens/sec/GPU:  13,763
  Peak Memory:     45.89GB

AMD (AMD Instinct MI300X):
  GPUs:            8
  Avg Step Time:   1.156s
  Throughput:      113,852 tokens/sec (total)
  Tokens/sec/GPU:  14,231
  Peak Memory:     51.67GB

Result:
  AMD is 1.08x faster (by time)
  Tokens/sec/GPU ratio (NVIDIA/AMD): 0.97x
============================================================
```

## Troubleshooting

### Benchmark Callback Not Loading

```bash
# Check Python path
python3 -c "import sys; print('\n'.join(sys.path))"

# Manually add to PYTHONPATH
export PYTHONPATH="/workspace/support/week-02/code:$PYTHONPATH"

# Verify import works
python3 -c "from benchmark_utils import BenchmarkCallback; print('✅ Import successful')"
```

### Results Not Saving

Check permissions:
```bash
mkdir -p ./benchmark_results
chmod 777 ./benchmark_results
```

### Wrong Global Batch Size Detected

Manually specify in your integration:
```python
# After creating benchmark_callback
benchmark_callback.global_batch_size = 128
benchmark_callback.sequence_length = 2048
```

### Missing Sequence Length

Check your YAML config for `seq_length` or `sequence_length`:
```bash
grep -i "seq" examples/megatron/configs/MI300X/llama3.1_8B-pretrain.yaml
```

## Best Practices

1. **Same Configuration**: Use identical settings for AMD and NVIDIA:
   - Same global_batch_size (128)
   - Same train_iters (10)
   - Same fp8 setting (hybrid)
   - Same sequence length (2048)

2. **Multiple Runs**: Run 3-5 times on each platform for statistical significance

3. **Document Setup**: Save your Primus command and YAML config with results

4. **Check Logs**: Verify training completed successfully before comparing

## Quick Start

Simplest way to get started:

```bash
# 1. On AMD/ROCm system with Primus
cd /path/to/primus
export PYTHONPATH="/workspace/support/week-02/code:$PYTHONPATH"

# 2. Run your training (add logging)
EXP=examples/megatron/configs/MI300X/llama3.1_8B-pretrain.yaml \
bash ./examples/run_pretrain.sh \
    --train_iters 10 \
    --fp8 hybrid \
    --micro_batch_size 1 \
    --global_batch_size 128 \
    2>&1 | tee primus_training.log

# 3. Extract metrics from log
cd /workspace/support/week-02/code
python3 extract_primus_metrics.py \
    --log-file /path/to/primus/primus_training.log \
    --output benchmark_results/benchmark_rocm_manual.json \
    --num-gpus 8 \
    --global-batch-size 128 \
    --sequence-length 2048

# 4. Compare with NVIDIA results
python3 compare_results.py
```

## Summary

| Method | Complexity | Reliability | Recommended For |
|--------|-----------|-------------|-----------------|
| YAML Config | Low | High | If you control configs |
| Wrapper Script | Medium | High | Quick integration |
| Modify Run Script | High | High | Full control |
| Post-Processing | Low | Medium | Can't modify training |

**Recommended**: Start with **Post-Processing** (Method 4) to quickly get results, then move to **YAML Config** (Method 1) for automated future runs.

---

**Next Steps**:
1. Choose integration method
2. Run training on AMD (Primus)
3. Extract/collect metrics
4. Run on NVIDIA
5. Compare with `compare_results.py`

