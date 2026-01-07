# Throughput Metrics Explained

## Overview

The benchmarking system measures performance using **token-based throughput metrics**, which are the standard in LLM training and inference benchmarking.

## Key Metrics

### 1. **Throughput** (Total System Throughput)

**Definition**: How much work the system (or benchmark setup) completes per unit time.

**Units**: `tokens/sec` (tokens per second)

**Formula**:
```
Throughput = (global_batch_size × sequence_length) / avg_step_time
```

**What it means**:
- Measures the **total system capacity**
- Shows how many tokens the entire GPU cluster processes per second
- Indicates the system's ability to process requests with maximum load saturation
- Higher throughput = more efficient GPU utilization and higher concurrency handling

**Example**:
```
8 GPUs training Llama 3.1 8B
Global batch size: 8
Sequence length: 2048
Step time: 1.245s

Throughput = (8 × 2048) / 1.245 = 13,157 tokens/sec (total system)
```

### 2. **Tokens/sec/GPU** (Per-GPU Throughput)

**Definition**: A specific measurement of throughput **per GPU** - the efficiency metric independent of cluster size.

**Units**: `tokens/sec/GPU` (tokens per second per GPU)

**Formula**:
```
Tokens/sec/GPU = Total Throughput / Number of GPUs
```

**What it means**:
- Measures **per-GPU efficiency**
- Evaluates the efficiency of the software stack and hardware's raw power
- **Independent of total cluster size** - allows fair comparison across different setups
- Used to evaluate scaling efficiency and hardware utilization
- Engineers use this to compare different GPU types fairly

**Example**:
```
Total Throughput: 13,157 tokens/sec
Number of GPUs: 8

Tokens/sec/GPU = 13,157 / 8 = 1,645 tokens/sec/GPU
```

**In multi-node training**:
```
1-node (8-GPU) system showing 13,763 Tokens/sec/GPU
→ Total system throughput = 13,763 × 8 = 110,104 tokens/sec

100-node (800-GPU) cluster with same per-GPU efficiency
→ Total system throughput = 13,763 × 800 = 11,010,400 tokens/sec
```

## Why This Matters

### Fair Comparison Across Scales

Per-GPU throughput lets you compare:
- **Different GPU types**: H100 vs MI250X vs A100
- **Different cluster sizes**: 8-GPU node vs 1000-GPU cluster
- **Different configurations**: Various parallelism strategies

### Example Comparison

| System | GPUs | Total Throughput | Tokens/sec/GPU | Winner |
|--------|------|-----------------|----------------|---------|
| H100 (8-node) | 64 | 1,052,032 tokens/sec | 16,438 | ✅ More efficient |
| MI250X (8-node) | 64 | 819,200 tokens/sec | 12,800 | Lower efficiency |

Even though both have 64 GPUs, H100 is **1.28x more efficient per GPU**.

### Scaling Analysis

Per-GPU metrics reveal scaling efficiency:

```
8 GPUs:  14,000 tokens/sec/GPU  → 100% efficiency (baseline)
64 GPUs: 13,500 tokens/sec/GPU  → 96.4% efficiency (good scaling)
512 GPUs: 12,000 tokens/sec/GPU → 85.7% efficiency (communication overhead visible)
```

## Benchmark Output Format

### During Training

```
============================================================
BENCHMARK COMPLETE - Platform: CUDA
============================================================
GPUs: 8
Total Steps: 10
Total Time: 12.45s
Avg Step Time: 1.245s

Throughput Metrics:
  Total Throughput: 13,157 tokens/sec
  Per-GPU Throughput: 1,645 tokens/sec/GPU
  (Global batch size: 8, Sequence length: 2048)

Memory Usage:
  Avg Memory: 45.67GB
  Peak Memory: 45.89GB
============================================================
```

### Comparison Output

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

AMD (AMD Instinct MI250X):
  GPUs:            8
  Avg Step Time:   1.567s
  Throughput:      82,636 tokens/sec (total)
  Tokens/sec/GPU:  10,329
  Peak Memory:     48.23GB

Result:
  NVIDIA is 1.26x faster (by time)
  Tokens/sec/GPU ratio (NVIDIA/AMD): 1.33x
============================================================
```

## Calculation Details

### What Goes Into the Calculation

1. **Global Batch Size**: Total number of sequences processed per step across all GPUs
2. **Sequence Length**: Number of tokens in each sequence (e.g., 2048, 4096, 8192)
3. **Step Time**: Time to complete one training step (forward + backward + optimizer)

### Example Calculation Walkthrough

```python
# Training Configuration
global_batch_size = 8      # 8 sequences per step
sequence_length = 2048     # 2048 tokens per sequence
num_gpus = 8              # 8 GPUs in the system
avg_step_time = 1.245     # 1.245 seconds per step

# Step 1: Calculate tokens per step
tokens_per_step = global_batch_size × sequence_length
tokens_per_step = 8 × 2048 = 16,384 tokens

# Step 2: Calculate total system throughput
total_throughput = tokens_per_step / avg_step_time
total_throughput = 16,384 / 1.245 = 13,157 tokens/sec

# Step 3: Calculate per-GPU throughput
tokens_per_sec_per_gpu = total_throughput / num_gpus
tokens_per_sec_per_gpu = 13,157 / 8 = 1,645 tokens/sec/GPU
```

## Common Benchmarks

### Typical Values (LLaMA 3.1 8B, FP8, 2048 seq length)

| GPU | Tokens/sec/GPU | Notes |
|-----|----------------|-------|
| H100 80GB | 13,000-16,000 | Best performance |
| MI250X | 10,000-12,000 | Good performance |
| A100 80GB | 8,000-11,000 | Previous generation |
| A100 40GB | 7,000-10,000 | Memory constrained |

*Values vary based on parallelism strategy, batch size, and model configuration*

## Relationship to Other Metrics

### Steps per Second (Legacy)

```
Steps/sec = 1 / avg_step_time
```

Not recommended for comparison because:
- ❌ Doesn't account for batch size
- ❌ Doesn't account for sequence length
- ❌ Can't compare across different configurations

### Samples per Second

```
Samples/sec = global_batch_size / avg_step_time
```

Better than steps/sec but:
- ❌ Still doesn't account for sequence length
- ❌ Can't compare across different sequence lengths

### Tokens per Second ✅ (Recommended)

```
Tokens/sec = (global_batch_size × sequence_length) / avg_step_time
```

Best metric because:
- ✅ Accounts for both batch size and sequence length
- ✅ Enables fair comparison across all configurations
- ✅ Industry standard for LLM benchmarking

## Interpreting Results

### High Tokens/sec/GPU = Good

Indicates:
- ✅ Efficient hardware utilization
- ✅ Well-optimized software stack
- ✅ Good parallelism configuration
- ✅ Minimal communication overhead

### Low Tokens/sec/GPU = Investigate

Possible causes:
- ❌ Poor parallelism configuration
- ❌ Communication bottlenecks
- ❌ Memory bandwidth limitations
- ❌ Software inefficiencies
- ❌ Suboptimal batch size

## Best Practices

1. **Always report both metrics**:
   - Total throughput (for system capacity)
   - Per-GPU throughput (for efficiency comparison)

2. **Document configuration**:
   - Global batch size
   - Sequence length
   - Number of GPUs
   - Parallelism strategy (TP, PP, DP)

3. **Use consistent settings for comparison**:
   - Same model
   - Same batch size
   - Same sequence length
   - Same precision (FP8, FP16, etc.)

4. **Consider scaling efficiency**:
   - Compare per-GPU metrics across cluster sizes
   - Look for communication overhead
   - Evaluate cost-effectiveness

## Technical Notes

### Why Tokens?

In LLM training/inference:
- **Tokens** are the atomic unit of work
- Processing a 2048-token sequence is 2x the work of 1024-token
- Fair comparison requires accounting for actual compute

### Multi-GPU Scaling

Perfect scaling:
```
8 GPUs:  14,000 tokens/sec/GPU → Total: 112,000 tokens/sec
16 GPUs: 14,000 tokens/sec/GPU → Total: 224,000 tokens/sec (2x)
```

Reality (with communication overhead):
```
8 GPUs:  14,000 tokens/sec/GPU → Total: 112,000 tokens/sec
16 GPUs: 13,500 tokens/sec/GPU → Total: 216,000 tokens/sec (1.93x) - 96.4% efficiency
```

## Summary

| Metric | Unit | Purpose | When to Use |
|--------|------|---------|-------------|
| **Throughput** | tokens/sec | Total system capacity | Evaluating overall performance |
| **Tokens/sec/GPU** | tokens/sec/GPU | Per-GPU efficiency | Comparing hardware, scaling analysis |
| Steps/sec | steps/sec | Step rate | Internal monitoring only |
| Samples/sec | samples/sec | Sample rate | Batch efficiency analysis |

**Primary Metric**: **Tokens/sec/GPU** for fair hardware comparison and efficiency evaluation.

**Secondary Metric**: **Throughput (tokens/sec)** for total system capacity and time-to-solution.

---

**Related Documentation**:
- `PER_CORE_METRICS.md` - GPU core-level efficiency metrics
- `BENCHMARK_README.md` - Complete benchmarking reference
- `QUICK_START.md` - Getting started guide

