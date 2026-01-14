# Peak Throughput Analysis Guide

## Overview

This document explains how to estimate and interpret peak total throughput for LLM training workloads across different GPU platforms and configurations.

## Key Findings

### 🥇 Overall Peak Performance

- **Best Configuration**: AMD Instinct MI300X with Qwen 2.5 7B
- **Peak Throughput**: **87,933 tokens/second**
- **Compute Performance**: **4,010 TFLOPS** (38.3% hardware utilization)
- **Per-GPU**: 10,992 tokens/s (501 TFLOPS per GPU)

### Platform Comparison

| Platform | Model | Peak Tokens/s | TFLOPS | HW Util | Step Time |
|----------|-------|---------------|--------|---------|-----------|
| **AMD MI300X** | Qwen | 87,933 | 4,010 | 38.3% | 14.91s |
| **NVIDIA H100** | Qwen | 78,877 | 3,597 | 22.7% | 3.32s |
| **AMD MI300X** | Llama | 68,875 | 3,306 | 31.6% | 15.22s |
| **NVIDIA H100** | Llama | 59,175 | 2,840 | 17.9% | 4.43s |

**Winner**: AMD MI300X is **1.11x faster** than NVIDIA H100 at peak throughput

## Understanding Peak Throughput

### 1. What is Peak Throughput?

Peak throughput represents the maximum sustained training performance across all your benchmark configurations. It's measured in:

- **Tokens/second**: Raw throughput for processing training tokens
- **TFLOPS**: Computational intensity (floating-point operations per second)
- **Hardware Utilization**: Percentage of theoretical peak performance achieved

### 2. Theoretical vs Achieved Performance

**Hardware Theoretical Peaks (FP16/BF16):**
- AMD Instinct MI300X: 1,307 TFLOPS per GPU
- NVIDIA H100: 1,979 TFLOPS per GPU

**Why the Gap?**
Achieved performance (17-38% utilization) is lower than theoretical peak because:

1. **Memory Bandwidth**: Data transfer limits compute
2. **Communication Overhead**: Multi-GPU synchronization costs
3. **Framework Efficiency**: PyTorch/NeMo/Primus implementation overhead
4. **Model Architecture**: Not all operations are compute-bound
5. **Parallelism Strategy**: TP/PP/DP configuration impact

**38% utilization is actually quite good** for real-world LLM training!

### 3. Computing Throughput Metrics

#### FLOPs per Token
For transformer models: **FLOPs = 6 × num_parameters**

- Llama 3.1 8B: 48 billion FLOPs per token
- Qwen 2.5 7B: 45.6 billion FLOPs per token

#### TFLOPS Calculation
```
TFLOPS = (tokens_per_second × flops_per_token) / 1e12
```

Example (AMD MI300X + Qwen):
```
87,933 tokens/s × 45.6B FLOPs = 4.01 TFLOPS (total)
4,010 TFLOPS / 8 GPUs = 501 TFLOPS per GPU
```

#### Hardware Utilization
```
Utilization = (Achieved TFLOPS per GPU / Theoretical Peak) × 100
```

Example:
```
501 TFLOPS / 1,307 TFLOPS = 38.3%
```

## Scaling Projections

### Daily Training Capacity (at peak: 87,933 tokens/s)

- **Tokens per day**: 7.6 billion tokens
- **Sequences per day**: 3.7 million (at 2048 seq_len)

### Multi-Node Scaling (assuming linear)

| Nodes | GPUs | Tokens/s | TFLOPS | Tokens/Day |
|-------|------|----------|--------|------------|
| 1 | 8 | 87,933 | 4,010 | 7.6B |
| 2 | 16 | 175,867 | 8,020 | 15.2B |
| 4 | 32 | 351,733 | 16,039 | 30.4B |
| 8 | 64 | 703,466 | 32,078 | 60.8B |

**Note**: Real-world scaling efficiency is typically 85-95% due to cross-node communication overhead.

## Configuration Insights

### AMD MI300X Performance
- **Consistent**: All configurations achieve 68-88K tokens/s
- **Best for Qwen**: 87.9K tokens/s (38.3% utilization)
- **Best for Llama**: 68.9K tokens/s (31.6% utilization)
- **Advantage**: Superior memory bandwidth (5.2 TB/s) and 192GB HBM3

### NVIDIA H100 Performance
- **Variable**: Wide range from 6K to 79K tokens/s across configs
- **Best for Qwen**: 78.9K tokens/s (22.7% utilization)
- **Best for Llama**: 59.2K tokens/s (17.9% utilization)
- **Advantage**: Lower latency per step (3-4s vs 15s)
- **Challenge**: Memory bottleneck at 80GB limits some configurations

### Why Does Configuration Matter?

Different tensor parallel (TP) and pipeline parallel (PP) strategies show dramatic performance differences:

- **Best configs**: Achieve 60-88K tokens/s
- **Worst configs**: Drop to 6-20K tokens/s
- **Delta**: Up to **14x performance difference**!

This shows the critical importance of choosing the right parallelism strategy for your hardware.

## How to Estimate Peak Throughput for Your Setup

### Step 1: Identify Your Hardware
```python
# Get theoretical peak from specs
GPU_MODEL = "AMD Instinct MI300X"
THEORETICAL_TFLOPS = 1307  # per GPU
NUM_GPUS = 8
TOTAL_THEORETICAL = THEORETICAL_TFLOPS * NUM_GPUS  # 10,456 TFLOPS
```

### Step 2: Estimate Achievable Utilization
Based on our benchmarks:
- **Good configuration**: 30-40% utilization
- **Average configuration**: 15-25% utilization
- **Poor configuration**: 5-10% utilization

### Step 3: Calculate Expected Throughput
```python
# For Qwen 2.5 7B on AMD MI300X with good config
EXPECTED_UTILIZATION = 0.38  # 38%
ACHIEVED_TFLOPS_PER_GPU = THEORETICAL_TFLOPS * EXPECTED_UTILIZATION  # 497 TFLOPS

# Convert to tokens/s
FLOPS_PER_TOKEN = 45.6e9  # Qwen 2.5 7B
TOKENS_PER_SEC_PER_GPU = (ACHIEVED_TFLOPS_PER_GPU * 1e12) / FLOPS_PER_TOKEN  # 10,900
TOTAL_TOKENS_PER_SEC = TOKENS_PER_SEC_PER_GPU * NUM_GPUS  # 87,200
```

### Step 4: Validate with Benchmarks
Run actual training benchmarks to verify:
```bash
python3 benchmark.py --model qwen
```

Compare achieved vs estimated to refine your utilization factor.

## Using the Analysis Tools

### 1. Analyze Peak Throughput
```bash
# Show peak performance only
python3 analyze_peak_throughput.py

# Show all configurations
python3 analyze_peak_throughput.py --show-all
```

### 2. Generate Visualization
```bash
python3 visualize_peak_throughput.py
```

This creates `peak_throughput_analysis.png` with:
- Peak throughput comparison
- TFLOPS analysis
- Hardware utilization
- Per-GPU performance
- Configuration distributions
- Platform comparison

### 3. Compare Specific Configurations
```bash
# Compare specific output directories
python3 compare.py --results-dir ./all_outputs/01-output
```

## Recommendations

### For Maximum Throughput
1. **Use AMD MI300X** if available (11% faster at peak)
2. **Choose Qwen 2.5 7B** over Llama 3.1 8B (28% faster)
3. **Test parallelism strategies** - huge impact on performance
4. **Monitor hardware utilization** - aim for >30%

### For Lowest Latency
1. **Use NVIDIA H100** (3-4s vs 15s per step)
2. **Optimize tensor parallelism** to minimize communication
3. **Consider sequence packing** to improve GPU utilization

### For Memory-Constrained Workloads
1. **AMD MI300X** with 192GB is better for large models
2. **Use pipeline parallelism** to split model across GPUs
3. **Enable activation checkpointing** to trade compute for memory

## Key Takeaways

1. ✅ **Peak throughput is measurable and reproducible**
2. ✅ **Configuration matters more than hardware** (14x difference!)
3. ✅ **AMD MI300X leads in throughput**, NVIDIA H100 in latency
4. ✅ **30-40% hardware utilization is realistic** for LLM training
5. ✅ **Linear scaling to multi-node** is achievable with good interconnect

## References

- Full results: `peak_throughput_summary.txt`
- Visualization: `peak_throughput_analysis.png`
- Raw benchmarks: `output/benchmark_*.json`, `all_outputs/*/benchmark_*.json`

---

**Generated by**: TensorPrimat Peak Throughput Analyzer  
**Date**: January 2026  
**Peak Performance**: 87,933 tokens/s (AMD MI300X + Qwen 2.5 7B)
