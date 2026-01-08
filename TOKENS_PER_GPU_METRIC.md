# Tokens/sec/GPU - The Primary Efficiency Metric

## Overview

**Tokens/sec/GPU** is the **most important metric** for evaluating GPU efficiency in LLM training and inference. It measures the number of tokens processed by a **single GPU** in one second.

## Why This Metric Matters

### 1. Hardware-Independent Comparison
- Normalizes performance across different GPU counts
- Directly compares the efficiency of different GPU architectures
- Independent of cluster size or total system configuration

### 2. Scalability Prediction
When you know the per-GPU throughput, you can predict total system performance:

```
Total System Throughput = Tokens/sec/GPU × Number of GPUs
```

**Example from your benchmark:**
- AMD MI300X: `13,425 tokens/sec/GPU × 8 GPUs = 107,400 tokens/sec`
- NVIDIA H100: `5,115 tokens/sec/GPU × 8 GPUs = 40,918 tokens/sec`

### 3. Cost-Efficiency Analysis
When combined with GPU pricing, this metric helps determine:
- Performance per dollar
- ROI for hardware investments
- Optimal GPU selection for workloads

## Chart Position

In the comparison plots, **Tokens/sec/GPU** is displayed as:
- **Chart #1** (top-left, primary position)
- Larger font size and emphasis
- Includes efficiency ratio annotation
- Color-coded bars (NVIDIA green, AMD red)

## Interpretation

### High Tokens/sec/GPU means:
✅ Efficient utilization of GPU cores  
✅ Optimized memory bandwidth usage  
✅ Better software stack performance  
✅ Lower cost per token processed  

### When comparing two systems:
```
Efficiency Ratio = Higher tokens/sec/GPU ÷ Lower tokens/sec/GPU
```

**From your data:**
```
AMD efficiency = 13,425 ÷ 5,115 = 2.62x
```
This means AMD MI300X processes 2.62× more tokens per GPU compared to NVIDIA H100 for this specific workload.

## Related Metrics

| Metric | Description | Use Case |
|--------|-------------|----------|
| **Tokens/sec/GPU** | Per-GPU efficiency | Primary metric, architecture comparison |
| **Total Tokens/sec** | System throughput | Overall performance, cluster evaluation |
| **Avg Step Time** | Time per iteration | Debugging, latency analysis |
| **Memory Usage** | GPU memory consumption | Resource planning, batch size tuning |

## Training Configuration Impact

Tokens/sec/GPU is affected by:

1. **Batch Size**: Larger batches → better GPU utilization → higher tokens/sec
2. **Sequence Length**: Longer sequences → more computation per step
3. **Model Size**: Larger models → different memory/compute tradeoffs
4. **Precision**: FP8 vs FP16 vs FP32
5. **Parallelism Strategy**: Tensor/Pipeline/Data parallelism

## Best Practices

### When Benchmarking:
1. ✅ Run with identical batch sizes across platforms
2. ✅ Use same sequence lengths
3. ✅ Match precision (FP8, FP16, etc.)
4. ✅ Test with production-like workloads
5. ✅ Exclude warmup steps from measurements

### When Reporting:
1. Always include training configuration
2. Report both per-GPU and total system throughput
3. Include hardware specifications (GPU model, memory)
4. Specify software versions (PyTorch, CUDA, ROCm)

## Example Output

```
📊 Chart 1: Tokens/sec/GPU - Per-GPU Efficiency (PRIMARY)
   NVIDIA:      5,115 tokens/sec/GPU
   AMD:        13,425 tokens/sec/GPU
   → AMD is 2.62x more efficient per GPU ⭐
```

## Accessing the Metric

### From JSON output:
```json
{
  "performance_metrics": {
    "tokens_per_second_per_gpu": 13425.09
  }
}
```

### From Python:
```python
from compare_results import load_benchmark_results

cuda_data, rocm_data = load_benchmark_results('./output')
amd_efficiency = rocm_data['performance_metrics']['tokens_per_second_per_gpu']
nvd_efficiency = cuda_data['performance_metrics']['tokens_per_second_per_gpu']

print(f"AMD: {amd_efficiency:,.0f} tokens/sec/GPU")
print(f"NVIDIA: {nvd_efficiency:,.0f} tokens/sec/GPU")
```

## Real-World Implications

### For Your Current Results:

| GPU | Tokens/sec/GPU | 8-GPU System | 64-GPU Cluster | 512-GPU Datacenter |
|-----|----------------|--------------|----------------|---------------------|
| AMD MI300X | 13,425 | 107,400 | 859,200 | 6,873,600 |
| NVIDIA H100 | 5,115 | 40,918 | 327,360 | 2,618,880 |

**Insight:** At datacenter scale (512 GPUs), AMD would process **4.2 million more tokens per second** than NVIDIA for this workload.

## Summary

✨ **Tokens/sec/GPU is your north star metric** for GPU efficiency evaluation.

It tells you:
- ✅ How efficiently each GPU processes work
- ✅ How performance scales with more GPUs
- ✅ Which hardware gives best value for your workload
- ✅ Real-world performance in production systems

Always optimize for this metric when tuning training configurations!

