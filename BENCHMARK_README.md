# AMD vs NVIDIA GPU Benchmarking Guide

This guide explains how to fairly compare AMD and NVIDIA GPU performance using a unified benchmarking framework with NeMo.

## Overview

The benchmarking system provides:
- ✅ Platform-agnostic profiling (works on both CUDA and ROCm)
- ✅ Automated metric collection (step time, memory, throughput)
- ✅ Fair comparison with identical configurations
- ✅ Visual comparison reports

## Quick Start

### 1. Run Training on NVIDIA GPU

```bash
# On your NVIDIA system
python pretrain_llama.py
```

This will:
- Auto-detect CUDA platform
- Collect performance metrics
- Save results to `benchmark_results/benchmark_cuda_TIMESTAMP.json`

### 2. Run Training on AMD GPU

```bash
# On your AMD system  
python pretrain_llama.py
```

This will:
- Auto-detect ROCm platform
- Collect performance metrics
- Save results to `benchmark_results/benchmark_rocm_TIMESTAMP.json`

### 3. Compare Results

```bash
# After running on both platforms
python compare_results.py
```

This generates:
- `comparison_plot.png` - Visual comparison charts
- `comparison_report.md` - Detailed comparison report

## Configuration

### Ensuring Fair Comparison

The training scripts are already configured identically:

```python
# Llama 3.1 8B - Same on both AMD and NVIDIA
recipe.trainer.strategy.tensor_model_parallel_size = 4
recipe.trainer.strategy.pipeline_model_parallel_size = 1
recipe.data.micro_batch_size = 1
recipe.data.global_batch_size = 128  # Matches Primus config
recipe.data.seq_length = 2048  # Matches Primus config
recipe.trainer.max_steps = 10
recipe.model.config.fp8 = "hybrid"
recipe.model.config.fp8_param = True
```

### Key Requirements for Fair Comparison

1. **Same Model**: Use the same model architecture (e.g., Llama 3.1 8B)
2. **Same Configuration**: Keep all hyperparameters identical
3. **Same Data**: Use the same batch sizes and data
4. **Same Steps**: Run for the same number of steps
5. **Same Optimizations**: Use the same FP8 settings

## What Gets Measured

### Performance Metrics

- **Average Step Time**: Time per training step (lower is better)
- **Throughput**: Steps per second (higher is better)
- **Min/Max Step Time**: Best and worst step times
- **Variance**: Consistency of performance

### Memory Metrics

- **Average Memory**: Typical GPU memory usage
- **Peak Memory**: Maximum GPU memory usage
- **Reserved Memory**: Total memory allocated by PyTorch

### System Information

- GPU model and memory
- CUDA/ROCm version
- PyTorch version
- Training configuration

## Understanding Results

### Benchmark Output

During training, you'll see real-time metrics:

```
[CUDA] Step  10 | Time: 1.234s | Avg: 1.245s | Memory: 45.67GB
[CUDA] Step  20 | Time: 1.238s | Avg: 1.242s | Memory: 45.68GB
```

### Final Summary

At the end, you'll see:

```
============================================================
BENCHMARK COMPLETE - Platform: CUDA
============================================================
Total Steps: 10
Total Time: 12.45s
Avg Step Time: 1.245s
Throughput: 0.803 steps/s
Avg Memory: 45.67GB
Peak Memory: 45.89GB

Results saved to: benchmark_results/benchmark_cuda_20260105_143022.json
============================================================
```

### Comparison Report

The comparison script outputs:

```
============================================================
AMD vs NVIDIA GPU COMPARISON
============================================================

NVIDIA GPU (NVIDIA A100-SXM4-80GB):
  Avg Step Time: 1.245s
  Throughput:    0.803 steps/s
  Peak Memory:   45.89GB

AMD GPU (AMD Instinct MI250X):
  Avg Step Time: 1.567s
  Throughput:    0.638 steps/s
  Peak Memory:   48.23GB

Result:
  NVIDIA is 1.26x faster
  Throughput ratio (NVIDIA/AMD): 1.26x
============================================================
```

## Advanced Usage

### Custom Benchmark Configuration

You can customize the benchmark callback:

```python
from benchmark_utils import BenchmarkCallback

benchmark_callback = BenchmarkCallback(
    output_dir="./my_benchmark_results",
    platform="auto"  # or "cuda" / "rocm" to force
)
```

### Multiple Runs for Statistical Significance

Run multiple times and average the results:

```bash
# Run 5 times on each platform
for i in {1..5}; do
    echo "Run $i"
    python pretrain_llama.py
    sleep 10  # Cool down
done
```

### Comparing Different Models

To compare the same GPU across different models:

```bash
# Run all models on NVIDIA
python pretrain_llama.py    # Llama 3.1 8B
python pretrain_qwen.py     # Qwen 2.5 7B
python pretrain_mistral.py  # Mistral 7B

# Run all models on AMD
python pretrain_llama.py
python pretrain_qwen.py
python pretrain_mistral.py
```

## Profiling Options

### Basic Profiling (Included)

The benchmark callback provides:
- Step timing
- Memory usage
- Throughput metrics

### Advanced Profiling (Optional)

#### PyTorch Profiler

For detailed kernel-level profiling:

```python
from lightning.pytorch.profilers import PyTorchProfiler

recipe.trainer.profiler = PyTorchProfiler(
    dirpath="./profile_logs",
    filename="detailed_profile",
    export_to_chrome=True,
    profile_memory=True,
    with_stack=True,
)
```

View results:
- Open `chrome://tracing` in Chrome
- Load the generated JSON file

#### NVIDIA Nsight Systems

For NVIDIA-specific profiling:

```bash
nsys profile \
    --output=llama_profile \
    --trace=cuda,nvtx,osrt,cudnn,cublas \
    python pretrain_llama.py
```

#### AMD ROCProfiler

For AMD-specific profiling:

```bash
rocprof --stats --timestamp on \
    python pretrain_llama.py
```

## Troubleshooting

### No Results Generated

Check that:
- Training completed successfully
- `benchmark_results/` directory exists
- JSON files were created

### Platform Not Detected

If auto-detection fails, manually specify:

```python
benchmark_callback = BenchmarkCallback(
    platform="cuda"  # or "rocm"
)
```

### Comparison Script Fails

Ensure you have both CUDA and ROCm results:

```bash
ls benchmark_results/
# Should show both:
# benchmark_cuda_*.json
# benchmark_rocm_*.json
```

### Memory Errors

If you run out of memory:
- Reduce `micro_batch_size`
- Reduce `global_batch_size`
- Increase tensor parallelism
- Enable gradient checkpointing

## Best Practices

1. **Warm Up**: The first step is automatically excluded (warmup)
2. **Multiple Runs**: Run 3-5 times and average for reliability
3. **Idle System**: Close other applications during benchmarking
4. **Same Environment**: Use same PyTorch/NeMo versions on both platforms
5. **Document Everything**: Save system info, versions, and configurations

## File Structure

```
week-02/code/
├── benchmark_utils.py          # Core benchmarking framework
├── compare_results.py          # Comparison and visualization
├── pretrain_llama.py          # Llama training with benchmarking
├── pretrain_qwen.py           # Qwen training with benchmarking
├── pretrain_mistral.py        # Mistral training with benchmarking
├── BENCHMARK_README.md        # This file
└── benchmark_results/         # Generated results
    ├── benchmark_cuda_*.json
    ├── benchmark_rocm_*.json
    ├── comparison_plot.png
    └── comparison_report.md
```

## Metrics Reference

### Step Time
- **Definition**: Time to complete one training step
- **Includes**: Forward pass, backward pass, optimizer step
- **Excludes**: Data loading (handled asynchronously)
- **Lower is better**

### Throughput
- **Definition**: Number of steps completed per second
- **Formula**: `steps / total_time`
- **Higher is better**
- **Key metric for training speed**

### Memory Usage
- **Allocated**: Actually used by tensors
- **Reserved**: Total memory held by PyTorch
- **Peak**: Maximum at any point during training

### Speedup
- **Definition**: How much faster one platform is
- **Formula**: `slower_time / faster_time`
- **Example**: 1.5x means 50% faster

## Support

For issues or questions:
1. Check training logs in `benchmark_results/`
2. Verify configurations match between platforms
3. Ensure both platforms use compatible PyTorch/NeMo versions

---

**Happy Benchmarking! 🚀**

