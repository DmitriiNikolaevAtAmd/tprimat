# NeMo GPU Benchmarking Suite

**Complete toolkit for profiling and comparing AMD vs NVIDIA GPU performance with NeMo**

## 🎯 What This Does

Provides a **unified, fair, and automated** way to benchmark NeMo training on AMD and NVIDIA GPUs, then compare the results with detailed analysis and visualizations.

## 🚀 Quickest Start (30 seconds)

```bash
# On NVIDIA GPU
./run_benchmark.sh llama

# On AMD GPU  
./run_benchmark.sh llama

# Compare
python3 compare_results.py
```

**Done!** You'll get charts, reports, and a clear answer on which platform is faster.

## 📚 Documentation

Choose your level of detail:

| File | Purpose | When to Use |
|------|---------|-------------|
| **[QUICK_START.md](QUICK_START.md)** | 5-minute guide with examples | Just want to run it |
| **[WORKFLOW.md](WORKFLOW.md)** | Visual diagrams and architecture | Want to understand how it works |
| **[BENCHMARK_README.md](BENCHMARK_README.md)** | Complete reference manual | Need all the details |
| **[../BENCHMARKING_GUIDE.md](../BENCHMARKING_GUIDE.md)** | High-level overview | Want the big picture |

## 🛠️ What's Included

### Core Components

```
benchmark_utils.py          Platform-agnostic benchmarking framework
compare_results.py          Generate comparison reports and charts
analyze_existing_logs.py    Analyze your existing profiling data
run_benchmark.sh            Automated benchmark runner
```

### Training Scripts (Updated)

```
pretrain_llama.py           Llama 3.1 8B + benchmarking
pretrain_qwen.py            Qwen 2.5 7B + benchmarking
pretrain_mistral.py         Mistral 7B + benchmarking
```

All scripts now include automatic profiling without changing training logic.

### Documentation

```
README.md                   This file (start here)
QUICK_START.md              Fast track guide
WORKFLOW.md                 Architecture and diagrams
BENCHMARK_README.md         Complete reference
requirements.txt            Python dependencies
```

## 📊 What You Get

### During Training

Real-time metrics displayed in terminal:

```
[CUDA] Step  10 | Time: 1.234s | Avg: 1.245s | Memory: 45.67GB
[CUDA] Step  20 | Time: 1.238s | Avg: 1.242s | Memory: 45.68GB
```

### After Training

JSON file with complete metrics:

```json
{
  "platform": "cuda",
  "performance_metrics": {
    "avg_step_time_seconds": 1.245,
    "throughput_steps_per_second": 0.803
  },
  "memory_metrics": {
    "peak_memory_allocated_gb": 45.89
  }
}
```

### After Comparison

1. **comparison_plot.png** - 4-panel visualization:
   - Average step time (bar chart)
   - Throughput (bar chart)
   - Memory usage (grouped bars)
   - Step time distribution (line plot)

2. **comparison_report.md** - Detailed analysis:
   - Executive summary with winner
   - Hardware specifications
   - Performance metrics tables
   - Statistical breakdown
   - Timestamps and metadata

3. **Console output** - Quick summary:
```
🏆 RESULT: NVIDIA is 1.26x FASTER
```

## 🎓 Key Features

✅ **Platform Agnostic** - One codebase for both CUDA and ROCm  
✅ **Fair Comparison** - Identical configurations guaranteed  
✅ **Automated** - Scripts handle everything  
✅ **Visual** - Generate charts automatically  
✅ **Detailed** - Comprehensive metrics and statistics  
✅ **Non-Invasive** - Doesn't change training logic  
✅ **Multiple Models** - Llama, Qwen, Mistral support  

## 📋 Prerequisites

```bash
# Already installed with NeMo
torch, lightning, nemo_run

# For visualization (optional but recommended)
pip install matplotlib numpy

# For TensorBoard analysis (optional)
pip install tensorboard
```

## 🎯 Usage Examples

### Basic AMD vs NVIDIA Comparison

```bash
# 1. Run on NVIDIA system
./run_benchmark.sh llama

# 2. Transfer JSON to comparison machine (or run on AMD)
# benchmark_results/benchmark_cuda_*.json

# 3. Run on AMD system
./run_benchmark.sh llama

# 4. Compare (can run on either system)
python3 compare_results.py
```

### Multiple Runs for Statistical Significance

```bash
# Run 5 times for more reliable results
./run_benchmark.sh llama 5
```

### Compare All Models

```bash
# Run all three models on each platform
for model in llama qwen mistral; do
    ./run_benchmark.sh $model
done
```

### Analyze Existing Logs

```bash
# Check what you already have
python3 analyze_existing_logs.py
```

## 🔍 Understanding Output

### Speedup Factor

```
1.0x = Same performance
1.5x = 50% faster
2.0x = 2x faster (double speed)
```

### Throughput

Higher is better. Measured in steps/second.

```
0.803 steps/s (NVIDIA) > 0.638 steps/s (AMD) = NVIDIA faster
```

### Memory

Lower is better (more efficient). Measured in GB.

```
45.89GB (NVIDIA) < 48.23GB (AMD) = NVIDIA more efficient
```

## 🐛 Common Issues

### "No GPU detected"

```bash
# Check NVIDIA
nvidia-smi

# Check AMD  
rocm-smi

# Verify PyTorch
python3 -c "import torch; print(torch.cuda.is_available())"
```

### "Cannot compare - missing results"

Need results from **both** platforms:
- `benchmark_cuda_*.json` (from NVIDIA)
- `benchmark_rocm_*.json` (from AMD)

### "Module not found: matplotlib"

```bash
pip install matplotlib numpy
```

### Out of Memory

Edit training script to reduce batch size:

```python
recipe.data.global_batch_size = 4  # Default is 8
```

## 🎨 Customization

### Change Number of Steps

Edit training scripts:

```python
recipe.trainer.max_steps = 20  # Default is 10
```

### Change Output Directory

```python
benchmark_callback = BenchmarkCallback(
    output_dir="./my_results"
)
```

### Force Platform Detection

```python
benchmark_callback = BenchmarkCallback(
    platform="cuda"  # or "rocm", default is "auto"
)
```

## 📁 Output Structure

```
benchmark_results/
├── benchmark_cuda_20260105_143022.json   # NVIDIA results
├── benchmark_rocm_20260105_154533.json   # AMD results
├── comparison_plot.png                   # Visual comparison
└── comparison_report.md                  # Detailed analysis
```

## 🔬 What's Measured

- **Step Time**: Time per training step (forward + backward + optimizer)
- **Throughput**: Steps completed per second
- **Memory**: GPU memory allocation (average and peak)
- **Stability**: Variance and consistency of timings
- **System Info**: GPU model, CUDA/ROCm version, PyTorch version

## 🤝 Integration with Existing Workflow

This system **does not replace** your existing profiling:

- **AMD logs** (`../amd-logs/`) - Keep for detailed kernel analysis
- **NVIDIA logs** (`../nvi-logs/`) - Keep for TensorBoard visualization
- **New benchmarks** (`benchmark_results/`) - For fair comparison

Use `analyze_existing_logs.py` to see everything in one place.

## 💡 Best Practices

1. ✅ Run multiple times (3-5) and average
2. ✅ Use identical configurations on both platforms
3. ✅ Close other GPU applications during benchmarking
4. ✅ Save system information and versions
5. ✅ Let the first step complete (automatic warmup)
6. ✅ Wait between runs (10s cooldown automatic)

## 🎯 Workflow Summary

```
Run on NVIDIA ──> benchmark_cuda_*.json ─┐
                                          ├──> Compare ──> Report + Charts
Run on AMD ─────> benchmark_rocm_*.json ─┘
```

## 📞 Quick Help

| Question | Answer |
|----------|--------|
| How do I run it? | `./run_benchmark.sh llama` |
| Where are results? | `benchmark_results/*.json` |
| How do I compare? | `python3 compare_results.py` |
| Which doc to read? | Start with `QUICK_START.md` |
| Need more steps? | Increase in training script |
| Different model? | Use `qwen` or `mistral` |
| Multiple runs? | `./run_benchmark.sh llama 5` |
| Out of memory? | Reduce `global_batch_size` |

## 🚀 Next Steps

1. **First time?** → Read [QUICK_START.md](QUICK_START.md)
2. **Want details?** → Read [BENCHMARK_README.md](BENCHMARK_README.md)
3. **Ready to run?** → Execute `./run_benchmark.sh llama`
4. **Have questions?** → Check [WORKFLOW.md](WORKFLOW.md)

---

**Version**: 1.0  
**Compatible with**: NeMo 24.x+, PyTorch 2.x+  
**Platforms**: NVIDIA CUDA, AMD ROCm  

**Happy Benchmarking! 🎉**

