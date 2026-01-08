# TensorPrimat

**Complete toolkit for profiling and comparing AMD vs NVIDIA GPU performance with NeMo**

## 🎯 What This Does

Provides a **unified, fair, and automated** way to benchmark NeMo training on AMD and NVIDIA GPUs, then compare the results with detailed analysis and visualizations.

## 🚀 Quickest Start (30 seconds)

### On NVIDIA (NeMo):
```bash
./run_benchmark.sh all  # Run all models
```

### On AMD (Primus):
```bash
# Run the benchmark script - it will show extraction instructions
./run_benchmark.sh all

# Or extract directly from existing logs
python3 extract_primus_metrics.py \
    --log-file training.log \
    --model-name llama \
    --num-gpus 8 \
    --global-batch-size 128 \
    --sequence-length 2048
```

### Compare Results:
```bash
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
run_benchmark.sh            Automated benchmark runner (supports 'all' models)
extract_primus_metrics.py  Extract metrics from Primus training logs
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
  "platform": "nvd",
  "gpu_info": {
    "device_name": "NVIDIA H100 80GB HBM3",
    "software_stack": "cuda",
    "software_version": "12.8",
    "pytorch_version": "2.7.0a0+7c8ec84dab.nv25.03"
  },
  "performance_metrics": {
    "avg_step_time_seconds": 1.602,
    "tokens_per_second_per_gpu": 5114.792,
    "steps_per_second": 0.624
  },
  "memory_metrics": {
    "peak_memory_allocated_gb": 22.295
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

### On NVIDIA Platform (NeMo)

```bash
# Run all models at once
./run_benchmark.sh all

# Or run individual models
./run_benchmark.sh llama
./run_benchmark.sh mistral
./run_benchmark.sh qwen

# This creates:
# - output/benchmark_cuda_llama.json
# - output/benchmark_cuda_mistral.json
# - output/benchmark_cuda_qwen.json
```

### On AMD Platform (Primus)

**Option A: Extract from existing logs**
```bash
# After your Primus training run
python3 extract_primus_metrics.py \
    --log-file primus_llama_training.log \
    --model-name llama \
    --num-gpus 8 \
    --global-batch-size 128 \
    --sequence-length 2048

# Repeat for each model
python3 extract_primus_metrics.py \
    --log-file primus_mistral_training.log \
    --model-name mistral \
    --num-gpus 8 \
    --global-batch-size 128 \
    --sequence-length 2048

# Creates:
# - output/benchmark_rocm_llama.json
# - output/benchmark_rocm_mistral.json
# - output/benchmark_rocm_qwen.json
```


### Complete AMD vs NVIDIA Workflow

```bash
# 1. Run on NVIDIA system (NeMo)
./run_benchmark.sh all

# 2. Run on AMD system (Primus) - extract from logs
python3 extract_primus_metrics.py \
    --log-file primus_training.log \
    --model-name llama \
    --num-gpus 8 \
    --global-batch-size 128 \
    --sequence-length 2048

# 3. Compare results
python3 compare_results.py

# Results will show performance comparison for all models
```

### Multiple Runs for Statistical Significance

```bash
# Run each model 5 times
./run_benchmark.sh all 5

# Or single model multiple times
./run_benchmark.sh llama 5
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

### "ModuleNotFoundError: No module named 'nemo'" (AMD/Primus)

This is **expected** - NeMo is not typically installed on AMD/Primus systems.

**Solution**: Use the Primus log extraction method instead:
```bash
python3 extract_primus_metrics.py \
    --log-file your_training.log \
    --model-name llama \
    --num-gpus 8 \
    --global-batch-size 128 \
    --sequence-length 2048
```

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

Need results from **both** platforms for the same model:
- `output/benchmark_cuda_llama.json` (from NVIDIA)
- `output/benchmark_rocm_llama.json` (from AMD)

Each model creates a single file that gets overwritten on each run.

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
    output_dir="./my_results",
    model_name="llama"  # Model name for filename
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
output/
├── benchmark_cuda_llama.json      # NVIDIA Llama results
├── benchmark_cuda_mistral.json    # NVIDIA Mistral results
├── benchmark_cuda_qwen.json       # NVIDIA Qwen results
├── benchmark_rocm_llama.json      # AMD Llama results
├── benchmark_rocm_mistral.json    # AMD Mistral results
├── benchmark_rocm_qwen.json       # AMD Qwen results
├── comparison_plot.png            # Visual comparison
└── comparison_report.md           # Detailed analysis
```

**Note**: Each model creates a **single file per platform** that gets overwritten on each run. This ensures you always have the latest results for each model.

## 🔬 What's Measured

### Performance Metrics
- **Step Time**: Time per training step (forward + backward + optimizer)
- **Tokens/sec/GPU**: Per-GPU throughput efficiency (primary metric)
- **Total Throughput**: System-wide tokens processed per second
- **Steps per Second**: Training iterations per second

### Memory Metrics
- **Average Memory**: Typical GPU memory usage during training
- **Peak Memory**: Maximum GPU memory allocated
- **Reserved Memory**: Total memory reserved by PyTorch

### System Information
- **Platform**: `nvd` (NVIDIA) or `amd` (AMD)
- **Software Stack**: `cuda` or `rocm`
- **Software Version**: CUDA/ROCm version
- **PyTorch Version**: Full PyTorch version string
- **GPU Model**: Exact device name (e.g., "NVIDIA H100 80GB HBM3")
- **GPU Cores**: Number of CUDA cores or Stream Processors

## 🤝 Integration with Existing Workflow

This system **does not replace** your existing profiling:

- **AMD logs** (`../amd-logs/`) - Keep for detailed kernel analysis
- **NVIDIA logs** (`../nvi-logs/`) - Keep for TensorBoard visualization
- **New benchmarks** (`output/`) - For fair comparison

Use `analyze_existing_logs.py` to see everything in one place.

### Working with Primus Logs

If you have Primus training logs, extract metrics with:

```bash
python3 extract_primus_metrics.py \
    --log-file primus_training.log \
    --model-name llama \
    --num-gpus 8 \
    --global-batch-size 128 \
    --sequence-length 2048
```

This creates `output/benchmark_rocm_llama.json` from existing logs.

## 💡 Best Practices

1. ✅ Run multiple times (3-5) and average
2. ✅ Use identical configurations on both platforms
3. ✅ Close other GPU applications during benchmarking
4. ✅ Save system information and versions
5. ✅ Let the first step complete (automatic warmup)
6. ✅ Wait between runs (10s cooldown automatic)

## 🎯 Workflow Summary

```
NVIDIA (NeMo):
./run_benchmark.sh all           ──> benchmark_cuda_llama.json    ─┐
                                 ──> benchmark_cuda_mistral.json  ─┤
                                 ──> benchmark_cuda_qwen.json     ─┤
                                                                   │
AMD (Primus):                                                      ├──> Compare
Primus training.log             ──┐                               │
  + extract_primus_metrics.py   ──┼──> benchmark_rocm_llama.json  ─┤
                                  └──> benchmark_rocm_mistral.json─┤
                                  └──> benchmark_rocm_qwen.json    ─┘
                                                                   │
                                                                   ▼
                                                         comparison_plot.png
                                                         comparison_report.md
```

## 📞 Quick Help

| Question | Answer |
|----------|--------|
| How on NVIDIA? | `./run_benchmark.sh all` |
| How on AMD/Primus? | `python3 extract_primus_metrics.py --log-file training.log --model-name llama ...` |
| Run all models? | NVIDIA: `./run_benchmark.sh all` |
| Where are results? | `output/benchmark_cuda_llama.json`, `output/benchmark_rocm_llama.json` |
| How do I compare? | `python3 compare_results.py` |
| No NeMo on AMD? | **Expected** - use log extraction instead |
| Which doc to read? | Start with `QUICK_START.md` |
| Different model? | Use `llama`, `mistral`, `qwen`, or `all` |
| Extract from logs? | `python3 extract_primus_metrics.py --help` |

## 🚀 Next Steps

1. **First time?** → Read [QUICK_START.md](QUICK_START.md)
2. **Want details?** → Read [BENCHMARK_README.md](BENCHMARK_README.md)
3. **Ready to run?** → Execute `./run_benchmark.sh all`
4. **Have questions?** → Check [WORKFLOW.md](WORKFLOW.md)

## 🆕 Recent Updates

- ✅ **Single file per model**: Each model creates one file (e.g., `benchmark_cuda_llama.json`)
- ✅ **Run all models**: Use `./run_benchmark.sh all` to benchmark all models at once
- ✅ **Model-based naming**: Filenames include model name for clarity
- ✅ **Improved metrics**: Added `tokens_per_second_per_gpu` as primary efficiency metric
- ✅ **Software stack info**: Track CUDA/ROCm versions explicitly
- ✅ **Output directory**: Changed from `benchmark_results` to `output`
- ✅ **Platform naming**: Use `nvd`/`amd` for clarity

---

**Version**: 2.0  
**Compatible with**: NeMo 24.x+, PyTorch 2.x+  
**Platforms**: NVIDIA CUDA, AMD ROCm  

**Happy Benchmarking! 🎉**

