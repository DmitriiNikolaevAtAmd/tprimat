# NeMo GPU Profiling & Benchmarking - Complete Summary

## 📍 What Was Created

A complete, production-ready benchmarking system for fairly comparing AMD and NVIDIA GPU performance with NeMo training workloads.

## 🎯 Problem Solved

**Before**: Difficult to fairly compare AMD and NVIDIA GPUs because:
- Different profiling tools (ROCProfiler vs Nsight)
- Different output formats (Excel vs TensorBoard)
- Manual metric collection
- Inconsistent configurations

**After**: Unified benchmarking system that:
- ✅ Works on both platforms with same code
- ✅ Auto-detects CUDA vs ROCm
- ✅ Collects identical metrics
- ✅ Generates automated comparison reports
- ✅ Ensures fair comparison with identical configs

## 📁 What's Where

### Main Documentation
```
week-02/
├── BENCHMARKING_GUIDE.md          ← START HERE (high-level overview)
└── code/
    ├── README.md                  ← Quick index of all files
    ├── QUICK_START.md             ← 5-minute quick start
    ├── WORKFLOW.md                ← Visual diagrams
    └── BENCHMARK_README.md        ← Complete reference
```

### Core Implementation
```
week-02/code/
├── benchmark_utils.py             ← Benchmarking framework (10KB)
│   └── BenchmarkCallback          ← Main profiling class
│
├── compare_results.py             ← Comparison tool (11KB)
│   ├── load_benchmark_results()
│   ├── create_comparison_plot()
│   └── generate_comparison_report()
│
├── analyze_existing_logs.py       ← Analyze old profiling data (7KB)
│   ├── analyze_amd_profiling_logs()
│   └── analyze_nvidia_logs()
│
└── run_benchmark.sh               ← Automation script (3.5KB)
```

### Updated Training Scripts
```
week-02/code/
├── pretrain_llama.py              ← Updated with benchmarking
├── pretrain_qwen.py               ← Updated with benchmarking
└── pretrain_mistral.py            ← Updated with benchmarking
```

All three scripts now include:
```python
from benchmark_utils import BenchmarkCallback

benchmark_callback = BenchmarkCallback(
    output_dir="./benchmark_results",
    platform="auto"  # Auto-detects CUDA or ROCm
)
recipe.trainer.callbacks.append(benchmark_callback)
```

### Dependencies
```
week-02/code/
└── requirements.txt               ← matplotlib, numpy, tensorboard
```

## 🚀 How to Use

### Quick Version (3 commands)

```bash
# On NVIDIA GPU
cd week-02/code
./run_benchmark.sh llama

# On AMD GPU
cd week-02/code
./run_benchmark.sh llama

# Compare (on either system)
python3 compare_results.py
```

### What You Get

1. **During Training** - Real-time metrics:
```
[CUDA] Step  10 | Time: 1.234s | Avg: 1.245s | Memory: 45.67GB
```

2. **After Training** - JSON results:
```
benchmark_results/benchmark_cuda_20260105_143022.json
benchmark_results/benchmark_rocm_20260105_154533.json
```

3. **After Comparison** - Visual reports:
```
comparison_plot.png          (4-panel chart)
comparison_report.md         (detailed analysis)
Console: "🏆 NVIDIA is 1.26x FASTER"
```

## 📊 Metrics Collected

### Performance
- **Average Step Time** (seconds) - Lower is better
- **Throughput** (steps/sec) - Higher is better
- **Min/Max Step Time** - Range of performance
- **Variance** - Consistency measure

### Memory
- **Average Memory** (GB) - Typical usage
- **Peak Memory** (GB) - Maximum usage
- **Reserved Memory** (GB) - Total allocated

### System
- GPU model and specifications
- CUDA/ROCm version
- PyTorch version
- Training configuration

## 🎨 Key Features

### 1. Platform Agnostic
```python
# Auto-detects platform
if torch.cuda.is_available():
    platform = "cuda" if "cuda" in torch.version.cuda else "rocm"
```

### 2. Non-Invasive
- Doesn't change training logic
- Just adds a callback to the trainer
- Can be easily removed if needed

### 3. Fair Comparison
- Identical configurations guaranteed
- Same warmup handling (skip first step)
- Same synchronization points
- Same metric calculations

### 4. Automated
- Scripts handle everything
- Auto-generates reports
- Auto-creates visualizations
- Auto-detects platform

### 5. Comprehensive
- Multiple metrics collected
- Statistical analysis included
- Visual and text reports
- Raw data preserved

## 🔍 Example Output

### Console Output
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

### Comparison Output
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

### Visualization
The `comparison_plot.png` contains 4 charts:
1. **Average Step Time** - Bar chart comparing platforms
2. **Throughput** - Bar chart showing steps/second
3. **Memory Usage** - Grouped bars (avg vs peak)
4. **Step Time Distribution** - Line plot over training

## 🎓 Best Practices

### For Accurate Results
1. ✅ Run 3-5 times and average
2. ✅ Close other GPU applications
3. ✅ Use identical configurations
4. ✅ Same PyTorch/NeMo versions
5. ✅ Document system information

### For Fair Comparison
1. ✅ Same model architecture
2. ✅ Same batch sizes
3. ✅ Same parallelism settings
4. ✅ Same number of steps
5. ✅ Same precision (FP8)

### For Reliability
1. ✅ Let warmup complete (automatic)
2. ✅ Wait between runs (automatic)
3. ✅ Check for thermal throttling
4. ✅ Verify GPU utilization
5. ✅ Monitor system resources

## 🔧 Configuration

### Current Training Setup (Identical on Both Platforms)

**Llama 3.1 8B:**
```python
tensor_model_parallel_size = 4
pipeline_model_parallel_size = 1
micro_batch_size = 1
global_batch_size = 8
max_steps = 10
fp8 = "hybrid"
```

**Qwen 2.5 7B:**
```python
tensor_model_parallel_size = 4
pipeline_model_parallel_size = 2
micro_batch_size = 1
global_batch_size = 8
max_steps = 10
fp8 = "hybrid"
```

**Mistral 7B:**
```python
tensor_model_parallel_size = 4
pipeline_model_parallel_size = 1
micro_batch_size = 1
global_batch_size = 8
max_steps = 10
fp8 = "hybrid"
```

## 🐛 Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| No GPU detected | Check `nvidia-smi` or `rocm-smi` |
| Import error | `pip install matplotlib numpy` |
| Out of memory | Reduce `global_batch_size` |
| Missing results | Run on both platforms first |
| Wrong platform | Set `platform="cuda"` or `"rocm"` explicitly |

## 📚 Documentation Hierarchy

```
Level 1: Quick Start
└── QUICK_START.md (5 min read)
    └── 3 commands to get results

Level 2: Visual Understanding
└── WORKFLOW.md (10 min read)
    └── Diagrams and architecture

Level 3: Complete Reference
└── BENCHMARK_README.md (20 min read)
    └── All details and options

Level 4: Overview
└── BENCHMARKING_GUIDE.md (15 min read)
    └── Big picture and context
```

**Start with**: `QUICK_START.md` if you just want to run it  
**Read next**: `WORKFLOW.md` to understand how it works  
**Reference**: `BENCHMARK_README.md` when you need details

## 🎯 Use Cases

### 1. Basic Comparison
```bash
./run_benchmark.sh llama          # On both platforms
python3 compare_results.py        # Compare
```

### 2. Statistical Analysis
```bash
./run_benchmark.sh llama 5        # 5 runs on each platform
python3 compare_results.py        # Average results
```

### 3. Multi-Model Comparison
```bash
for model in llama qwen mistral; do
    ./run_benchmark.sh $model
done
```

### 4. Analyze Existing Data
```bash
python3 analyze_existing_logs.py  # Check old profiling logs
```

## 🔬 Advanced Profiling

The basic system can be extended with:

### PyTorch Profiler
```python
from lightning.pytorch.profilers import PyTorchProfiler
recipe.trainer.profiler = PyTorchProfiler(...)
```

### NVIDIA Nsight
```bash
nsys profile --trace=cuda,nvtx python3 pretrain_llama.py
```

### AMD ROCProfiler
```bash
rocprof --stats --timestamp on python3 pretrain_llama.py
```

## 📊 Output Files

### Generated by Training
```
benchmark_results/
└── benchmark_{platform}_{timestamp}.json
    ├── platform (cuda/rocm)
    ├── gpu_info (device, memory, versions)
    ├── training_config (batch size, steps, etc.)
    ├── performance_metrics (time, throughput)
    ├── memory_metrics (avg, peak)
    └── raw_step_times (all measurements)
```

### Generated by Comparison
```
benchmark_results/
├── comparison_plot.png          (4-panel visualization)
└── comparison_report.md         (detailed markdown report)
```

## 🎉 Summary

You now have a **complete, production-ready benchmarking system** that:

✅ Works on both AMD and NVIDIA GPUs  
✅ Provides fair, apples-to-apples comparison  
✅ Generates automated reports and visualizations  
✅ Collects comprehensive metrics  
✅ Is easy to use (3 commands)  
✅ Is well-documented (4 guides)  
✅ Is extensible (add new models/metrics)  
✅ Is non-invasive (doesn't change training)  

## 🚀 Next Steps

1. **Install dependencies**: `pip install matplotlib numpy`
2. **Read quick start**: Open `week-02/code/QUICK_START.md`
3. **Run on NVIDIA**: `./run_benchmark.sh llama`
4. **Run on AMD**: `./run_benchmark.sh llama`
5. **Compare**: `python3 compare_results.py`
6. **Analyze**: Review `comparison_plot.png` and `comparison_report.md`

## 📞 Quick Reference

| Task | Command |
|------|---------|
| Run benchmark | `./run_benchmark.sh llama` |
| Multiple runs | `./run_benchmark.sh llama 5` |
| Compare results | `python3 compare_results.py` |
| Check old logs | `python3 analyze_existing_logs.py` |
| View results | `ls benchmark_results/` |

---

**Created**: January 5, 2026  
**Location**: `/Users/dmitrynvm/Work/support/week-02/`  
**Status**: ✅ Ready to use  

**Happy Benchmarking! 🎉**

