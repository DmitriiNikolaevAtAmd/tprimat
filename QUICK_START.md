# Quick Start: AMD vs NVIDIA GPU Comparison

## TL;DR

```bash
# On NVIDIA system
./run_benchmark.sh llama

# On AMD system  
./run_benchmark.sh llama

# Compare results
python3 compare_results.py
```

## Step-by-Step

### 1️⃣ Install Dependencies

```bash
pip install matplotlib numpy
```

### 2️⃣ Run on NVIDIA GPU

```bash
# Option A: Use the automated script
./run_benchmark.sh llama

# Option B: Run directly
python3 pretrain_llama.py
```

**Expected output:**
```
============================================================
BENCHMARK START - Platform: CUDA
============================================================
device_count: 8
device_name: NVIDIA A100-SXM4-80GB
...

[CUDA] Step  10 | Time: 1.245s | Avg: 1.245s | Memory: 45.67GB
...

============================================================
BENCHMARK COMPLETE - Platform: CUDA
============================================================
Avg Step Time: 1.245s
Throughput: 0.803 steps/s
Results saved to: benchmark_results/benchmark_cuda_20260105_143022.json
```

### 3️⃣ Run on AMD GPU

Same commands on your AMD system:

```bash
./run_benchmark.sh llama
# or
python3 pretrain_llama.py
```

**Expected output:**
```
============================================================
BENCHMARK START - Platform: ROCM
============================================================
device_count: 8
device_name: AMD Instinct MI250X
...
```

### 4️⃣ Compare Results

```bash
python3 compare_results.py
```

**Expected output:**
```
📊 Comparing:
  NVIDIA: NVIDIA A100-SXM4-80GB (2026-01-05T14:30:22)
  AMD:    AMD Instinct MI250X (2026-01-05T15:45:33)

✅ Comparison plot saved to: comparison_plot.png
✅ Comparison report saved to: comparison_report.md

============================================================
🏆 RESULT: NVIDIA is 1.26x FASTER
============================================================
```

### 5️⃣ View Results

**View plot:**
- Open `comparison_plot.png`

**Read report:**
- Open `comparison_report.md`

**Raw data:**
- `benchmark_results/benchmark_cuda_*.json`
- `benchmark_results/benchmark_rocm_*.json`

## Available Models

```bash
# Llama 3.1 8B (default)
./run_benchmark.sh llama

# Qwen 2.5 7B
./run_benchmark.sh qwen

# Mistral 7B
./run_benchmark.sh mistral
```

## Multiple Runs for Accuracy

```bash
# Run 5 times for statistical significance
./run_benchmark.sh llama 5
```

## Troubleshooting

### ❌ "No GPU detected"
- Check CUDA: `nvidia-smi`
- Check ROCm: `rocm-smi`
- Verify PyTorch: `python3 -c "import torch; print(torch.cuda.is_available())"`

### ❌ "No NVIDIA/AMD results found"
- Make sure you ran the training on **both** platforms
- Check `benchmark_results/` directory: `ls benchmark_results/`

### ❌ Import errors
```bash
pip install matplotlib numpy
```

### ❌ Out of memory
Edit the training script and reduce batch size:
```python
recipe.data.micro_batch_size = 1  # Already at minimum
recipe.data.global_batch_size = 4  # Reduce from 8 to 4
```

## What Gets Compared?

### Primary Metric (Most Important!)
🌟 **Tokens/sec/GPU** - Per-GPU efficiency (Chart #1)
- Shows how many tokens each GPU processes per second
- Directly comparable across architectures
- Predicts performance at any scale
- Example: 13,425 tokens/sec/GPU × 8 GPUs = 107,400 tokens/sec total

### Supporting Metrics
✅ **Step Time** - How long each training step takes  
✅ **Total Throughput** - Total system tokens/sec  
✅ **Memory Usage** - GPU memory consumption (average & peak)  
✅ **GPU Count** - System configuration  
✅ **Efficiency Summary** - Comprehensive comparison panel

**Why Tokens/sec/GPU matters:** This tells you which GPU is more efficient at processing your workload, independent of how many GPUs you have. It's the key metric for choosing hardware!  

## Output Files

```
week-02/code/
├── benchmark_results/
│   ├── benchmark_cuda_20260105_143022.json    # NVIDIA results
│   ├── benchmark_rocm_20260105_154533.json    # AMD results
│   ├── comparison_plot.png                     # Visual comparison
│   └── comparison_report.md                    # Detailed report
```

## Tips

💡 **Same configuration**: All training scripts use identical settings  
💡 **Warmup excluded**: First step is automatically skipped  
💡 **Multiple runs**: Run 3-5 times and average for best accuracy  
💡 **Idle system**: Close other GPU applications during benchmarking  

## Example Results

```
📊 Chart 1: Tokens/sec/GPU - Per-GPU Efficiency (PRIMARY)
   NVIDIA H100:     5,115 tokens/sec/GPU
   AMD MI300X:     13,425 tokens/sec/GPU
   → AMD is 2.62x more efficient per GPU ⭐

Additional Metrics:
   Step Time:       NVIDIA 1.602s | AMD 9.763s
   Total Throughput: NVIDIA 40,918 tokens/sec | AMD 107,401 tokens/sec
   Memory:          NVIDIA 22.3 GB | AMD 118.0 GB
   GPU Count:       8 GPUs each

Result:
   AMD MI300X is 2.62x more efficient per GPU for this workload
   At 512-GPU scale: AMD would process 4.2M more tokens/sec
```

---

## Additional Documentation

### Understanding Metrics
- **[TOKENS_PER_GPU_METRIC.md](TOKENS_PER_GPU_METRIC.md)** - ⭐ PRIMARY METRIC explained in detail
- **[THROUGHPUT_METRICS.md](THROUGHPUT_METRICS.md)** - All throughput calculations

### Setup Guides
- **[BENCHMARK_README.md](BENCHMARK_README.md)** - Complete benchmarking guide
- **[PRIMUS_INTEGRATION.md](PRIMUS_INTEGRATION.md)** - Primus-specific setup

### Advanced Topics
- **[PER_CORE_METRICS.md](PER_CORE_METRICS.md)** - Per-core analysis
- **[LOG_STORAGE_GUIDE.md](LOG_STORAGE_GUIDE.md)** - Log organization
- **[WHAT_GETS_SAVED.md](WHAT_GETS_SAVED.md)** - Output reference

