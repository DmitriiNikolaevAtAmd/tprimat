# TensorPrimat

**Unified benchmarking toolkit for comparing AMD vs NVIDIA GPU performance with LLM training**

---

## Table of Contents

- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Usage](#-usage)
  - [Basic Commands](#basic-commands)
  - [Primus Training Scripts](#primus-training-scripts)
  - [Complete Workflow](#complete-workflow)
- [What Gets Measured](#-what-gets-measured)
- [Comparison Methodology](#-comparison-methodology)
- [Primus Training Guide](#-primus-training-guide)
- [Troubleshooting](#-troubleshooting)
- [Enhanced Metrics](#-enhanced-metrics)
- [ROCm Compatibility](#-rocm-compatibility)
- [Core Components](#-core-components)
- [Advanced Usage](#-advanced-usage)
- [Best Practices](#-best-practices)
- [Reference](#-reference)

---

## 🚀 Quick Start

```bash
# Run on any platform (NVIDIA or AMD)
./benchmark.py

# Compare results after running on both platforms
python3 compare_results.py
```

**That's it!** Automatically detects your platform and runs all models.

### What You Get

- ✅ **Automatic Platform Detection**: NVIDIA (CUDA) or AMD (ROCm)
- ✅ **Benchmark Results**: JSON files for each model on each platform
- ✅ **Comparison Reports**: Visual charts and detailed analysis
- ✅ **Smart Log Discovery**: Finds Primus logs automatically on AMD

**Benchmark Files:**
- `output/benchmark_cuda_llama.json`
- `output/benchmark_cuda_qwen.json`
- `output/benchmark_rocm_llama.json`
- `output/benchmark_rocm_qwen.json`

**Comparison Output:**
- `comparison_plot.png` - Visual charts
- `comparison_report.md` - Detailed analysis

---

## 💻 Installation

### Prerequisites

**Already installed with NeMo/Primus:**
- Python 3.8+
- PyTorch 2.x+
- CUDA or ROCm

**For visualization (optional):**
```bash
pip install matplotlib numpy
```

Or use the included requirements:
```bash
pip install -r requirements.txt
```

### Platform Support

- **NVIDIA (CUDA)**: Runs NeMo training scripts directly
- **AMD (ROCm)**: Extracts metrics from Primus training logs
- **No GPU**: Analyzes logs without GPU (perfect for remote analysis)

---

## 🎯 Usage

### Basic Commands

```bash
# Run all models (default)
./benchmark.py

# Run specific model
./benchmark.py --model llama
./benchmark.py --model qwen

# Run multiple times for statistics
./benchmark.py --runs 3

# Get help
./benchmark.py --help
```

### Primus Training Scripts

**Easy-to-use scripts for AMD/Primus training:**

```bash
# Run single model
./run_primus_llama.sh     # Llama 3.1 8B
./run_primus_qwen.sh      # Qwen 2.5 7B

# Run all models
./run_primus_all.sh
```

**Features:**
- ✅ Automatic log capture
- ✅ Automatic metric extraction
- ✅ Generates benchmark JSON files
- ✅ Shows next steps

**Configuration:**
```bash
# Custom Primus path (default: /workspace/Primus)
export PRIMUS_PATH=/custom/path/to/Primus

# Custom training iterations (default: 10)
export TRAIN_ITERS=50

# Then run
./run_primus_llama.sh
```

### Complete Workflow

```bash
# 1. Run on NVIDIA system
./benchmark.py
# → Creates: output/benchmark_cuda_*.json

# 2. Run on AMD system  
./benchmark.py  # or ./run_primus_all.sh
# → Creates: output/benchmark_rocm_*.json

# 3. Compare results (on either system)
python3 compare_results.py
# → Creates: comparison_plot.png, comparison_report.md

# 4. View enhanced metrics
python3 compare_with_enhanced_metrics.py
```

---

## 📊 What Gets Measured

### Performance Metrics
- **Tokens/sec/GPU**: Per-GPU efficiency (primary metric)
- **Total Throughput**: System-wide tokens per second
- **Step Time**: Time per training iteration (avg, min, max, std dev)
- **Steps per Second**: Training speed

### Memory Metrics
- **Average Memory**: Typical GPU memory usage
- **Peak Memory**: Maximum allocation
- **Reserved Memory**: Total PyTorch reservation

### System Information
- **Platform**: `nvd` (NVIDIA) or `amd` (AMD)
- **Software Stack**: `cuda` or `rocm`
- **Software Version**: CUDA/ROCm version string
- **PyTorch Version**: Full version
- **GPU Model**: Device name and specs
- **GPU Cores**: CUDA cores or Stream Processors

---

## 📚 Comparison Methodology

TensorPrimat uses **"Maximum Performance"** comparison methodology by default.

### Two Comparison Approaches

#### 1. Maximum Performance (Default)
Each platform is configured for **optimal performance** on that specific hardware.

**Why this approach?**
- ✅ Answers: *"What's the best real-world performance each platform can deliver?"*
- ✅ Real-world deployments optimize for each platform's strengths
- ✅ Cloud providers tune separately for AMD vs NVIDIA instances
- ✅ Represents actual production usage

**Example Results:**
| Metric | NVIDIA H100 | AMD MI300X | Advantage |
|--------|-------------|------------|-----------|
| Tokens/s/GPU | 1,380 | 13,363 | 6.34x AMD |
| Memory/GPU | 22 GB | 118 GB | 5.3x AMD |
| Configuration | TP=4, FP8 | TP=1, BF16 | Different |

**Valid for:** Production deployment decisions, cost analysis, real-world planning

#### 2. Identical Configuration (Optional)
Both platforms use the same parallelism strategy and settings.

**When to use:**
- ❌ Academic hardware studies
- ❌ Isolating pure hardware differences
- ❌ Framework maturity comparisons

**How to run:**
```bash
# Create identical configs (both use TP=4, BF16)
cd /workspace/Primus/examples/megatron/configs/MI300X/
cp llama3.1_8B-pretrain.yaml llama3.1_8B-pretrain-tp4.yaml
# Edit to match NVIDIA: TP=4, same precision, same batch sizes

# Run with fair config
FAIR_CONFIG=1 ./run_amd_dual_comparison.sh
```

### Configuration Checker

```bash
# Verify your Primus configuration
./check_primus_config.sh

# Run both comparisons
./run_amd_dual_comparison.sh
```

See your comparison results and understand what they mean:

**Maximum Performance (6.34x AMD advantage):**
- Real-world optimal configurations
- Different TP strategies (TP=4 vs TP=1)
- Each platform at its best

**Identical Configuration (~2-3x AMD advantage expected):**
- Same TP=4 on both
- Isolates hardware differences
- More apples-to-apples

---

## 🔧 Primus Training Guide

### Quick Start with Primus

```bash
# Run single model
./run_primus_llama.sh

# What happens automatically:
# 1. ✅ Validates Primus installation
# 2. ✅ Checks config file exists
# 3. ✅ Creates output directory
# 4. ✅ Runs Primus training
# 5. ✅ Captures logs (two copies)
# 6. ✅ Extracts metrics automatically
# 7. ✅ Generates benchmark JSON
# 8. ✅ Shows next steps
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PRIMUS_PATH` | `/workspace/Primus` | Primus installation directory |
| `TRAIN_ITERS` | `10` | Number of training iterations |

### Output Files

**Log Files:**
```
output/training_llama.log                    # Primary log (overwritten)
output/primus_training_llama_<timestamp>.log # Backup (timestamped)
```

**Benchmark Files:**
```
output/benchmark_rocm_llama.json
output/benchmark_rocm_qwen.json
```

### Manual Primus Training

If you need to run Primus manually:

```bash
cd /workspace/Primus

# Run Llama
export EXP=examples/megatron/configs/MI300X/llama3.1_8B-pretrain.yaml
bash ./examples/run_pretrain.sh --train_iters 10 2>&1 | tee /workspace/tprimat/training_llama.log

# Run Qwen
export EXP=examples/megatron/configs/MI300X/qwen2.5_7B-pretrain.yaml
bash ./examples/run_pretrain.sh --train_iters 10 2>&1 | tee /workspace/tprimat/training_qwen.log

# Extract metrics
cd /workspace/tprimat
./benchmark.py  # Auto-detects logs
```

### Customizing Primus Config

```bash
cd /workspace/Primus/examples/megatron/configs/MI300X/
vi llama3.1_8B-pretrain.yaml
```

**Common parameters:**
```yaml
# Parallelism
tensor_model_parallel_size: 1    # Model parallelism
pipeline_model_parallel_size: 1

# Batch sizes
micro_batch_size: 1
global_batch_size: 128

# Precision
precision: bf16  # or fp16, fp8

# Sequence length
seq_length: 2048
```

---

## 🔧 Troubleshooting

### CUDA Out of Memory Error

**Quick Fix:**
```bash
./fix_gpu_memory.sh
```

This will:
- Show current GPU memory usage
- Kill lingering Python processes
- Clear PyTorch cache
- Verify memory is freed

**Manual Solutions:**

```bash
# Kill processes
nvidia-smi | grep python | awk '{print $5}' | xargs -r kill -9

# Clear cache
python3 -c "import torch; torch.cuda.empty_cache(); print('Cache cleared')"

# Set memory allocator (already in pretrain scripts)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Reduce memory usage in scripts
recipe.data.global_batch_size = 64  # Instead of 128
recipe.trainer.strategy.tensor_model_parallel_size = 8  # Instead of 4
```

### No Logs Found Error

```bash
# ⚠️ No log file found for llama
```

**Solutions:**

```bash
# 1. Provide log paths explicitly
LLAMA_LOG=/path/to/llama.log ./benchmark.py

# 2. Copy logs to current directory
cp /path/to/logs/*.log .
./benchmark.py

# 3. Name logs correctly
# Script looks for: training_llama.log, training_qwen.log
```

### NeMo Not Found (AMD Systems)

This is **expected** on AMD systems! The benchmark automatically:
1. Detects no NeMo available
2. Switches to log extraction mode
3. Searches for Primus training logs

Just run Primus training first, then:
```bash
./benchmark.py  # Will auto-extract from logs
```

### Platform Detection Issues

If you see `❌ No AMD benchmark results found!` but you have ROCm files:

**Solution:** Already fixed! Make sure your JSON files have `"software_stack": "rocm"` or `"cuda"`.

Re-run: `python3 compare_results.py`

### Performance Issues

**Checklist:**

```bash
# 1. Check GPU utilization (should be ~100%)
watch -n 1 nvidia-smi

# 2. Check MFU (Model FLOPs Utilization)
python3 compare_with_enhanced_metrics.py
# Good: 30-50%, Excellent: 50-65%

# 3. Memory efficiency
# If memory < 50%, try larger batch size
# If memory > 95%, reduce batch size

# 4. Check for CPU bottleneck
htop  # Should not be at 100%
```

### Best Practices for Benchmarking

**Before:**
```bash
# 1. Clean GPU memory
./fix_gpu_memory.sh

# 2. Check system is idle
nvidia-smi
htop

# 3. Set environment variables
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 4. Run benchmark
./benchmark.py
```

**During:**
- ✅ Don't run other GPU processes
- ✅ Monitor with `nvidia-smi`
- ✅ Let it run without interruption

**After:**
- ✅ Verify JSON files in `output/`
- ✅ Check metrics make sense
- ✅ Run comparison scripts

---

## 📈 Enhanced Metrics

Beyond basic performance metrics, TensorPrimat can calculate:

### 1. Cost-Normalized Metrics 💰

```python
# Tokens per Dollar-Hour
tokens_per_dollar_hour = (tokens_per_second * 3600) / cost_per_hour

# Cost to Train 1 Trillion Tokens
cost_per_trillion_tokens = (1e12 / tokens_per_second) * (cost_per_hour / 3600)
```

**Cloud Pricing (approximate):**
- NVIDIA H100 (8 GPUs): ~$32/hr
- AMD MI300X (8 GPUs): ~$24/hr

### 2. Model FLOPs Utilization (MFU) 📊

Industry-standard metric for training efficiency:

```python
# Peak theoretical FLOPs
peak_flops_h100 = 989e12  # 989 TFLOPs (FP8)
peak_flops_mi300x = 653e12  # 653 TFLOPs (FP16)

# Model FLOPs per token (Llama 8B)
model_flops_per_token = 6 * num_parameters  # 48e9 for Llama 8B

# MFU
mfu = achieved_flops / (peak_flops * num_gpus)
```

**Typical values:**
- Good: 30-40% MFU
- Excellent: 40-55% MFU
- State-of-art: 55-65% MFU

### 3. Memory Efficiency 💾

```python
# Memory utilization percentage
memory_utilization = (memory_used / total_memory) * 100

# Potential batch size
potential_batch_size = (total_memory * 0.9) / (memory_per_token * seq_length)
```

### 4. Power Efficiency ⚡

```python
# Tokens per watt-hour
tokens_per_watt_hour = tokens_per_second * 3600 / (tdp * num_gpus)

# TDP values
# H100: 700W, MI300X: 750W
```

### 5. Training Time Estimates 🎯

```python
# Time to train 1 Trillion tokens
time_to_1T_tokens_hours = (1e12 / tokens_per_second) / 3600

# Full Llama 3.1 8B training (≈ 15T tokens)
time_to_full_training_days = (15e12 / tokens_per_second) / (3600 * 24)
```

### Generate Enhanced Report

```bash
python3 compare_with_enhanced_metrics.py
```

**Output includes:**
- Cost per trillion tokens
- MFU comparison
- Memory efficiency
- Training time estimates
- Power efficiency
- ROI analysis

---

## 🔌 ROCm Compatibility

**TensorPrimat works seamlessly with both NVIDIA and AMD GPUs** without code modifications.

### How It Works

AMD's ROCm provides CUDA API compatibility through **HIP (Heterogeneous Interface for Portability)**.

### Supported APIs (Work on Both)

```python
torch.cuda.is_available()          # ✓ Works on both
torch.cuda.device_count()          # ✓ Works on both
torch.cuda.get_device_name(0)      # ✓ Works on both
torch.cuda.get_device_properties() # ✓ Works on both
torch.cuda.memory_allocated()      # ✓ Works on both
torch.cuda.synchronize()           # ✓ Works on both
```

### Platform Detection

```python
# Automatic detection
is_rocm = hasattr(torch.version, 'hip') and torch.version.hip is not None
software_stack = "rocm" if is_rocm else "cuda"
software_version = torch.version.hip if is_rocm else torch.version.cuda
```

**Detection Points:**
- **NVIDIA (CUDA)**: `torch.version.cuda` is set (e.g., "12.8")
- **AMD (ROCm)**: `torch.version.hip` is set (e.g., "6.3.0")

### GPU Metrics

**NVIDIA (CUDA):**
- Cores: CUDA Cores (e.g., H100: 16,896)
- Device: "NVIDIA H100 80GB HBM3"

**AMD (ROCm):**
- Cores: Stream Processors (e.g., MI300X: 19,456 SPs)
- Device: "AMD Instinct MI300X"

### Log Analysis Mode (No GPU)

Run without a GPU present:
```bash
# Copy logs from training server
scp training-server:~/logs/*.log .

# Analyze locally (no GPU needed)
./benchmark.py
```

---

## 🛠️ Core Components

### Main Scripts

| File | Purpose |
|------|---------|
| **`benchmark.py`** | Main entrypoint - runs everything |
| **`compare_results.py`** | Generate comparison reports |
| **`compare_with_enhanced_metrics.py`** | Enhanced metrics report |
| **`benchmark_utils.py`** | Core benchmarking framework |
| **`extract_primus_metrics.py`** | Extract from Primus logs |
| **`enhanced_metrics.py`** | Advanced metric calculations |

### Training Scripts (NeMo)

| File | Model |
|------|-------|
| `pretrain_llama.py` | Llama 3.1 8B |
| `pretrain_qwen.py` | Qwen 2.5 7B |

All include automatic benchmarking via `BenchmarkCallback`.

### Primus Scripts (AMD)

| Script | Purpose |
|--------|---------|
| `run_primus_llama.sh` | Run Llama 3.1 8B training |
| `run_primus_qwen.sh` | Run Qwen 2.5 7B training |
| `run_primus_all.sh` | Run all models in sequence |

### Utility Scripts

| Script | Purpose |
|--------|---------|
| `fix_gpu_memory.sh` | Clean up GPU memory |
| `check_primus_config.sh` | Verify Primus configuration |
| `run_amd_dual_comparison.sh` | Run both comparison types |

---

## 🔬 Advanced Usage

### How Automatic Log Detection Works

On AMD platforms without NeMo:

**Search Strategy:**
1. **Environment Variables**: `LLAMA_LOG`, `QWEN_LOG`
2. **Standard Filenames**: `training_llama.log`, `training_qwen.log`, etc.
3. **Pattern Matching**: `*llama*.log`, `primus_*llama*.log`
4. **Multiple Directories**: Current dir, `output/`, `/workspace/Primus/`, `/workspace/tprimat/`
5. **Content-Based Search**: Scans all `.log` and `.txt` files for model keywords

**Result:** No need to specify log paths manually!

### Customizing Benchmarks

Edit training scripts:

```python
# In pretrain_llama.py (or qwen)

# Change number of steps
recipe.trainer.max_steps = 20  # Default is 10

# Change batch size
recipe.data.global_batch_size = 256  # Default is 128

# Customize output directory
benchmark_callback = BenchmarkCallback(
    output_dir="./my_results",
    model_name="llama"
)
```

### Manual Log Extraction

```bash
python3 extract_primus_metrics.py \
    --log-file training_llama.log \
    --model-name llama \
    --num-gpus 8 \
    --global-batch-size 128 \
    --sequence-length 2048
```

### Multiple Runs for Statistics

```bash
# Run 5 times for better statistics
./benchmark.py --runs 5

# Or manually
for i in {1..5}; do
    echo "Run $i"
    ./benchmark.py
    sleep 30  # Cool down
done
```

---

## 💡 Best Practices

### For Accurate Benchmarks

1. ✅ **Run multiple times** (3-5 runs) for statistical significance
2. ✅ **Close other GPU applications** during benchmarking
3. ✅ **Let warmup complete** (first step is automatically excluded)
4. ✅ **Clean GPU memory** before each run (`./fix_gpu_memory.sh`)
5. ✅ **Document your setup** (GPU model, driver versions, configs)
6. ✅ **Monitor during run** (`nvidia-smi` or `rocm-smi`)

### For Fair Comparisons

1. ✅ **Decide on methodology**: Maximum performance or identical configuration
2. ✅ **Document configurations**: TP, PP, precision, batch sizes
3. ✅ **Use same workload**: Same model, same data, same steps
4. ✅ **Check MFU**: Both platforms should have reasonable utilization
5. ✅ **Report context**: Explain why configurations differ (if they do)

### For Production Use

1. ✅ **Consider cost**: Not just performance, but $/token
2. ✅ **Consider memory**: Can you fit your target model?
3. ✅ **Consider scaling**: How does it scale to your cluster size?
4. ✅ **Consider ecosystem**: Framework maturity, support, tooling
5. ✅ **Run long tests**: Not just 10 steps, but 100+ for stability

---

## 📋 Reference

### Quick Commands

| Command | Purpose |
|---------|---------|
| `./benchmark.py` | Run all models on current platform |
| `./benchmark.py --model llama` | Run single model |
| `./benchmark.py --runs 3` | Run 3 times per model |
| `python3 compare_results.py` | Generate comparison report |
| `python3 compare_with_enhanced_metrics.py` | Enhanced metrics report |
| `./fix_gpu_memory.sh` | Clean GPU memory |
| `./check_primus_config.sh` | Verify Primus config |
| `./run_primus_all.sh` | Run all Primus models |

### Environment Variables

| Variable | Purpose |
|----------|---------|
| `LLAMA_LOG` | Path to Llama training log |
| `QWEN_LOG` | Path to Qwen training log |
| `PRIMUS_PATH` | Primus installation directory |
| `TRAIN_ITERS` | Number of training iterations |

### Output Structure

```
output/
├── benchmark_cuda_llama.json      # NVIDIA Llama results
├── benchmark_cuda_qwen.json       # NVIDIA Qwen results
├── benchmark_rocm_llama.json      # AMD Llama results
├── benchmark_rocm_qwen.json       # AMD Qwen results
├── training_*.log                 # Training logs (Primus)
├── comparison_plot.png            # Visual comparison
└── comparison_report.md           # Detailed report
```

### JSON Output Format

```json
{
  "platform": "nvd",
  "gpu_info": {
    "device_name": "NVIDIA H100 80GB HBM3",
    "device_count": 8,
    "gpu_cores": 16896,
    "software_stack": "cuda",
    "software_version": "12.8",
    "pytorch_version": "2.7.0"
  },
  "performance_metrics": {
    "avg_step_time_seconds": 1.602,
    "tokens_per_second": 40918.340,
    "tokens_per_second_per_gpu": 5114.792,
    "steps_per_second": 0.624
  },
  "memory_metrics": {
    "avg_memory_allocated_gb": 22.295,
    "peak_memory_allocated_gb": 22.295
  }
}
```

All float values are automatically rounded to 3 decimal places.

### Workflow Diagram

```
NVIDIA (NeMo):
  ./benchmark.py  →  pretrain_llama.py  →  benchmark_cuda_llama.json
                  →  pretrain_qwen.py   →  benchmark_cuda_qwen.json

AMD (Primus):
  training_llama.log  →  extract_primus_metrics.py  →  benchmark_rocm_llama.json
  training_qwen.log   →  extract_primus_metrics.py  →  benchmark_rocm_qwen.json

Both Platforms:
  output/benchmark_*.json  →  compare_results.py  →  comparison_plot.png
                                                  →  comparison_report.md
```

---

## 🆕 Features

- ✅ **Single Python entrypoint** - `./benchmark.py` does everything
- ✅ **Automatic platform detection** - works on NVIDIA and AMD
- ✅ **Smart log discovery** - finds logs automatically
- ✅ **Model-based filenames** - single file per model
- ✅ **Automatic rounding** - clean 3-decimal JSON output
- ✅ **Environment variable support** - specify log paths easily
- ✅ **Beautiful CLI output** - colored, formatted, clear
- ✅ **Comprehensive metrics** - tokens/sec/GPU, memory, timing
- ✅ **Fair comparison** - multiple methodologies supported
- ✅ **Enhanced metrics** - MFU, cost, power, training time
- ✅ **ROCm compatibility** - seamless AMD support via HIP
- ✅ **Log analysis mode** - no GPU required

---

## 🔄 Version History

**Version 2.0** (Current)
- Python main entrypoint
- Automatic platform detection
- Smart log file discovery
- Model-based output filenames
- Enhanced metrics (MFU, cost, power)
- Improved comparison methodology
- Primus training scripts
- ROCm compatibility

**Version 1.0**
- Initial shell script implementation
- Basic benchmarking functionality

---

**Version**: 2.0  
**Compatible with**: NeMo 24.x+, PyTorch 2.x+  
**Platforms**: NVIDIA CUDA, AMD ROCm  
**Python**: 3.8+

**Happy Benchmarking! 🎉**
