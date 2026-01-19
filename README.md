# TPrimat - LLM Training Benchmark Suite

**Tensor Parallelism AMD/NVIDIA Training** - Comprehensive LLM training benchmarks comparing AMD MI300X vs NVIDIA H100 GPUs with multiple parallelism strategies.

[![Platform](https://img.shields.io/badge/Platform-AMD%20MI300X%20%7C%20NVIDIA%20H100-blue)]()
[![Models](https://img.shields.io/badge/Models-Llama%203.1%208B%20%7C%20Qwen%202.5%207B-green)]()
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)]()

---

## 📋 Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
  - [Basic Commands](#basic-commands)
  - [Parallelism Strategies](#parallelism-strategies)
  - [Examples](#examples)
- [Configuration](#configuration)
  - [Parallelism Strategies Explained](#parallelism-strategies-explained)
  - [Tensor Parallelism Constraints](#tensor-parallelism-constraints)
- [Profiling](#profiling)
  - [Enabling Kineto Profiler](#enabling-kineto-profiler)
  - [Viewing Profile Results](#viewing-profile-results)
- [Troubleshooting](#troubleshooting)
- [Output Files](#output-files)
- [Advanced Topics](#advanced-topics)
- [API Reference](#api-reference)

---

## Overview

TPrimat is a comprehensive benchmarking suite for comparing LLM training performance between AMD and NVIDIA GPUs using different parallelism strategies.

### Key Features

- **Multi-Platform**: Supports AMD MI300X (ROCm/Primus) and NVIDIA H100 (CUDA/NeMo)
- **Multiple Models**: Llama 3.1 8B and Qwen 2.5 7B
- **5 Parallelism Strategies**: From TP=1 (fastest) to TP=4, PP=2 (memory-optimized)
- **Automatic Benchmarking**: JSON metrics + training logs + comparison plots
- **Fair Comparisons**: Identical configurations for apples-to-apples hardware comparison

### Supported Hardware

| Platform | GPU | Memory | Framework | Software Stack |
|----------|-----|--------|-----------|----------------|
| NVIDIA | H100 80GB | 80GB HBM3 | NeMo | CUDA 12.8 |
| AMD | MI300X | 192GB HBM3 | Primus | ROCm 6.x |

### Supported Models

| Model | Parameters | Attention Heads | Valid TP Values |
|-------|-----------|-----------------|-----------------|
| Llama 3.1 8B | 8.0B | 32 | 1, 2, 4, 8, 16, 32 |
| Qwen 2.5 7B | 7.6B | 28 | 1, 2, 4, 7, 14, 28 |

---

## Quick Start

### Run a single benchmark

```bash
# Default configuration
./benchmark.py

# Specific parallelism strategy
./benchmark.py --parallel minimal_communication

# Single model with custom output
./benchmark.py --model llama --parallel balanced --output-dir ./results
```

### Run all 5 configurations

```bash
./run_all_configs.sh
```

This creates `output-00/` through `output-04/` with complete benchmarks and comparison plots.

---

## Installation

### Prerequisites

```bash
# Python 3.10+
python3 --version

# For NVIDIA
nvidia-smi  # Verify CUDA GPUs

# For AMD
rocm-smi    # Verify ROCm GPUs
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

**Dependencies:**
- PyTorch (with CUDA or ROCm)
- NeMo (NVIDIA) or Primus (AMD)
- PyYAML, matplotlib, numpy

---

## Usage

### Basic Commands

```bash
# Run all models with default settings
./benchmark.py

# Specify model
./benchmark.py --model llama
./benchmark.py --model qwen
./benchmark.py --model all  # default

# Specify parallelism strategy
./benchmark.py --parallel maximum_performance
./benchmark.py --parallel minimal_communication

# Custom output directory
./benchmark.py --output-dir ./my-results

# Multiple runs for statistical significance
./benchmark.py --runs 3

# Combine options
./benchmark.py \
    --model llama \
    --parallel balanced \
    --output-dir ./llama-balanced \
    --runs 3
```

### Command-Line Arguments

| Argument | Options | Description |
|----------|---------|-------------|
| `--model` | `llama`, `qwen`, `all` | Which model(s) to benchmark (default: `all`) |
| `--parallel` | See strategies below | Parallelism strategy (default: `minimal_communication`) |
| `--output-dir` | Path | Where to save results and logs (default: `./output`) |
| `--runs` | Number | Repetitions per model (default: `1`) |
| `--help` | - | Show all options |

### Parallelism Strategies

TPrimat supports 5 different parallelism configurations:

```bash
# TP=1, PP=1 - No tensor parallelism (DEFAULT, fastest if model fits)
./benchmark.py --parallel minimal_communication

# TP=4, PP=1 - Platform-optimized for speed
./benchmark.py --parallel maximum_performance

# TP=4, PP=1/2 - Same on both platforms (fair comparison)
./benchmark.py --parallel truly_identical

# TP=4, PP=2 - Minimize memory per GPU
./benchmark.py --parallel memory_optimized

# TP=2, PP=1/2 - Balanced approach
./benchmark.py --parallel balanced
```

### Examples

#### Example 1: Quick Performance Test

```bash
# Test fastest configuration
./benchmark.py --parallel minimal_communication
```

#### Example 2: Compare Different Strategies

```bash
# Test TP=1
./benchmark.py --parallel minimal_communication --output-dir ./tp1

# Test TP=2
./benchmark.py --parallel balanced --output-dir ./tp2

# Test TP=4
./benchmark.py --parallel truly_identical --output-dir ./tp4

# Generate comparison plots
python3 compare.py --results-dir ./tp1
python3 compare.py --results-dir ./tp2
python3 compare.py --results-dir ./tp4
```

#### Example 3: Fair Hardware Comparison

```bash
# On NVIDIA machine
./benchmark.py --parallel truly_identical --output-dir ./nvidia-results

# On AMD machine (or later)
./benchmark.py --parallel truly_identical --output-dir ./amd-results

# Both use IDENTICAL settings for fair comparison
```

#### Example 4: Automated Testing

```bash
# Test all 5 configurations automatically
./run_all_configs.sh

# Results in:
#   output-00/ (maximum_performance)
#   output-01/ (truly_identical)
#   output-02/ (memory_optimized)
#   output-03/ (minimal_communication)
#   output-04/ (balanced)
#   all_outputs/compare-*.png
```

#### Example 5: Memory-Constrained Setup

```bash
# Use memory-optimized configuration (TP=4, PP=2)
./benchmark.py --parallel memory_optimized
```

---

## Configuration

### Parallelism Strategies Explained

Each strategy offers different trade-offs between memory, speed, and communication:

#### 00: Maximum Performance

**Goal:** Best performance for each platform

**Configuration:**
- **NVIDIA (H100):**
  - Llama: TP=4, PP=1, DP=2
  - Qwen: TP=4, PP=2, DP=1
- **AMD (MI300X):**
  - Both: TP=1, PP=1, DP=8 (leverages 192GB memory)

**When to use:** Want best raw performance on each platform

#### 01: Identical Config

**Goal:** Fair hardware comparison

**Configuration:**
- **Both platforms use same settings:**
  - Llama: TP=4, PP=1, DP=2
  - Qwen: TP=4, PP=2, DP=1

**When to use:** Apples-to-apples hardware comparison

#### 02: Memory Optimized

**Goal:** Minimize memory per GPU

**Configuration:**
- **Both models:** TP=4, PP=2, DP=1

**When to use:** Limited GPU memory (24-40GB per GPU)

#### 03: Minimal Communication

**Goal:** Maximum speed (no TP communication overhead)

**Configuration:**
- **All:** TP=1, PP=1, DP=8

**When to use:** Model fits in single GPU memory (best performance)

#### 04: Balanced

**Goal:** Balance between memory and communication

**Configuration:**
- Llama: TP=2, PP=1, DP=4
- Qwen: TP=2, PP=2, DP=2

**When to use:** General-purpose middle ground

### Configuration Reference

| Config | Name | Llama (NVIDIA) | Qwen (NVIDIA) | Use Case |
|--------|------|----------------|---------------|----------|
| 00 | maximum_performance | TP=4, PP=1, DP=2 | TP=4, PP=2, DP=1 | Best performance |
| 01 | truly_identical | TP=4, PP=1, DP=2 | TP=4, PP=2, DP=1 | Fair comparison |
| 02 | memory_optimized | TP=4, PP=2, DP=1 | TP=4, PP=2, DP=1 | Save memory |
| 03 | minimal_communication | TP=1, PP=1, DP=8 | TP=1, PP=1, DP=8 | Fastest |
| 04 | balanced | TP=2, PP=1, DP=4 | TP=2, PP=2, DP=2 | Balanced |

### Tensor Parallelism Constraints

**Critical Rule:** Tensor Parallelism (TP) must evenly divide the number of attention heads.

#### Model Specifications

**Llama 3.1 8B:**
- **32 attention heads**
- Valid TP: 1, 2, 4, 8, 16, 32
- Most common: TP=1, 2, 4

**Qwen 2.5 7B:**
- **28 attention heads**
- Valid TP: 1, 2, 4, 7, 14, 28
- Most common: TP=1, 2, 4

**Common Valid TP (both models):** 1, 2, 4

#### Why This Matters

```yaml
# ✅ Valid: TP=4 works for both
llama: 32 heads ÷ 4 = 8 heads per GPU
qwen:  28 heads ÷ 4 = 7 heads per GPU

# ❌ Invalid: TP=8 fails for Qwen
llama: 32 heads ÷ 8 = 4 heads per GPU ✅
qwen:  28 heads ÷ 8 = 3.5 heads per GPU ❌ NOT AN INTEGER!
```

**Error you'll see:**
```
ValueError: num_attention_heads (28) must be a multiple of tensor_model_parallel_size (8).
```

#### GPU Count Validation

**Rule:** TP × PP × DP must equal total GPUs (8 in this setup)

**Valid combinations for 8 GPUs:**
- 1 × 1 × 8 ✅
- 2 × 1 × 4 ✅
- 4 × 1 × 2 ✅
- 2 × 2 × 2 ✅
- 4 × 2 × 1 ✅
- 8 × 1 × 1 ✅ (but only for Llama!)

---

## Profiling

TPrimat includes integrated Kineto profiling for detailed GPU performance analysis. Kineto captures kernel-level traces, memory allocations, and CPU activity.

### Enabling Kineto Profiler

**1. Edit `config.yaml` to enable profiling:**

```yaml
profiling:
  enabled: true                     # Enable Kineto profiling
  export_chrome_trace: true         # Export Chrome trace JSON
  profile_memory: true              # Track memory allocations
  with_stack: true                  # Include Python stack traces
  with_flops: true                  # Estimate FLOPs
  record_shapes: true               # Record tensor shapes
  schedule:
    wait: 1                         # Skip first N steps (warmup)
    warmup: 1                       # Warmup for N steps
    active: 5                       # Profile N steps
    repeat: 1                       # Repeat cycle N times
```

**2. Run your benchmark:**

```bash
./benchmark.py --parallel minimal_communication
```

**3. Find profiler output in your output directory:**

```
output/
├── profile_cuda_llama_*.json                # Profiler traces
├── benchmark_cuda_llama.json                # Benchmark metrics
└── training_llama.log                       # Training logs
```

### Profiler Configuration

**Schedule Settings:**

- `wait`: Skip initial steps (avoid startup overhead)
- `warmup`: Warmup steps before profiling
- `active`: Number of steps to profile
- `repeat`: How many times to repeat the cycle

**Example schedules:**

```yaml
# Quick profile (5 steps)
schedule: {wait: 1, warmup: 1, active: 5, repeat: 1}

# Extended profile (20 steps)
schedule: {wait: 1, warmup: 2, active: 10, repeat: 2}

# Single step detail
schedule: {wait: 2, warmup: 1, active: 1, repeat: 1}
```

**Profiler Options:**

- `profile_memory`: Track GPU memory allocations (adds overhead)
- `with_stack`: Include Python stack traces (helps identify bottlenecks)
- `with_flops`: Estimate floating-point operations
- `record_shapes`: Record tensor shapes (useful for debugging)

### Viewing Profile Results

#### Method 1: TensorBoard (Recommended)

```bash
# Start TensorBoard
tensorboard --logdir=./output/profiler

# Open browser to http://localhost:6006
# Navigate to "PROFILE" tab
```

**What you'll see:**
- Kernel execution timeline
- Memory usage over time
- Operator breakdown (forward, backward, optimizer)
- Distributed training communication patterns
- Per-GPU utilization

#### Method 2: Chrome Trace Viewer

```bash
# Decompress trace file
# No longer gzipped by default, saved directly in output/
ls output/profile_cuda_llama_*.json

# Open Chrome and navigate to: chrome://tracing
# Click "Load" and select the .json file
```

**What you'll see:**
- GPU kernel timeline (CUDA operations)
- CPU activity (Python, data loading)
- Memory allocations/deallocations
- Synchronization points

#### Method 3: Command-Line Analysis

```python
import torch.profiler as profiler

# Load and analyze profile
trace_path = "output/profile_cuda_llama_*.json"
# ... analysis code ...

# Print top operations by CUDA time
print(prof.key_averages().table(
    sort_by="cuda_time_total",
    row_limit=20
))
```

### Profiling Tips

**1. Minimize Overhead:**
- Only profile a few steps (5-10)
- Use `wait=1` to skip warmup
- Disable `with_stack` for faster profiling

**2. Distributed Training:**
- Profiling only runs on rank 0 (automatically)
- Each rank would generate its own trace otherwise

**3. Storage Considerations:**
- Chrome traces can be large (50-200MB per model)
- Use `.json.gz` format (10x compression)
- Clean up old traces regularly

**4. Performance Analysis:**
```bash
# Profile fast config
./benchmark.py --parallel minimal_communication

# Profile memory-optimized config
./benchmark.py --parallel memory_optimized

# Compare kernel execution times in TensorBoard
```

### Example: Finding Bottlenecks

**1. Enable profiling and run benchmark:**

```bash
# Edit config.yaml: profiling.enabled = true
./benchmark.py --model llama --parallel minimal_communication
```

**2. View in TensorBoard:**

```bash
tensorboard --logdir=./output/profiler
```

**3. Look for:**
- Long-running kernels (potential optimization targets)
- GPU idle time (data loading bottlenecks)
- Memory spikes (potential OOM causes)
- Communication overhead (collective operations)

**4. Common bottlenecks:**
- **Data loading**: CPU-to-GPU transfer time
- **Attention kernels**: Flash attention vs standard
- **All-reduce**: Gradient synchronization time
- **Memory copies**: Host-device transfers

### Disabling Profiling

```yaml
# config.yaml
profiling:
  enabled: false  # Disable profiling
```

Or remove profiler overhead entirely for production benchmarks.

### Advanced Profiling

**Profile specific training steps:**

Modify `utils.py` to profile specific iterations:

```python
# In BenchmarkCallback
def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
    # Only profile steps 10-15
    if batch_idx == 10 and self.profiler is None:
        self._start_profiler()
    elif batch_idx == 15 and self.profiler is not None:
        self._stop_profiler()
```

**Export profiles programmatically:**

```python
# After training
prof.export_chrome_trace("custom_trace.json")
prof.export_stacks("flame_graph.txt", "self_cuda_time_total")
```

---

## Troubleshooting

### GPU Detection Issues

#### Problem: "No supported gpu backend found!"

**Symptoms:**
```
MisconfigurationException: No supported gpu backend found!
Can't initialize NVML
No CUDA runtime is found
```

**Causes:**
1. GPUs still allocated from previous run
2. CUDA context not released
3. GPU memory locked by zombie processes

**Solutions:**

**1. Add cooldown between models (automatic now):**
```bash
./benchmark.py  # Now includes 20-second cooldown
```

**2. Run models separately:**
```bash
python3 pretrain_llama.py
sleep 30
python3 pretrain_qwen.py
```

**3. Clear GPU memory manually:**
```bash
# Check GPU status
nvidia-smi

# Kill Python processes
pkill -f python

# Reset GPUs (requires sudo)
sudo nvidia-smi --gpu-reset
```

**4. Check for zombie processes:**
```bash
# Find processes using GPUs
fuser -v /dev/nvidia*

# Kill specific process
kill -9 <PID>
```

### Tensor Parallelism Errors

#### Problem: "num_attention_heads must be a multiple of tensor_model_parallel_size"

**Error:**
```
ValueError: num_attention_heads (28) must be a multiple of tensor_model_parallel_size (8).
```

**Solution:** Use compatible TP values (1, 2, or 4 work for both models):

```bash
# These work for both models
./benchmark.py --parallel balanced  # Uses TP=2
./benchmark.py --parallel minimal_communication  # Uses TP=1
./benchmark.py --parallel truly_identical  # Uses TP=4
```

### Memory Issues

#### Problem: CUDA Out of Memory (OOM)

**Solutions:**

**1. Use higher tensor parallelism:**
```bash
./benchmark.py --parallel memory_optimized  # Uses TP=4, PP=2
```

**2. Reduce batch size in `config.yaml`:**
```yaml
training:
  data:
    micro_batch_size: 1
    global_batch_size: 64  # Reduce if needed
```

**3. Check available memory:**
```bash
nvidia-smi --query-gpu=memory.free --format=csv
```

### Configuration Not Applied

#### Problem: All configurations show identical performance

**Cause:** Parallelism strategy not being specified

**Solution:** Use `--parallel` flag:
```bash
./benchmark.py --parallel minimal_communication
```

**Verification:** Check log output for:
```
🔧 Using parallelism strategy: minimal_communication
🔧 Parallelism: TP=1, PP=1, DP=8, GradAccum=16
```

### No Benchmark Results Found

**Problem:** Comparison script can't find JSON files

**Check:**
```bash
# Verify files exist
ls output/*.json

# Check logs for errors
cat output/training_llama.log | grep -i "error"

# Manually extract metrics (for AMD/Primus)
python3 extract_primus_metrics.py \
    --log-file output/training_llama.log \
    --model-name llama \
    --num-gpus 8 \
    --global-batch-size 128 \
    --sequence-length 2048
```

### Performance Issues

#### Training is Very Slow

**Common causes:**

1. **High TP (TP=8)** → More communication overhead
   - Solution: Try TP=1 if model fits

2. **Pipeline parallelism (PP>1)** → Pipeline bubbles
   - Solution: Use PP=1 if possible

3. **Network bottleneck** → Slow inter-GPU communication
   - Check topology: `nvidia-smi topo -m`

4. **Too many gradient accumulation steps**
   - This is expected for large batch sizes

**Performance tuning:**
```bash
# Try minimal communication (fastest)
./benchmark.py --parallel minimal_communication  # TP=1

# Check MFU (should be 30-50% for good performance)
# Look in benchmark JSON output
```

### Platform-Specific Issues

#### NVIDIA Issues

**1. FP8 not working:**
- Requires H100 GPU
- Check: `nvidia-smi -q | grep "Product Name"`

**2. NVLink issues:**
- Verify: `nvidia-smi nvlink -s`

#### AMD Issues

**1. ROCm not detected:**
```bash
# Check ROCm
rocm-smi

# Check PyTorch ROCm
python3 -c "import torch; print(torch.version.hip)"
```

**2. Primus configuration:**
- Ensure `PRIMUS_PATH` is set
- Check: `echo $PRIMUS_PATH`

### Quick Diagnostic Checklist

- [ ] GPUs visible (`nvidia-smi` or `rocm-smi`)
- [ ] PyTorch sees GPUs (`python3 -c "import torch; print(torch.cuda.is_available())"`)
- [ ] No zombie processes holding GPU memory
- [ ] Correct parallelism strategy set (`--parallel`)
- [ ] Output directory exists and writable
- [ ] Sufficient GPU memory for model
- [ ] Cooldown time between model runs

### Common Commands

```bash
# Check GPU status
nvidia-smi

# Monitor in real-time
watch -n 1 nvidia-smi

# Check running processes
ps aux | grep python

# Kill all Python processes (careful!)
pkill -f python

# View logs
tail -f output/training_llama.log

# Search for errors
grep -i "error\|failed\|exception" output/*.log

# Count completed steps
grep "elapsed time per iteration" output/training_llama.log | wc -l
```

---

## Output Files

After running a benchmark, you'll find everything in your specified output directory:

```
output/  (or --output-dir path)
├── benchmark_cuda_llama.json      # NVIDIA Llama metrics
├── benchmark_cuda_qwen.json       # NVIDIA Qwen metrics
├── benchmark_rocm_llama.json      # AMD Llama metrics (if available)
├── benchmark_rocm_qwen.json       # AMD Qwen metrics (if available)
├── training_llama.log             # Complete training logs
└── training_qwen.log              # Complete training logs
```

### Benchmark JSON Structure

```json
{
  "platform": "nvd",
  "gpu_info": {
    "device_name": "NVIDIA H100 80GB HBM3",
    "device_count": 8,
    "software_stack": "cuda"
  },
  "parallelism_config": {
    "tensor_model_parallel_size": 4,
    "pipeline_model_parallel_size": 1,
    "data_parallel_size": 2,
    "gradient_accumulation_steps": 64,
    "strategy_name": "truly_identical"
  },
  "training_config": {
    "global_batch_size": 128,
    "sequence_length": 2048,
    "num_gpus": 8
  },
  "performance_metrics": {
    "tokens_per_second": 11114.48,
    "tokens_per_second_per_gpu": 1389.31,
    "avg_step_time_seconds": 23.59,
    "steps_per_second": 0.0424
  },
  "raw_step_times": [...],
  "raw_loss_values": [...],
  "raw_memory_values": [...]
}
```

### Parallelism Configuration in JSON

Each benchmark JSON now includes a `parallelism_config` section that records:

- **tensor_model_parallel_size** (TP) - How many GPUs share model tensors
- **pipeline_model_parallel_size** (PP) - Pipeline stages
- **data_parallel_size** (DP) - Data parallelism degree
- **gradient_accumulation_steps** - Gradient accumulation
- **strategy_name** - Which strategy was used (e.g., "balanced", "minimal_communication")

This makes it easy to track which configuration was used for each benchmark result.

### Generating Comparison Plots

```bash
# Compare specific directory
python3 compare.py --results-dir ./output

# Generate all comparison plots
./compare_all_configs.sh

# Results:
#   all_outputs/compare-00.png
#   all_outputs/compare-01.png
#   ...
```

### Comparison Plot Contents

Each plot includes 6 subplots:

1. **Per-GPU Throughput (TFLOP/s/GPU)** - Bar chart
2. **Memory Usage (GB)** - Bar chart
3. **Training Loss over Time** - Line plot
4. **Learning Rate over Time** - Line plot
5. **Memory Usage over Time** - Line plot
6. **Step Duration over Time** - Line plot

---

## Peak Throughput Analysis

TPrimat includes comprehensive tools for analyzing peak throughput across all your benchmark configurations.

### Quick Start

```bash
# Analyze peak throughput across all configurations
python3 analyze_peak_throughput.py

# Show all configurations (not just peaks)
python3 analyze_peak_throughput.py --show-all

# Generate visualization
python3 visualize_peak_throughput.py
```

### What You Get

**1. Peak Throughput Analysis (`analyze_peak_throughput.py`)**

Comprehensive analysis showing:
- Peak tokens/second for each platform-model combination
- Total TFLOPS achieved
- Hardware utilization percentage
- Per-GPU performance metrics
- Multi-node scaling projections
- Daily training capacity estimates

**Example output:**
```
🥇 OVERALL PEAK THROUGHPUT
Best Configuration: AMD Instinct MI300X - qwen
Peak Throughput:    87,933 tokens/s
                    4,009.8 TFLOPS total
                    501.2 TFLOPS per GPU
Hardware:           8 × AMD Instinct MI300X
HW Utilization:     38.3%

Projected Training Capacity (24h at peak):
  Tokens:         7,597,433,417 (7.6B tokens)
  Samples:        3,709,684
```

**2. Peak Throughput Visualization (`visualize_peak_throughput.py`)**

Creates `peak_throughput_analysis.png` with 9 comprehensive plots:
- Peak throughput comparison (tokens/s)
- Peak compute performance (TFLOPS)
- Hardware utilization by configuration
- Per-GPU throughput and TFLOPS
- Average step time comparison
- Configuration distribution (box plots)
- Best platform performance

**3. Peak Throughput Guide (`PEAK_THROUGHPUT_GUIDE.md`)**

Detailed documentation including:
- How to estimate peak throughput for your hardware
- Understanding TFLOPS and hardware utilization
- Configuration impact analysis
- Scaling projections
- Performance optimization recommendations

### Key Insights from Analysis

**Platform Winners:**
- **AMD MI300X**: Peak 87,933 tokens/s (38.3% utilization)
- **NVIDIA H100**: Peak 78,877 tokens/s (22.7% utilization)
- **Speedup**: AMD is 1.11x faster at peak

**Configuration Impact:**
- **Best configs**: 60-88K tokens/s
- **Worst configs**: 6-20K tokens/s
- **Range**: Up to 14x performance difference!

**Hardware Utilization:**
- **Good**: 30-40% (realistic for LLM training)
- **Average**: 15-25%
- **Poor**: 5-10%

### Understanding Peak Metrics

#### Tokens per Second
Raw throughput for processing training tokens. Higher is better.

#### TFLOPS (Tera Floating-Point Operations per Second)
Computational intensity. Calculated as:
```
TFLOPS = (tokens/s × FLOPs_per_token) / 1e12
```

For our models:
- Llama 3.1 8B: 48 billion FLOPs per token
- Qwen 2.5 7B: 45.6 billion FLOPs per token

#### Hardware Utilization
Percentage of theoretical peak performance achieved:
```
Utilization = (Achieved TFLOPS / Theoretical Peak) × 100
```

Theoretical peaks:
- AMD MI300X: 1,307 TFLOPS per GPU
- NVIDIA H100: 1,979 TFLOPS per GPU

#### Why Not 100% Utilization?

Real-world LLM training achieves 20-40% utilization due to:
1. Memory bandwidth bottlenecks
2. Multi-GPU communication overhead
3. Framework inefficiencies
4. Non-compute operations (I/O, data loading)

**38% utilization is excellent for production workloads!**

### Estimating Peak for Your Setup

**Step 1: Run benchmarks across configurations**
```bash
./run_all_configs.sh
```

**Step 2: Analyze peak performance**
```bash
python3 analyze_peak_throughput.py --show-all > analysis.txt
```

**Step 3: Identify best configuration**
Look for highest tokens/s in the results table.

**Step 4: Project to your scale**
```python
# Example: Scale to 4 nodes (32 GPUs)
peak_1_node = 87_933  # tokens/s
scaling_efficiency = 0.90  # 90% (realistic)
nodes = 4

projected_throughput = peak_1_node * nodes * scaling_efficiency
# = 316,547 tokens/s
```

### Multi-Node Scaling

Based on peak single-node performance (87,933 tokens/s):

| Nodes | GPUs | Tokens/s (Linear) | Tokens/s (90% Eff) | Daily Tokens |
|-------|------|-------------------|-------------------|--------------|
| 1 | 8 | 87,933 | 87,933 | 7.6B |
| 2 | 16 | 175,867 | 158,280 | 13.7B |
| 4 | 32 | 351,733 | 316,560 | 27.4B |
| 8 | 64 | 703,466 | 633,119 | 54.7B |

**Note**: Requires high-speed interconnect (InfiniBand, RoCE) for efficient scaling.

### Performance Recommendations

**For Maximum Throughput:**
1. Use the best platform-model combo (AMD MI300X + Qwen)
2. Choose the best configuration from analysis
3. Monitor hardware utilization (aim for >30%)
4. Test different parallelism strategies

**For Cost Efficiency:**
1. Optimize hardware utilization
2. Use configurations with best TFLOPS per dollar
3. Consider multi-node only if >85% scaling efficiency

**For Memory-Constrained:**
1. Use memory_optimized configuration (TP=4, PP=2)
2. AMD MI300X has 2.4x more memory (192GB vs 80GB)
3. Check per-GPU memory in analysis

### Files Generated

After running the analysis tools:
```
tprimat/
├── peak_throughput_analysis.png    # Comprehensive visualization
├── peak_throughput_summary.txt     # Full text analysis
└── PEAK_THROUGHPUT_GUIDE.md        # Detailed documentation
```

### API Reference

#### analyze_peak_throughput.py
```bash
python3 analyze_peak_throughput.py [OPTIONS]

Options:
  --show-all    Show all configurations, not just peaks
```

**Output:**
- Peak throughput by platform-model
- Overall peak performance
- Throughput scaling projections
- Complete results table

#### visualize_peak_throughput.py
```bash
python3 visualize_peak_throughput.py
```

**Output:**
- Creates `peak_throughput_analysis.png`
- 9 comprehensive comparison plots
- Peak performance highlighted

### Example Workflow

```bash
# 1. Run comprehensive benchmarks
./run_all_configs.sh

# 2. Analyze peak throughput
python3 analyze_peak_throughput.py --show-all

# 3. Generate visualization
python3 visualize_peak_throughput.py

# 4. Read detailed guide
cat PEAK_THROUGHPUT_GUIDE.md
```

### Understanding Your Results

**Good Performance:**
- Hardware utilization: 30-40%
- Tokens/s: >60K for 8 GPUs
- TFLOPS per GPU: >400

**Average Performance:**
- Hardware utilization: 15-25%
- Tokens/s: 20-60K for 8 GPUs
- TFLOPS per GPU: 200-400

**Poor Performance:**
- Hardware utilization: <15%
- Tokens/s: <20K for 8 GPUs
- TFLOPS per GPU: <200

If you're seeing poor performance, try different parallelism strategies!

---

## Advanced Topics

### Environment Variables

Instead of command-line flags, you can use environment variables:

```bash
# Set parallelism strategy
export PARALLEL="minimal_communication"
python3 pretrain_llama.py

# Set output directory
export OUTPUT_DIR="./my-results"
python3 benchmark.py
```

### Running Individual Models

```bash
# Direct script execution
export PARALLEL="balanced"
export OUTPUT_DIR="./output"

python3 pretrain_llama.py
# Wait for GPU memory to clear
sleep 30
python3 pretrain_qwen.py
```

### Custom Configurations

Edit `config.yaml` to add custom parallelism strategies:

```yaml
parallelism:
  my_custom_config:
    llama:
      nvidia:
        tensor_model_parallel_size: 2
        pipeline_model_parallel_size: 1
        data_parallel_size: 4
        gradient_accumulation_steps: 32
```

Then run:
```bash
./benchmark.py --parallel my_custom_config
```

### Model Size Information

From `config.yaml`:

```yaml
models:
  llama:
    model_size_gb: 88                # ~88GB total
    memory_per_gpu_tp4: 22           # ~22GB with TP=4
  
  qwen:
    model_size_gb: 136               # ~136GB total
    memory_per_gpu_tp4_pp2: 17       # ~17GB with TP=4, PP=2
```

### Batch Processing Script

```bash
#!/bin/bash
# test_all_strategies.sh

for strategy in maximum_performance minimal_communication balanced; do
    echo "Testing: $strategy"
    ./benchmark.py \
        --parallel $strategy \
        --output-dir "results-$strategy"
    sleep 30  # Cooldown
done

# Generate comparisons
for dir in results-*; do
    python3 compare.py --results-dir "$dir"
    mv compare.png "${dir}/comparison.png"
done
```

### CI/CD Integration

Example GitHub Actions workflow:

```yaml
name: GPU Benchmark

on: [push]

jobs:
  benchmark:
    runs-on: [self-hosted, gpu]
    
    steps:
      - uses: actions/checkout@v2
      
      - name: Run Benchmark
        run: |
          ./benchmark.py --parallel balanced --output-dir ./results
      
      - name: Upload Results
        uses: actions/upload-artifact@v2
        with:
          name: benchmark-results
          path: ./results/
```

---

## API Reference

### benchmark.py

Main entry point for running benchmarks.

```bash
./benchmark.py [OPTIONS]

Options:
  --model {llama,qwen,all}      Model to benchmark (default: all)
  --parallel STRATEGY           Parallelism strategy (default: minimal_communication)
  --output-dir PATH            Output directory (default: ./output)
  --runs N                     Number of runs per model (default: 1)
  --help                       Show help message
```

### Parallelism Strategies

- `maximum_performance` - Platform-optimized settings
- `truly_identical` - Same settings for fair comparison
- `memory_optimized` - TP=4, PP=2 (saves memory)
- `minimal_communication` - TP=1 (fastest)
- `balanced` - TP=2 (middle ground)

### compare.py

Generate comparison plots.

```bash
python3 compare.py [--results-dir PATH]

Options:
  --results-dir PATH          Directory with benchmark JSON files
                             (default: ./output)
```

### extract_primus_metrics.py

Extract metrics from AMD/Primus training logs.

```bash
python3 extract_primus_metrics.py \
    --log-file PATH \
    --model-name {llama,qwen} \
    --num-gpus 8 \
    --global-batch-size 128 \
    --sequence-length 2048
```

### run_all_configs.sh

Automated testing of all 5 configurations.

```bash
./run_all_configs.sh [llama|qwen|both]

Arguments:
  llama                        Run only Llama configurations
  qwen                         Run only Qwen configurations
  both                         Run both models (default)
```

### compare_all_configs.sh

Generate comparison plots for all configurations.

```bash
./compare_all_configs.sh [OUTPUT_DIR]

Arguments:
  OUTPUT_DIR                   Where to save plots (default: ./all_outputs)
```

### config.yaml

Main configuration file structure:

```yaml
experiment:                    # Experiment metadata
hardware:                      # Platform specifications
  platforms:                   # NVIDIA and AMD configs
models:                        # Model specifications
  llama:                       # Llama 3.1 8B config
  qwen:                        # Qwen 2.5 7B config
training:                      # Training hyperparameters
  data:                        # Batch sizes, sequence length
  optimizer:                   # Learning rate, warmup
parallelism:                   # 5 parallelism strategies
  maximum_performance:         # Config 00
  truly_identical:            # Config 01
  memory_optimized:            # Config 02
  minimal_communication:       # Config 03
  balanced:                    # Config 04
platform_optimizations:        # NVIDIA/AMD specific settings
benchmarking:                  # Metrics and output config
```

### File Structure

```
tprimat/
├── benchmark.py               # Main entry point
├── pretrain_llama.py         # Llama training script
├── pretrain_qwen.py          # Qwen training script
├── compare.py                # Comparison plot generator
├── extract_primus_metrics.py # AMD log parser
├── config_loader.py          # Configuration loader
├── utils.py                  # Benchmark utilities
├── config.yaml               # Main configuration
├── requirements.txt          # Python dependencies
├── run_all_configs.sh        # Run all configurations
├── compare_all_configs.sh    # Generate all plots
├── validate_config.py        # Configuration validator
└── README.md                 # This file
```

---

## Credits & License

**TPrimat** - Tensor Parallelism AMD/NVIDIA Training Benchmark Suite

Developed for fair, reproducible LLM training benchmarks across GPU platforms.

### Frameworks Used

- **NeMo**: NVIDIA's neural module framework
- **Primus**: AMD's training framework
- **Megatron**: Model-parallel transformers

### Models

- **Llama 3.1 8B**: Meta AI
- **Qwen 2.5 7B**: Alibaba Cloud

---

## Quick Reference Card

```bash
# QUICK START
./benchmark.py --parallel minimal_communication

# ALL CONFIGURATIONS
./run_all_configs.sh

# SPECIFIC MODEL
./benchmark.py --model llama --parallel balanced

# CUSTOM OUTPUT
./benchmark.py --output-dir ./my-results

# MULTIPLE RUNS
./benchmark.py --runs 3

# GENERATE PLOTS
python3 compare.py --results-dir ./output

# CHECK GPUS
nvidia-smi  # NVIDIA
rocm-smi    # AMD

# HELP
./benchmark.py --help
```

### Parallelism Quick Guide

| Strategy | TP | PP | When to Use |
|----------|----|----|-------------|
| `minimal_communication` | 1 | 1 | Fastest (if fits) |
| `balanced` | 2 | 1-2 | General purpose |
| `truly_identical` | 4 | 1-2 | Fair comparison |
| `memory_optimized` | 4 | 2 | Save memory |
| `maximum_performance` | Platform-specific | Best performance |

---

**For questions, issues, or contributions, please check the project repository.**

