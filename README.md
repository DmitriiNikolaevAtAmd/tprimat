# TensorPrimat

**Unified benchmarking toolkit for comparing AMD vs NVIDIA GPU performance with LLM training**

Automatically benchmarks LLM training on NVIDIA (NeMo) or AMD (Primus) with a single shared configuration.

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install pyyaml matplotlib numpy

# 2. (Optional) Review/edit configuration
python3 config_loader.py    # View config
vim config.yaml             # Edit if needed

# 3. Run on any platform (NVIDIA or AMD)
./benchmark.py              # Python/NeMo (NVIDIA)
./run_primus_llama.sh       # Shell/Primus (AMD)

# 4. Compare results after running on both platforms
python3 compare.py
```

**All scripts use `config.yaml`** - Edit once, applies everywhere!

**⚠️ Important:** All optimizer settings (learning rate, warmup, etc.) are now synchronized between NVIDIA and AMD platforms via `config.yaml` for fair comparison.

---

## 📋 Table of Contents

- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Unified Configuration](#-unified-configuration)
  - [Quick Start](#configuration-quick-start)
  - [File Structure](#configuration-file-structure)
  - [Basic Usage](#basic-usage)
  - [Access Methods](#access-methods)
  - [Helper Methods](#helper-methods)
  - [Integration Examples](#integration-examples)
  - [Validation](#validation)
- [Learning Rate Synchronization](#-learning-rate-synchronization)
- [Configuration Reference](#-configuration-reference)
  - [Core Concepts](#core-concepts)
  - [Comparison Methodologies](#comparison-methodologies)
  - [Parallelism Explained](#parallelism-explained)
  - [Platform Optimizations](#platform-optimizations)
  - [All Parameters](#all-parameters)
- [Usage](#-usage)
  - [Basic Commands](#basic-commands)
  - [Primus Training Scripts](#primus-training-scripts)
  - [Complete Workflow](#complete-workflow)
- [What Gets Measured](#-what-gets-measured)
- [Enhanced Metrics](#-enhanced-metrics)
- [Troubleshooting](#-troubleshooting)
- [Best Practices](#-best-practices)
- [Advanced Topics](#-advanced-topics)
- [Reference](#-reference)

---

## 💻 Installation

### Prerequisites

**Already installed with NeMo/Primus:**
- Python 3.8+
- PyTorch 2.x+
- CUDA or ROCm

**Additional requirement:**
```bash
pip install pyyaml
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

## ⚙️ Unified Configuration

**Single YAML configuration for both AMD and NVIDIA hardware!**

All experiment settings are now centralized in `config.yaml`.

### Configuration Quick Start

```bash
# Install dependency
pip install pyyaml

# View configuration
python3 config_loader.py

# Run examples
python3 example_config_usage.py
```

### Configuration File Structure

`config.yaml` contains:

```yaml
# Single source of truth for all configurations
experiment:
  name: "tprimat_benchmark"
  methodology: "maximum_performance"  # or "identical_config"

hardware:
  platforms:
    nvidia:
      gpu_model: "H100"
      memory_per_gpu_gb: 80
      num_gpus: 8
    amd:
      gpu_model: "MI300X"
      memory_per_gpu_gb: 192
      num_gpus: 8

models:
  llama:
    name: "llama3.1_8b"
    num_parameters: 8.0e9
  qwen:
    name: "qwen2.5_7b"
    num_parameters: 7.6e9

training:
  data:
    global_batch_size: 128
    micro_batch_size: 1
    seq_length: 2048
  duration:
    max_steps: 10

parallelism:
  maximum_performance:
    llama:
      nvidia: { TP: 4, PP: 1, DP: 2 }   # H100 optimized
      amd:    { TP: 1, PP: 1, DP: 8 }   # MI300X optimized
  
  identical_config:
    llama:
      nvidia: { TP: 4, PP: 1, DP: 2 }   # Same for both
      amd:    { TP: 4, PP: 1, DP: 2 }   # platforms
```

---

## 🔄 Learning Rate Synchronization

**Ensuring Fair Comparison: Unified Optimizer Configuration**

### Problem Identified

Previously, NVIDIA and AMD platforms used **different learning rates**:
- **NVIDIA (NeMo)**: `3.0e-4` from `config.yaml` ✓
- **AMD (Primus)**: `1.0e-5` from hardcoded Primus config files ❌
- **Difference**: 30x different learning rates made comparisons unfair!

### Solution Implemented

All optimizer parameters are now shared via `config.yaml`:

```yaml
training:
  optimizer:
    learning_rate: 3.0e-4    # Peak LR (same for both platforms)
    warmup_steps: 10         # Warmup duration
    weight_decay: 0.1        # Weight decay
    beta1: 0.9              # Adam beta1
    beta2: 0.95             # Adam beta2
```

### How It Works

**NVIDIA (NeMo):**
- `pretrain_llama.py` and `pretrain_qwen.py` read directly from `config.yaml`
- No changes needed - already integrated

**AMD (Primus):**
- `run_primus_llama.sh` and `run_primus_qwen.sh` now export config to shell variables
- Scripts pass learning rate to Primus via command-line arguments:
  ```bash
  bash ./examples/run_pretrain.sh \
      --lr $LEARNING_RATE \
      --min_lr $MIN_LEARNING_RATE \
      --lr_warmup_iters $WARMUP_STEPS \
      --weight_decay $WEIGHT_DECAY
  ```

### Verification

**Check configuration is applied:**

```bash
# Run AMD training - look for these lines in output:
./run_primus_llama.sh

# Should display:
# 📈 Learning Rate: 0.0003
# 📉 Min Learning Rate: 3e-05
# 🔥 Warmup Steps: 10
```

**Verify in logs:**

```bash
# NVIDIA logs
grep "learning" output/training_llama.log

# AMD logs  
grep "lr \." output/primus_training_llama_*.log
# Should show: lr .............................................. 0.0003
```

### Benefits

✅ **Single source of truth** - All hyperparameters in `config.yaml`  
✅ **Fair comparison** - Both platforms use identical training configuration  
✅ **Easy modification** - Change learning rate once, applies everywhere  
✅ **Reproducibility** - No hidden hyperparameters in separate config files  
✅ **Transparency** - Learning rate curves in comparison plots show synchronization

### Comparison Impact

With synchronized learning rates, you'll see:
- **Similar LR schedules** in comparison plots (both warmup, peak, decay)
- **Comparable loss convergence** (same optimization strategy)
- **Fair hardware comparison** (identical software configuration)
- **Valid performance metrics** (throughput, memory remain hardware-dependent)

---

```python
from config_loader import load_config

# Load configuration
config = load_config()

# Access settings (dot notation)
batch_size = config.training.data.global_batch_size  # 128
max_steps = config.training.duration.max_steps       # 10
precision = config.training.precision.default        # "bf16"

# Get parallelism for specific platform
llama_nvidia = config.get_parallelism("llama", "nvidia")
# Returns: {
#   'tensor_model_parallel_size': 4,
#   'pipeline_model_parallel_size': 1,
#   'data_parallel_size': 2,
#   'gradient_accumulation_steps': 64
# }

# Get platform optimizations
opts = config.get_platform_optimizations("nvidia")
# Returns: {'precision': 'fp8', 'fp8_hybrid': True, ...}

# Print summary
config.print_config_summary("llama", "nvidia")
```

### Access Methods

```python
config = load_config()

# Dot notation
config.experiment.name                    # "tprimat_benchmark"
config.training.data.global_batch_size    # 128
config.models.llama.num_parameters        # 8.0e9

# Helper methods
config.get_models_list()                  # ['llama', 'qwen']
config.get_platforms_list()               # ['nvidia', 'amd']
config.get_parallelism("llama", "nvidia") # Dict with TP/PP/DP
config.get_platform_optimizations("amd")  # Dict with opts
config.get_model_config("llama")          # Dict with model info
config.get_hardware_config("nvidia")      # Dict with hw specs
```

### Helper Methods

```python
# Paths and filenames
config.get_output_dir()                          # "./output"
config.get_log_filename("llama")                 # "training_llama.log"
config.get_benchmark_filename("cuda", "llama")   # "benchmark_cuda_llama.json"
config.get_primus_path()                         # "/workspace/Primus"
config.get_primus_config_path("llama")           # Full path to Primus config

# Costs and specs
config.get_cloud_cost("nvidia")                  # 32.0 ($/hour)
config.get_hardware_specs("nvidia")              # Dict with TFLOPs, TDP, etc.

# Training config
config.get_training_config()                     # Dict with all training params
config.get_benchmark_config()                    # Dict with benchmark settings
```

### Integration Examples

#### Training Script

```python
from config_loader import load_config

def run_pretrain():
    config = load_config()
    
    # Get settings
    parallelism = config.get_parallelism("llama", "nvidia")
    optimizations = config.get_platform_optimizations("nvidia")
    
    # Create recipe
    recipe = llm.llama31_8b.pretrain_recipe(
        num_gpus_per_node=config.hardware.platforms.nvidia.num_gpus
    )
    
    # Apply parallelism from config
    recipe.trainer.strategy.tensor_model_parallel_size = \
        parallelism['tensor_model_parallel_size']
    recipe.trainer.strategy.pipeline_model_parallel_size = \
        parallelism['pipeline_model_parallel_size']
    
    # Apply data config
    recipe.data.micro_batch_size = config.training.data.micro_batch_size
    recipe.data.global_batch_size = config.training.data.global_batch_size
    recipe.data.seq_length = config.training.data.seq_length
    
    # Apply optimizations
    if optimizations['fp8_hybrid']:
        recipe.model.config.fp8 = "hybrid"
        recipe.model.config.fp8_param = optimizations.get('fp8_param', False)
    
    # Run
    run.run(recipe, direct=True)
```

#### Benchmark Script

```python
from config_loader import load_config

def benchmark():
    config = load_config()
    
    # Get output paths
    output_dir = config.get_output_dir()
    log_file = config.get_log_filename("llama")
    
    # Run training
    # ...
    
    # Save with configured filename
    filename = config.get_benchmark_filename("cuda", "llama")
    save_results(os.path.join(output_dir, filename))
```

#### Comparison Script

```python
from config_loader import load_config

def compare():
    config = load_config()
    
    # Get costs
    nvidia_cost = config.get_cloud_cost("nvidia")
    amd_cost = config.get_cloud_cost("amd")
    
    # Get hardware specs for MFU
    nvidia_specs = config.get_hardware_specs("nvidia")
    peak_tflops = nvidia_specs['peak_tflops_fp8']
    
    # Calculate MFU
    mfu = (achieved_flops / peak_tflops) * 100
```

### Validation

```python
config = load_config()

# Validate parallelism (TP × PP × DP = num_gpus)
is_valid = config.validate_parallelism(tp=4, pp=1, dp=2, num_gpus=8)
# Returns: True (4 × 1 × 2 = 8 ✓)

# Calculate gradient accumulation steps
# Formula: GBS = MBS × DP × GA_steps
ga_steps = config.calculate_gradient_accumulation_steps(
    global_batch_size=128,
    micro_batch_size=1,
    data_parallel_size=2
)
# Returns: 64 (128 / (1 × 2) = 64)
```

### Configuration Benefits

- ✅ **Single source of truth** - all settings in one place
- ✅ **Two methodologies** - maximum performance or identical config
- ✅ **Easy switching** - change methodology with one line
- ✅ **Validation** - built-in consistency checks
- ✅ **Documentation** - self-documenting configuration
- ✅ **Reproducibility** - share exact experiment settings
- ✅ **Fully integrated** - all training scripts use config automatically
- ✅ **Works everywhere** - Python scripts and Shell scripts both use config.yaml

### Integrated Scripts

All scripts now automatically load settings from `config.yaml`:

| Type | Script | Status | Loads |
|------|--------|--------|-------|
| **Python Training** | `pretrain_llama.py` | ✅ | Parallelism, batch size, steps, optimizations |
| **Python Training** | `pretrain_qwen.py` | ✅ | Parallelism, batch size, steps, optimizations |
| **Shell Training** | `run_primus_llama.sh` | ✅ | Batch size, steps, GPU count, paths |
| **Shell Training** | `run_primus_qwen.sh` | ✅ | Batch size, steps, GPU count, paths |
| **Shell Training** | `run_primus_all.sh` | ✅ | Calls other scripts (inherits config) |
| **Python Metrics** | `enhanced_metrics.py` | ✅ | Hardware specs, costs, model params |

**Result:** Edit `config.yaml` once → all scripts update automatically!

### Shell Script Integration

Shell scripts (`run_primus_*.sh`) also load configuration from `config.yaml`:

```bash
# Shell scripts load config automatically
./run_primus_llama.sh

# Output shows config values:
# 📋 Loading configuration from config.yaml...
# ✓ Configuration loaded
# 📊 Training Iterations: 10      (from config)
# 📦 Global Batch Size: 128       (from config)
# 📏 Sequence Length: 2048        (from config)
```

**How it works:**
1. Script calls `config_to_shell.py` to export config as environment variables
2. Variables like `CONFIG_TRAIN_ITERS` are available
3. Script uses config values with fallbacks (env var → config → default)
4. Can still override with environment variables

**Manual override:**
```bash
# Override config value temporarily
TRAIN_ITERS=5 ./run_primus_llama.sh
```

**Testing:**
```bash
# View exported variables
python3 config_to_shell.py

# Test script integration
./run_primus_llama.sh 2>&1 | head -20
```

---

## 📖 Configuration Reference

### Core Concepts

#### 1. Single Source of Truth

**One file (`config.yaml`) contains:**
- Model definitions (Llama, Qwen)
- Hardware specs (H100, MI300X)
- Training parameters (batch sizes, steps)
- Parallelism strategies (TP, PP, DP)
- Platform optimizations (FP8, BF16)
- Benchmarking settings (metrics, costs)

#### 2. Two Methodologies

##### Maximum Performance (Default)
Each platform uses its optimal configuration.

**Why different configs?**
- **AMD MI300X**: 192GB memory → can fit full model (TP=1) → less communication overhead
- **NVIDIA H100**: 80GB memory → needs model splitting (TP=4) → more communication

**Use for:** Production deployment, cost analysis, real-world performance

**Example:**
```yaml
parallelism:
  maximum_performance:
    llama:
      nvidia: { TP: 4, PP: 1, DP: 2 }  # H100: needs TP=4
      amd:    { TP: 1, PP: 1, DP: 8 }  # MI300X: can use TP=1
```

##### Identical Configuration
Both platforms use the same settings.

**Use for:** Academic research, hardware-only comparison

**Example:**
```yaml
parallelism:
  identical_config:
    llama:
      nvidia: { TP: 4, PP: 1, DP: 2 }  # Same
      amd:    { TP: 4, PP: 1, DP: 2 }  # Same
```

**Switch by editing config.yaml:**
```yaml
experiment:
  methodology: "identical_config"  # or "maximum_performance"
```

### Comparison Methodologies

#### Maximum Performance Results

| Metric | NVIDIA H100 | AMD MI300X | Advantage |
|--------|-------------|------------|-----------|
| Tokens/s/GPU | ~5,115 | ~13,363 | 6.34x AMD |
| Memory/GPU | 22 GB | 118 GB | 5.3x AMD |
| Configuration | TP=4, FP8 | TP=1, BF16 | Different |

**Valid for:** Production decisions, cost analysis, real-world planning

#### Identical Configuration Results

Both platforms use TP=4, same precision, same batch sizes.

**Expected:** 2-3x AMD advantage (isolates hardware differences)

**Valid for:** Academic hardware studies, framework maturity comparisons

### Parallelism Explained

#### The Formula

```
TP × PP × DP = num_gpus
```

Must always be satisfied!

#### What Each Means

**Tensor Parallel (TP):** Splits individual layers across GPUs
- Example: TP=4 → each layer split into 4 pieces across 4 GPUs
- Trade-off: ✅ Reduces memory per GPU | ❌ Requires all-reduce per layer

**Pipeline Parallel (PP):** Splits model vertically (layers across GPUs)
- Example: PP=2 → first half of layers on GPU group 1, second half on group 2
- Trade-off: ✅ Less communication than TP | ❌ Pipeline bubbles (idle time)

**Data Parallel (DP):** Replicates full model, splits data batches
- Example: DP=8 → 8 copies of model, each processes different data
- Trade-off: ✅ Perfect scaling | ❌ Requires memory for full model per GPU

#### Examples (8 GPUs)

| TP | PP | DP | Product | Valid? | Use Case |
|----|----|----|---------|--------|----------|
| 1  | 1  | 8  | 8       | ✓      | Maximum DP (AMD with large memory) |
| 4  | 1  | 2  | 8       | ✓      | Balanced (NVIDIA Llama) |
| 4  | 2  | 1  | 8       | ✓      | No DP (NVIDIA Qwen) |
| 8  | 1  | 1  | 8       | ✓      | Maximum TP (no DP) |
| 2  | 2  | 2  | 8       | ✓      | Balanced mix |
| 3  | 3  | 1  | 9       | ❌     | 3×3×1 = 9 ≠ 8 |

#### Gradient Accumulation

**Formula:**
```
global_batch_size = micro_batch_size × DP × gradient_accumulation_steps
```

**Example:**
```
128 = 1 × 2 × 64

Where:
- micro_batch_size = 1 (per GPU, per step)
- DP = 2 (2 data parallel groups)
- gradient_accumulation_steps = 64
- global_batch_size = 128
```

This means:
- Each GPU processes 1 sample at a time (MBS=1)
- 2 GPU groups process data in parallel (DP=2)
- Each group accumulates gradients over 64 steps
- Total effective batch size = 128

#### Choosing TP/PP/DP

**High memory (AMD MI300X: 192GB):**
- Prefer TP=1 (less communication)
- Use more DP (better scaling)
- Example: TP=1, PP=1, DP=8

**Limited memory (NVIDIA H100: 80GB):**
- Need higher TP (model splitting)
- Less DP available
- Example: TP=4, PP=1, DP=2

**Very large models:**
- May need PP (pipeline stages)
- Example: TP=4, PP=2, DP=1

### Platform Optimizations

#### NVIDIA H100

```yaml
platform_optimizations:
  nvidia:
    precision: "fp8"              # H100 excels at FP8
    fp8_hybrid: true
    fp8_param: true
    cuda_alloc_conf: "expandable_segments:True"
    activation_checkpointing: true
    gradient_checkpointing: false
```

**Why FP8?**
- H100 has 989 TFLOPs in FP8 (vs 494 TFLOPs in BF16)
- 2x throughput improvement
- Minimal accuracy loss with hybrid mode

#### AMD MI300X

```yaml
platform_optimizations:
  amd:
    precision: "bf16"             # BF16 is well-supported
    fp8_hybrid: false             # FP8 support may vary
    fp8_param: false
    activation_checkpointing: false  # Less needed with 192GB
    gradient_checkpointing: false
```

**Why BF16?**
- Excellent BF16 support: 653 TFLOPs
- 192GB memory reduces need for activation checkpointing
- More stable than FP8 on current ROCm versions

### All Parameters

#### Experiment

```yaml
experiment:
  name: "tprimat_benchmark"           # Experiment name
  description: "..."                  # Description
  version: "2.0"                      # Config version
  methodology: "maximum_performance"  # or "identical_config"
```

#### Hardware

```yaml
hardware:
  platforms:
    nvidia:
      gpu_model: "H100"               # GPU model name
      memory_per_gpu_gb: 80           # Memory per GPU
      num_gpus: 8                     # Number of GPUs
      software_stack: "cuda"          # Software stack
      framework: "nemo"               # Framework name
```

#### Models

```yaml
models:
  llama:
    name: "llama3.1_8b"               # Model identifier
    full_name: "Llama 3.1 8B"         # Display name
    num_parameters: 8.0e9             # Parameter count
    num_layers: 32                    # Number of layers
    hidden_size: 4096                 # Hidden dimension
    num_attention_heads: 32           # Attention heads
    primus_config: "path/to/config"   # Primus config path
    nemo_recipe: "recipe_name"        # NeMo recipe name
```

#### Training

```yaml
training:
  data:
    micro_batch_size: 1               # Per-GPU batch per step
    global_batch_size: 128            # Total batch size
    seq_length: 2048                  # Sequence length
  
  duration:
    max_steps: 10                     # Training steps
    train_iters: 10                   # Alternative name
  
  optimizer:
    type: "adam"                      # Optimizer type
    learning_rate: 3.0e-4             # Learning rate (shared across platforms)
    warmup_steps: 10                  # LR warmup steps
    weight_decay: 0.1                 # Weight decay
    beta1: 0.9                        # Adam beta1
    beta2: 0.95                       # Adam beta2
  
  precision:
    default: "bf16"                   # Default precision
    fp8_hybrid: false                 # FP8 hybrid mode
    fp8_param: false                  # FP8 parameters
```

#### Parallelism

```yaml
parallelism:
  maximum_performance:
    llama:
      nvidia:
        tensor_model_parallel_size: 4       # TP
        pipeline_model_parallel_size: 1     # PP
        data_parallel_size: 2               # DP
        gradient_accumulation_steps: 64     # GA steps
```

#### Benchmarking

```yaml
benchmarking:
  output:
    directory: "./output"
    filename_format: "benchmark_{software_stack}_{model}.json"
    log_format: "training_{model}.log"
  
  metrics:
    performance:
      - "tokens_per_second_per_gpu"
      - "avg_step_time_seconds"
    memory:
      - "avg_memory_allocated_gb"
      - "peak_memory_allocated_gb"
  
  enhanced_metrics:
    cloud_costs:
      nvidia_h100_8gpu_per_hour: 32.0     # $/hour
      amd_mi300x_8gpu_per_hour: 24.0      # $/hour
    
    hardware_specs:
      nvidia_h100:
        peak_tflops_fp8: 989.0e12         # TFLOPs
        tdp_watts: 700                     # Power
```

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

**Note:** All training scripts (Python and Shell) now automatically load settings from `config.yaml`:
- **Python:** `pretrain_llama.py`, `pretrain_qwen.py`
- **Shell:** `run_primus_llama.sh`, `run_primus_qwen.sh`

To change parallelism, batch sizes, or other parameters, simply edit `config.yaml` and re-run the benchmarks.

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
python3 compare.py
# → Creates: compare.png with all metrics
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

## 📈 Enhanced Metrics

### 1. Cost-Normalized Metrics 💰

```python
# Tokens per Dollar-Hour
tokens_per_dollar_hour = (tokens_per_second * 3600) / cost_per_hour

# Cost to Train 1 Trillion Tokens
cost_per_trillion_tokens = (1e12 / tokens_per_second) * (cost_per_hour / 3600)
```

**Cloud Pricing (from config.yaml):**
- NVIDIA H100 (8 GPUs): $32/hr
- AMD MI300X (8 GPUs): $24/hr

### 2. Model FLOPs Utilization (MFU) 📊

Industry-standard metric for training efficiency:

```python
# Peak theoretical FLOPs (from config.yaml)
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
```

### 4. Power Efficiency ⚡

```python
# Tokens per watt-hour (using TDP from config.yaml)
tokens_per_watt_hour = tokens_per_second * 3600 / (tdp * num_gpus)

# TDP values from config:
# H100: 700W, MI300X: 750W
```

### 5. Training Time Estimates 🎯

```python
# Time to train 1 Trillion tokens
time_to_1T_tokens_hours = (1e12 / tokens_per_second) / 3600

# Full Llama 3.1 8B training (≈ 15T tokens)
time_to_full_training_days = (15e12 / tokens_per_second) / (3600 * 24)
```

### Generate Comparison Report

```bash
python3 compare.py
```

**Output includes:**
- Visual charts (compare.png)
- Throughput comparison
- Memory efficiency
- Cost per trillion tokens
- MFU comparison
- Training time estimates
- Power efficiency

---

## 🔧 Troubleshooting

### CUDA Out of Memory Error

**Quick Fix:**
```bash
./fix_gpu_memory.sh
```

**Manual Solutions:**

```bash
# Kill processes
nvidia-smi | grep python | awk '{print $5}' | xargs -r kill -9

# Clear cache
python3 -c "import torch; torch.cuda.empty_cache()"

# Set memory allocator (already in pretrain scripts)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
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

### Configuration Issues

**Parallelism validation fails:**
```python
# Check that TP × PP × DP = num_gpus
config = load_config()
p = config.get_parallelism("llama", "nvidia")
product = p['tensor_model_parallel_size'] * \
          p['pipeline_model_parallel_size'] * \
          p['data_parallel_size']
print(f"Product: {product} (should be 8)")
# If mismatch, edit config.yaml
```

**Batch size mismatch:**
```python
# Check: GBS = MBS × DP × GA_steps
gbs = config.training.data.global_batch_size
mbs = config.training.data.micro_batch_size
dp = parallelism['data_parallel_size']
ga = parallelism['gradient_accumulation_steps']
calculated = mbs * dp * ga
print(f"GBS={gbs}, calculated={calculated} (should match)")
# If mismatch, update gradient_accumulation_steps in config.yaml
```

### Platform Detection Issues

If you see `❌ No AMD benchmark results found!` but you have ROCm files:

**Solution:** Make sure your JSON files have `"software_stack": "rocm"` or `"cuda"`.

---

## 💡 Best Practices

### For Accurate Benchmarks

1. ✅ **Run multiple times** (3-5 runs) for statistical significance
2. ✅ **Close other GPU applications** during benchmarking
3. ✅ **Let warmup complete** (first step is automatically excluded)
4. ✅ **Clean GPU memory** before each run (`./fix_gpu_memory.sh`)
5. ✅ **Document your setup** (GPU model, driver versions, configs)
6. ✅ **Monitor during run** (`nvidia-smi` or `rocm-smi`)

### For Configuration

1. ✅ **Always validate** after editing config.yaml
2. ✅ **Check batch size math** after changing DP
3. ✅ **Print summary** before running expensive training
4. ✅ **Version control** configuration changes
5. ✅ **Document changes** with comments in YAML

### For Fair Comparisons

1. ✅ **Decide on methodology**: Maximum performance or identical configuration
2. ✅ **Document configurations**: TP, PP, precision, batch sizes
3. ✅ **Use same workload**: Same model, same data, same steps
4. ✅ **Check MFU**: Both platforms should have reasonable utilization
5. ✅ **Report context**: Explain why configurations differ (if they do)

---

## 🚀 Advanced Topics

### Changing Configuration Settings

**All settings in one place:**

```yaml
# Edit config.yaml
training:
  data:
    global_batch_size: 256      # Changed from 128
  duration:
    max_steps: 20               # Changed from 10

experiment:
  methodology: "identical_config"  # Changed from "maximum_performance"
```

```bash
# Run - automatically uses new settings
./benchmark.py --model llama
./run_primus_llama.sh
```

**All scripts (Python and Shell) automatically use the new values!**

### Switching Methodologies

```bash
# 1. Maximum performance (each platform optimized)
# Edit config.yaml: methodology: "maximum_performance"
./benchmark.py --model llama

# 2. Identical configuration (fair hardware comparison)
# Edit config.yaml: methodology: "identical_config"
./benchmark.py --model llama

# 3. Compare both
python3 compare.py
```

### Adding New Models

**1. Add to config.yaml:**
```yaml
models:
  llama_70b:
    name: "llama3.1_70b"
    num_parameters: 70.0e9
    primus_config: "path/to/config.yaml"
    nemo_recipe: "llama31_70b.pretrain_recipe"
```

**2. Add parallelism configs:**
```yaml
parallelism:
  maximum_performance:
    llama_70b:
      nvidia:
        tensor_model_parallel_size: 8
        pipeline_model_parallel_size: 1
        data_parallel_size: 1
```

**3. Use in script:**
```python
config = load_config()
parallelism = config.get_parallelism("llama_70b", "nvidia")
```

### Custom Validation

```python
config = load_config()

def validate_memory_estimate(model, platform):
    """Estimate if model fits in memory."""
    model_config = config.get_model_config(model)
    hw_config = config.get_hardware_config(platform)
    parallelism = config.get_parallelism(model, platform)
    
    params = model_config['num_parameters']
    memory_per_gpu = hw_config['memory_per_gpu_gb']
    tp = parallelism['tensor_model_parallel_size']
    
    # Model size per GPU (with TP)
    model_size_gb = (params * 2) / (1024**3) / tp
    
    # Total memory needed (rough estimate)
    total_needed = model_size_gb * 6  # 6x for optimizer + gradients
    
    if total_needed > memory_per_gpu * 0.9:
        print(f"⚠️  Warning: Need ~{total_needed:.1f}GB, have {memory_per_gpu}GB")
        return False
    
    return True
```

### Environment Variables

Use `${VAR:-default}` syntax in config.yaml:

```yaml
paths:
  primus:
    installation: "${PRIMUS_PATH:-/workspace/Primus}"
```

Then:
```bash
export PRIMUS_PATH=/custom/path
python3 config_loader.py  # Will use /custom/path
```

---

## 📋 Reference

### Quick Commands

| Command | Purpose |
|---------|---------|
| `./benchmark.py` | Run all models on current platform |
| `./benchmark.py --model llama` | Run single model |
| `./benchmark.py --runs 3` | Run 3 times per model |
| `python3 compare.py` | Generate comparison plot |
| `python3 config_loader.py` | View configuration |
| `python3 config_to_shell.py` | Export config as shell variables |
| `./run_primus_llama.sh` | Run Llama on AMD/Primus |
| `./run_primus_qwen.sh` | Run Qwen on AMD/Primus |
| `./run_primus_all.sh` | Run all Primus models |
| `./fix_gpu_memory.sh` | Clean GPU memory |

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
└── training_*.log                 # Training logs (Primus)

compare.png                       # Visual comparison
config.yaml                        # Configuration file
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

### Formulas

#### Parallelism Constraint
```
TP × PP × DP = num_gpus
```

#### Batch Size Calculation
```
global_batch_size = micro_batch_size × DP × gradient_accumulation_steps
```

#### Memory Estimate (Rough)
```
memory_needed ≈ (params × 2 bytes / TP) × 6
```

#### Model FLOPs Utilization
```
MFU = (achieved_FLOPs / peak_FLOPs) × 100%
```

### File Descriptions

| File | Purpose |
|------|---------|
| **Configuration** | |
| `config.yaml` | Main configuration file - edit this! |
| `config_loader.py` | Python module to load config |
| `config_to_shell.py` | Helper to export config for shell scripts |
| `example_config_usage.py` | Usage examples |
| **Benchmarking** | |
| `benchmark.py` | Main benchmark entrypoint |
| `utils.py` | Core benchmarking framework |
| **Python Training (NeMo/NVIDIA)** | |
| `pretrain_llama.py` | NeMo Llama training (uses config.yaml) |
| `pretrain_qwen.py` | NeMo Qwen training (uses config.yaml) |
| **Shell Training (Primus/AMD)** | |
| `run_primus_llama.sh` | Primus Llama training (uses config.yaml) |
| `run_primus_qwen.sh` | Primus Qwen training (uses config.yaml) |
| `run_primus_all.sh` | Run all Primus models |
| **Analysis** | |
| `extract_primus_metrics.py` | Extract from Primus logs |
| `compare.py` | Main comparison script (plots + enhanced metrics) |
| `enhanced_metrics.py` | MFU, cost, power metrics calculator |

---

## 🎉 Summary

TensorPrimat provides:

- ✅ **Unified configuration** - single YAML for all settings
- ✅ **Automatic platform detection** - works on NVIDIA and AMD
- ✅ **Two methodologies** - max performance or identical config
- ✅ **Smart log discovery** - finds logs automatically
- ✅ **Comprehensive metrics** - tokens/sec/GPU, memory, MFU, cost
- ✅ **Easy Python API** - dot notation and helper methods
- ✅ **Built-in validation** - consistency checks
- ✅ **Enhanced metrics** - MFU, cost, power, training time
- ✅ **ROCm compatibility** - seamless AMD support via HIP
- ✅ **Beautiful CLI output** - colored, formatted, clear

### Get Started

```bash
# 1. Install dependencies
pip install pyyaml matplotlib numpy

# 2. Review configuration
python3 config_loader.py

# 3. Run benchmark
./benchmark.py

# 4. Compare results
python3 compare.py
```

---

## 🎯 Configuration Summary

### All Scripts Integrated

**Python Scripts:**
- `pretrain_llama.py` - Loads config, applies parallelism/batch size/optimizations
- `pretrain_qwen.py` - Loads config, applies parallelism/batch size/optimizations
- `enhanced_metrics.py` - Loads hardware specs and costs from config

**Shell Scripts:**
- `run_primus_llama.sh` - Loads config via `config_to_shell.py`
- `run_primus_qwen.sh` - Loads config via `config_to_shell.py`
- `run_primus_all.sh` - Calls other scripts (inherits config)

### Configuration Files

- **`config.yaml`** - Main configuration (EDIT THIS!)
- **`config_loader.py`** - Python module (loads YAML)
- **`config_to_shell.py`** - Shell helper (exports env vars)
- **`example_config_usage.py`** - Usage examples

### Quick Workflow

1. **Edit:** `vim config.yaml` (change batch size, steps, methodology, etc.)
2. **Run:** `./benchmark.py` or `./run_primus_llama.sh`
3. **Done:** All scripts automatically use new settings!

### Key Features

✅ **Single source of truth** - One file controls everything  
✅ **Fully integrated** - All scripts (Python & Shell) use config  
✅ **Two methodologies** - Max performance or identical config  
✅ **Platform detection** - Auto-detects AMD/NVIDIA  
✅ **Validation** - Built-in consistency checks  
✅ **Flexible** - Can override with env vars  
✅ **Documented** - Self-documenting YAML with comments  

### Example Changes

```yaml
# config.yaml - Edit these values

# Change batch size globally
training:
  data:
    global_batch_size: 256  # All scripts use 256

# Change training duration
training:
  duration:
    max_steps: 20  # All scripts run 20 steps

# Switch methodology
experiment:
  methodology: "identical_config"  # Fair comparison
```

**Result:** All scripts automatically updated! 🎉

---

**Version**: 2.0  
**Compatible with**: NeMo 24.x+, PyTorch 2.x+  
**Platforms**: NVIDIA CUDA, AMD ROCm  
**Python**: 3.8+

**Happy Benchmarking! 🚀**
