# TPrimat - LLM Training Benchmark Suite

**Comprehensive LLM training benchmarks comparing AMD MI300X vs NVIDIA H100 GPUs across multiple frameworks and parallelism strategies.**

[![Platform](https://img.shields.io/badge/Platform-AMD%20MI300X%20%7C%20NVIDIA%20H100-blue)]()
[![Models](https://img.shields.io/badge/Models-Llama%203.1%208B%20%7C%20Qwen%202.5%207B-green)]()
[![Frameworks](https://img.shields.io/badge/Frameworks-6%20AMD%20%7C%204%20NVIDIA-orange)]()
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)]()

---

## 📋 Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Framework Matrix](#framework-matrix)
- [Installation](#installation)
- [Usage](#usage)
- [Script Naming Convention](#script-naming-convention)
- [Configuration](#configuration)
- [Output & Results](#output--results)
- [Comparison & Analysis](#comparison--analysis)
- [Docker Deployment](#docker-deployment)
- [Troubleshooting](#troubleshooting)
- [Advanced Topics](#advanced-topics)

---

## Overview

TPrimat is a comprehensive benchmarking suite that enables **apples-to-apples comparisons** of LLM training performance across:

- **2 GPU Platforms**: AMD MI300X (192GB) vs NVIDIA H100 (80GB)
- **6+ Frameworks per Platform**: Multiple training approaches
- **2 Models**: Llama 3.1 8B and Qwen 2.5 7B
- **Multiple Parallelism Strategies**: TP, PP, DP, ZeRO, FSDP

### Key Features

✨ **Framework Parity**: 6 frameworks on AMD, 4+ on NVIDIA  
✨ **Unified Output Format**: Consistent JSON metrics across all frameworks  
✨ **Automated Benchmarking**: One command runs all frameworks  
✨ **Fair Comparisons**: Identical configs for accurate hardware comparison  
✨ **Cross-Platform Scripts**: Some frameworks work on both AMD and NVIDIA  

### Supported Hardware

| Platform | GPU | Memory | Software Stack | Frameworks |
|----------|-----|--------|----------------|------------|
| **NVIDIA** | H100 80GB | 80GB HBM3 | CUDA 12.8 | NeMo, Megatron, DeepSpeed, Transformers |
| **AMD** | MI300X | 192GB HBM3 | ROCm 6.x | Primus, NeMo, DeepSpeed, Transformers, FSDP, Megatron-DS |

---

## Quick Start

### 🚀 Run All Frameworks

```bash
# On NVIDIA H100
./train_nvd.sh

# On AMD MI300X
./train_amd.sh

# In Docker
./train_docker_nvd.sh ./train_nvd.sh  # NVIDIA
./train_docker_amd.sh ./train_amd.sh  # AMD
```

**Result**: Multiple `train_*.json` files in `output/` directory

### ⚡ Run Single Framework

```bash
# Platform-agnostic (auto-detects AMD/NVIDIA)
python3 train_all_nemo.py llama
python3 train_all_nemo.py qwen
python3 train_all_tran.py llama
torchrun --nproc_per_node=8 train_all_mega.py llama

# Configurable scripts (use config.yaml settings)
python3 train_all_nemo_cfg.py llama              # NeMo with config
./train_amd_prim_llama_cfg.sh                    # Primus with config
./train_amd_prim_qwen_cfg.sh                     # Primus Qwen with config

# Platform-specific examples
deepspeed --num_gpus=8 train_nvd_deep.py llama  # NVIDIA DeepSpeed
./train_amd_deep_llama.sh                        # AMD DeepSpeed
./train_amd_prim_llama.sh                        # AMD Primus
./train_amd_fsdp_llama.sh                        # AMD FSDP
```

### 📊 Compare Results

```bash
# Generate comparison plots and analysis
python3 compare.py

# AMD-only comparison
python3 compare.py --platform amd

# NVIDIA-only comparison
python3 compare.py --platform nvidia
```

---

## Framework Matrix

### NVIDIA Frameworks

| Framework | Identifier | Script | Configurable | Description | Parallelism |
|-----------|-----------|--------|--------------|-------------|-------------|
| **NeMo** | `nemo` | `train_all_nemo.py` | `train_all_nemo_cfg.py` | Cross-platform NeMo framework | TP/PP/DP |
| **Megatron** | `mega` | `train_nvd_mega.py` | - | Model parallelism baseline | TP/PP/DP |
| **DeepSpeed** | `deep` | `train_nvd_deep.py` | - | Memory-efficient ZeRO-3 | ZeRO-3 |
| **Transformers** | `tran` | `train_all_tran.py` | - | HuggingFace portable baseline | DP |

### AMD Frameworks

| Framework | Identifier | Script | Configurable | Description | Parallelism |
|-----------|-----------|--------|--------------|-------------|-------------|
| **Primus** | `prim` | `train_amd_prim_llama.sh` | `train_amd_prim_llama_cfg.sh` | AMD-optimized (best perf) | TP/PP/DP |
| **NeMo** | `nemo` | `train_amd_nemo_llama.sh` | `train_all_nemo_cfg.py` | Cross-platform comparison | TP/PP/DP |
| **Transformers** | `tran` | `train_amd_tran_llama.sh` | - | HuggingFace portable | DP |
| **DeepSpeed** | `deep` | `train_amd_deep_llama.sh` | - | Memory-efficient ZeRO-3 | ZeRO-3 |
| **FSDP** | `fsdp` | `train_amd_fsdp_llama.sh` | - | PyTorch native sharding | Full Shard |
| **Megatron-DS** | `mgds` | `train_amd_mgds_llama.sh` | - | Hybrid parallelism (optional) | TP/PP/ZeRO |

### Platform-Agnostic Frameworks

These scripts work on **both** NVIDIA and AMD (auto-detect platform):

- `train_all_nemo.py` - NVIDIA NeMo (standard)
- `train_all_nemo_cfg.py` - NVIDIA NeMo (configurable via `config.yaml`)
- `train_all_tran.py` - HuggingFace Transformers
- `train_all_fsdp.py` - PyTorch FSDP
- `train_all_mega.py` - Megatron-LM
- `train_all_mgds.py` - Megatron-DeepSpeed

---

## Installation

### Prerequisites

- **NVIDIA**: H100 GPUs, CUDA 12.8+, Driver 550+
- **AMD**: MI300X GPUs, ROCm 6.x+
- Python 3.10+, Docker (optional)

### Option 1: Docker (Recommended)

```bash
# NVIDIA
docker build -f nvd.Dockerfile -t primat:nvd .
./train_docker_nvd.sh

# AMD
docker build -f amd.Dockerfile -t primat:amd .
./train_docker_amd.sh
```

### Option 2: Local Installation

```bash
# NVIDIA
pip install -r nvd-requirements.txt

# AMD
pip install -r amd-requirements.txt

# Set environment variables
export HF_TOKEN="your_huggingface_token"
export PRIMUS_PATH="/workspace/Primus"  # For AMD Primus
```

---

## Usage

### Basic Training

```bash
# Run all frameworks
./train_nvd.sh  # NVIDIA
./train_amd.sh  # AMD

# Run specific framework + model
python3 train_all_nemo.py llama    # NeMo + Llama (auto-detects platform)
./train_amd_nemo_llama.sh                # AMD NeMo + Llama
./train_amd_prim_qwen.sh                     # AMD Primus + Qwen
```

### Selective Framework Execution

```bash
# AMD: Run only specific frameworks
FRAMEWORKS="nemo,transformers,deepspeed" ./train_amd.sh

# AMD: Run only FSDP
FRAMEWORKS="fsdp" ./train_amd.sh
```

### Configuration

Training parameters are in `config.yaml`:

```yaml
training:
  data:
    micro_batch_size: 1
    global_batch_size: 128
    seq_length: 2048
  duration:
    max_steps: 500
  optimizer:
    learning_rate: 3.0e-4
    warmup_steps: 50

parallelism:
  maximum_performance:
    llama:
      nvidia:
        tensor_model_parallel_size: 4
        pipeline_model_parallel_size: 1
        data_parallel_size: 2
      amd:
        tensor_model_parallel_size: 1
        pipeline_model_parallel_size: 1
        data_parallel_size: 8
```

### Environment Variables

```bash
# Number of GPUs
export CONFIG_AMD_NUM_GPUS=8
export CONFIG_NVD_NUM_GPUS=8

# Output directory
export CONFIG_OUTPUT_DIR=./output

# Framework paths
export PRIMUS_PATH=/workspace/Primus
export MEGATRON_DEEPSPEED_PATH=/workspace/Megatron-DeepSpeed

# HuggingFace token
export HF_TOKEN="your_token_here"
```

---

## Script Naming Convention

All scripts follow a consistent naming pattern:

### Pattern: `run_<platform>_<framework>_<model>.{sh,py}`

| Prefix | Platform | Example |
|--------|----------|---------|
| `nvd_` | NVIDIA platform | `train_nvd_mega.py`, `train_nvd_deep.py` |
| `amd_` | AMD-specific | `train_amd_nemo_llama.sh` |
| `prim_` | AMD Primus | `train_amd_prim_llama.sh` |
| (none) | Platform-agnostic | `train_all_tran.py` |

### Examples

```bash
# NVIDIA-specific (nvd_ prefix)
train_nvd.sh       # Run all NVIDIA frameworks
train_all_nemo.py        # NeMo (platform-agnostic)
train_nvd_mega.py        # Megatron-LM
train_nvd_deep.py      # DeepSpeed
train_nvd_docker.sh          # NVIDIA Docker launcher

# AMD-specific (amd_ prefix)
train_amd.sh       # Run all AMD frameworks
train_amd_nemo_llama.sh      # AMD NeMo + Llama
train_amd_tran_llama.sh      # AMD Transformers + Llama
train_amd_deep_llama.sh      # AMD DeepSpeed + Llama
train_amd_fsdp_llama.sh      # AMD FSDP + Llama
train_amd_docker.sh          # AMD Docker launcher

# Primus (AMD-optimized)
train_prim.sh            # Run all Primus training
train_amd_prim_llama.sh          # Primus + Llama
train_amd_prim_qwen.sh           # Primus + Qwen

# Platform-agnostic (works on both)
train_all_tran.py          # HuggingFace Transformers
train_all_fsdp.py          # PyTorch FSDP
train_all_mgds.py        # Megatron-DeepSpeed
```

### Benefits

✓ **Clear identification** - Know the platform at a glance  
✓ **Consistent grouping** - Scripts sort alphabetically by platform  
✓ **Easy filtering** - Use globs like `run_nvd_*.py` or `run_amd_*.sh`  
✓ **Self-documenting** - Filename tells you what it does  

---

## Output & Results

### Output File Format

All frameworks produce unified JSON output:

```
output/
├── train_nvd_nemo_llama.json  # NeMo + Llama 3.1 8B
├── train_nvd_nemo_qwen.json   # NeMo + Qwen 2.5 7B
├── train_nvd_mega_llama.json  # Megatron + Llama
├── train_nvd_deep_llama.json  # DeepSpeed + Llama
├── train_nvd_tran_llama.json  # Transformers + Llama
├── train_fsdp_llama.json      # FSDP + Llama
├── train_amd_prim_llama.json      # Primus + Llama (AMD)
└── train_mgds_llama.json    # Megatron-DS + Llama
```

### JSON Structure

```json
{
  "platform": "amd",
  "gpu_info": {
    "device_name": "AMD Instinct MI300X",
    "device_count": 8,
    "total_memory_gb": 192.0
  },
  "training_config": {
    "max_steps": 500,
    "global_batch_size": 128,
    "micro_batch_size": 1,
    "sequence_length": 2048,
    "num_gpus": 8,
    "parallel_strategy": "data"
  },
  "performance_metrics": {
    "avg_step_time_seconds": 0.45,
    "tokens_per_second": 580000,
    "tokens_per_second_per_gpu": 72500,
    "steps_per_second": 2.22
  },
  "step_times": [...],
  "loss_values": [...],
  "learning_rates": [...]
}
```

### Framework Identifiers

| Framework | Identifier in Filename |
|-----------|----------------------|
| NeMo | `nemo` |
| Megatron | `mega` |
| DeepSpeed | `deep` |
| Transformers | `tran` |
| FSDP | `fsdp` |
| Primus | `prim` |
| Megatron-DeepSpeed | `mgds` |

**Pattern**: `train_<framework>_<model>.json`

---

## Comparison & Analysis

### Generate Comparison

```bash
# Compare all results (NVIDIA vs AMD, all frameworks)
python3 compare.py

# Platform-specific
python3 compare.py --platform nvidia
python3 compare.py --platform amd

# Specific frameworks
python3 compare.py --frameworks nemo,transformers,deepspeed

# Output comparison plot
# Results saved to: output/compare.png
```

### Comparison Output

The comparison tool generates:

1. **Performance Plots** - Throughput, latency, memory usage
2. **Console Summary** - Key metrics and speedup factors
3. **Markdown Report** - Detailed analysis (optional)

**Example Output**:

```
NVIDIA H100 vs AMD MI300X - FRAMEWORK COMPARISON

Per-GPU Throughput (tokens/sec):
  NVIDIA NeMo:    68,500
  AMD Primus:     72,500
  → AMD is 1.06x faster

Average Step Time:
  NVIDIA NeMo:    0.47s
  AMD Primus:     0.45s
  → AMD is 1.04x faster
```

### Framework Performance Expectations

**AMD MI300X (8 GPUs):**

| Framework | Llama 3.1 8B | Relative Perf | Best For |
|-----------|--------------|---------------|----------|
| Primus | ~72,500 tok/s/GPU | ⭐⭐⭐⭐⭐ | Max AMD performance |
| NeMo | ~70,000 tok/s/GPU | ⭐⭐⭐⭐⭐ | Cross-platform |
| FSDP | ~65,000 tok/s/GPU | ⭐⭐⭐⭐ | Native PyTorch |
| DeepSpeed | ~62,000 tok/s/GPU | ⭐⭐⭐⭐ | Memory efficiency |
| Transformers | ~58,000 tok/s/GPU | ⭐⭐⭐ | Simple baseline |

*Actual numbers vary by configuration*

---

## Docker Deployment

### Build Images

```bash
# NVIDIA
docker build -f nvd.Dockerfile -t primat:nvd .

# AMD
docker build -f amd.Dockerfile -t primat:amd .
```

### Run Containers

```bash
# Interactive mode
./train_docker_nvd.sh          # NVIDIA
./train_docker_amd.sh          # AMD

# Run specific script
./train_docker_nvd.sh ./train_nvd.sh
./train_docker_amd.sh ./train_amd_nemo_llama.sh

# With custom token
HF_TOKEN=your_token ./train_docker_nvd.sh ./train_nvd.sh
```

### Remote Server Deployment

**Option 1: Detached Mode (Recommended)**

```bash
# Start training in background
./train_docker_amd_detached.sh ./train_amd.sh

# Check logs
docker logs -f primat

# Stop container
docker stop primat
```

**Option 2: Tmux Inside Container**

```bash
# Launch container with tmux
./train_docker_amd_tmux.sh

# Inside container
tmux new -s training
./train_amd.sh
# Detach: Ctrl+B, then D

# Reattach later
tmux attach -t training
```

### Monitoring

```bash
# GPU usage (NVIDIA)
nvidia-smi -l 1
watch -n 1 nvidia-smi

# GPU usage (AMD)
rocm-smi -l 1
watch -n 1 rocm-smi

# Inside container
docker exec -it primat nvidia-smi  # or rocm-smi
```

---

## Troubleshooting

### Common Issues

#### Out of Memory

```bash
# Try DeepSpeed (has CPU offload)
./train_amd_deep_llama.sh

# Or reduce batch size in config.yaml
training:
  data:
    micro_batch_size: 1  # Reduce this
    global_batch_size: 64  # Or this
```

#### NeMo Not Working on AMD

```bash
# Build NeMo from source with ROCm support
git clone https://github.com/NVIDIA/NeMo.git
cd NeMo
pip install -e .
```

#### DeepSpeed Compatibility

```bash
# Ensure ROCm-compatible DeepSpeed
pip install deepspeed>=0.12.0
```

#### Primus Not Found

```bash
# Set Primus path
export PRIMUS_PATH=/workspace/Primus

# Or clone if needed
git clone https://github.com/amd/Primus.git
```

#### Megatron-DeepSpeed Not Found

```bash
# Clone and set path
git clone https://github.com/microsoft/Megatron-DeepSpeed.git
export MEGATRON_DEEPSPEED_PATH=/workspace/Megatron-DeepSpeed
```

### GPU Verification

```bash
# NVIDIA
nvidia-smi
nvidia-smi --query-gpu=name,memory.total --format=csv

# AMD
rocm-smi
rocm-smi --showproductname
```

### Check Logs

```bash
# Training logs
ls -lh output/
cat output/training_llama.log

# Docker logs
docker logs primat

# Find errors
grep -i error output/training_*.log
```

---

## Advanced Topics

### Custom Parallelism Strategy

Edit `config.yaml` to add custom parallelism:

```yaml
parallelism:
  my_custom_strategy:
    llama:
      amd:
        tensor_model_parallel_size: 2
        pipeline_model_parallel_size: 1
        data_parallel_size: 4
        gradient_accumulation_steps: 32
```

Use it:

```bash
PARALLEL=my_custom_strategy ./train_amd_nemo_llama.sh
```

### Profiling (NVIDIA)

Enable Nsight Systems profiling:

```yaml
# config.yaml
experiment:
  profiling: true
```

Output: `output/profile_cuda_llama_<strategy>.nsys-rep`

View with: `nsys-ui profile_cuda_llama_*.nsys-rep`

### Real Data Training

Set data paths in `config.yaml`:

```yaml
models:
  llama:
    dataset_path: "/data/my_dataset_text_document"
    tokenizer_path: "meta-llama/Llama-3.1-8B"
```

### Cloud Deployment

The benchmark tracks cloud costs automatically:

```yaml
benchmarking:
  enhanced_metrics:
    cloud_costs:
      nvidia_h100_8gpu_per_hour: 32.0
      amd_mi300x_8gpu_per_hour: 24.0
```

Results include `cost_per_trillion_tokens` metric.

---

## Project Structure

```
tprimat/
├── README.md                       # This file
├── config.yaml                     # Configuration
├── compare.py                      # Cross-platform comparison
│
├── NVIDIA Scripts (nvd_ prefix)
│   ├── train_nvd.sh       # Run all NVIDIA frameworks
│   ├── train_all_nemo.py            # NeMo (platform-agnostic)
│   ├── train_nvd_mega.py      # Megatron
│   ├── train_nvd_deep.py      # DeepSpeed
│   └── train_nvd_docker.sh          # Docker launcher
│
├── AMD Scripts (amd_ prefix)
│   ├── train_amd.sh       # Run all AMD frameworks
│   ├── train_amd_nemo_llama.sh      # NeMo + Llama
│   ├── train_amd_tran_llama.sh      # Transformers + Llama
│   ├── train_amd_deep_llama.sh      # DeepSpeed + Llama
│   ├── train_amd_fsdp_llama.sh      # FSDP + Llama
│   ├── train_amd_mgds_llama.sh    # Megatron-DS + Llama
│   └── train_amd_docker.sh          # Docker launcher
│
├── Primus (AMD-optimized)
│   ├── train_prim.sh            # Run all Primus
│   ├── train_amd_prim_llama.sh          # Llama
│   └── train_amd_prim_qwen.sh           # Qwen
│
├── Platform-Agnostic
│   ├── train_all_tran.py          # HuggingFace Transformers
│   ├── train_all_fsdp.py          # PyTorch FSDP
│   └── train_all_mgds.py        # Megatron-DeepSpeed
│
├── Training Implementations
│   ├── train_nemo_llama.py        # NeMo Llama trainer
│   ├── train_nemo_qwen.py         # NeMo Qwen trainer
│   └── utils.py                   # Benchmark utilities
│
├── Docker
│   ├── nvd.Dockerfile             # NVIDIA image
│   ├── amd.Dockerfile             # AMD image
│   ├── nvd-requirements.txt       # NVIDIA deps
│   └── amd-requirements.txt       # AMD deps
│
└── output/                        # Results directory
    ├── train_*.json               # Benchmark results
    ├── compare.png                # Comparison plots
    └── *.log                      # Training logs
```

---

## Contributing

We welcome contributions! Areas of interest:

- Additional frameworks (JAX, PyTorch Lightning, etc.)
- More models (Llama 3.3, Mixtral, etc.)
- Performance optimizations
- Bug fixes and documentation

---

## License

[Add your license here]

---

## Citation

If you use TPrimat in your research, please cite:

```bibtex
@software{tprimat2024,
  title = {TPrimat: LLM Training Benchmark Suite},
  author = {[Your Name]},
  year = {2024},
  url = {https://github.com/yourusername/tprimat}
}
```

---

## Summary

**TPrimat provides:**

✅ **6 frameworks on AMD** (Primus, NeMo, Transformers, DeepSpeed, FSDP, Megatron-DS)  
✅ **4 frameworks on NVIDIA** (NeMo, Megatron, DeepSpeed, Transformers)  
✅ **Unified benchmarking** across platforms  
✅ **Consistent naming** (nvd_, amd_, prim_ prefixes)  
✅ **One-command execution** for all frameworks  
✅ **Automatic comparison** and analysis  
✅ **Docker deployment** with remote server support  

**Quick Commands:**

```bash
# Run everything
./train_nvd.sh  # NVIDIA
./train_amd.sh  # AMD

# Compare results
python3 compare.py

# Docker
./train_docker_nvd.sh ./train_nvd.sh
./train_docker_amd.sh ./train_amd.sh
```

**Ready to benchmark!** 🚀

---

For questions or issues, please open a GitHub issue or contact [your contact info].


---

## Script Variants

This project provides two types of scripts:

### Standard Scripts
- Simple, standalone with hardcoded defaults
- Quick to run, no configuration needed
- Examples: `train_all_nemo.py`, `train_amd_prim_llama.sh`

### Configurable Scripts (`_cfg`)
- Load settings from `config.yaml`
- Flexible parallelism strategies
- Configurable training parameters
- Examples: `train_all_nemo_cfg.py`, `train_amd_prim_llama_cfg.sh`, `train_amd_prim_qwen_cfg.sh`

**When to use configurable scripts:**
- Experimenting with different parallelism strategies (TP/PP/DP)
- Custom training hyperparameters
- Platform-specific optimizations
- Profiling and detailed benchmarking
