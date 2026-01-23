# TPrimat - LLM Training Benchmark Suite

**Tensor Parallelism AMD/NVD Training** - Comprehensive LLM training benchmarks comparing AMD MI300X vs NVD H100 GPUs with multiple parallelism strategies.

[![Platform](https://img.shields.io/badge/Platform-AMD%20MI300X%20%7C%20NVD%20H100-blue)]()
[![Models](https://img.shields.io/badge/Models-Llama%203.1%208B%20%7C%20Qwen%202.5%207B-green)]()
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)]()

---

## 📋 Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
  - [Plain Scripts](#plain-scripts-multiple-frameworks)
- [Logging & Output](#logging--output)
  - [Logging Configuration](#logging-configuration)
  - [Output File Naming Convention](#output-file-naming-convention)
  - [Troubleshooting Logs](#troubleshooting-logs)
- [Run Scripts Reference](#run-scripts-reference)
  - [Complete Run Script Matrix](#complete-run-script-matrix)
  - [Individual Framework Scripts](#individual-framework-scripts)
  - [Complete Workflow Examples](#complete-workflow-examples)
- [Installation](#installation)
- [Remote Server Deployment](#remote-server-deployment)
  - [Method 1: Docker Detached Mode](#method-1-docker-detached-mode-recommended)
  - [Method 2: Tmux Inside Container](#method-2-tmux-inside-container)
  - [Method 3: Host Tmux + Docker](#method-3-host-tmux--docker)
  - [Monitoring & Logs](#monitoring--logs)
- [Usage](#usage)
  - [Basic Commands](#basic-commands)
  - [Parallelism Strategies](#parallelism-strategies)
  - [Examples](#examples)
- [Configuration](#configuration)
  - [Parallelism Strategies Explained](#parallelism-strategies-explained)
  - [Tensor Parallelism Constraints](#tensor-parallelism-constraints)
- [Profiling](#profiling)
  - [Enabling Nsight Profiler](#enabling-nsight-profiler)
  - [Viewing Profile Results](#viewing-profile-results)
- [Troubleshooting](#troubleshooting)
- [Output Files](#output-files)
- [Advanced Topics](#advanced-topics)
- [API Reference](#api-reference)

---

## Overview

TPrimat is a comprehensive benchmarking suite for comparing LLM training performance between AMD and NVD GPUs using different parallelism strategies.

### Key Features

- **Multi-Platform**: Supports AMD MI300X (ROCm/Prim) and NVD H100 (CUDA/NeMo)
- **Multiple Models**: Llama 3.1 8B and Qwen 2.5 7B
- **5 Parallelism Strategies**: From TP=1 (fastest) to TP=4, PP=2 (memory-optimized)
- **Automatic Benchmarking**: JSON metrics + training logs + comparison plots
- **Fair Comparisons**: Identical configurations for apples-to-apples hardware comparison

### Supported Hardware

| Platform | GPU | Memory | Framework | Software Stack |
|----------|-----|--------|-----------|----------------|
| NVD | H100 80GB | 80GB HBM3 | NeMo | CUDA 12.8 |
| AMD | MI300X | 192GB HBM3 | Prim | ROCm 6.x |

### Supported Models

| Model | Parameters | Attention Heads | Valid TP Values |
|-------|-----------|-----------------|-----------------|
| Llama 3.1 8B | 8.0B | 32 | 1, 2, 4, 8, 16, 32 |
| Qwen 2.5 7B | 7.6B | 28 | 1, 2, 4, 7, 14, 28 |

---

## Quick Start

### 🚀 Fastest Start - All Frameworks

```bash
# Run all frameworks (HF, Mega-LM, NeMo) at once
./run_nvd_all_plain.sh        # On NVD
./run_amd_all_plain.sh           # On AMD

# Or in Docker
./run_docker_nvd.sh ./run_nvd_all_plain.sh
./run_docker_amd.sh ./run_amd_all_plain.sh
```

**Result:** 6 benchmark files in `output/` (takes ~30-60 min)

### ⚡ Quick Test - Single Framework

```bash
# Test one framework quickly
python3 -u run_tran_plain.py llama
torchrun --nproc_per_node=8 run_mega_plain.py llama
python3 -u run_nemo_plain.py llama
```

**Result:** 1 benchmark file in `output/` (takes ~5-10 min)

### 📊 Full Benchmark Suite

```bash
# Run all 5 parallelism configurations
./run_all_configs.sh

# Compare results
./compare_all_configs.sh
```

**Result:** `output-00/` through `output-04/` with complete benchmarks and comparison plots

### 🔧 Advanced Configuration

```bash
# Specific parallelism strategy
./benchmark.py --parallel minimal_communication

# Single model with custom output
./benchmark.py --model llama --parallel balanced --output-dir ./results

# Multiple runs for statistics
./benchmark.py --runs 3
```

### Plain Scripts (Multiple Frameworks)

TPrimat includes plain training scripts for multiple frameworks, allowing you to compare Transformers Transformers, Mega-LM, and NeMo/Prim side-by-side.

#### Run All Frameworks (Recommended)

```bash
# NVD: Run HF, Mega-LM, and NeMo
./run_nvd_all_plain.sh

# AMD: Run HF, Mega-LM, and NeMo (with ROCm)
./run_amd_all_plain.sh
```

**Generates 6 output files:**
- `output/train_tran_llama.json` & `output/train_tran_qwen.json`
- `output/train_mega_llama.json` & `output/train_mega_qwen.json`
- `output/train_nemo_llama.json` & `output/train_nemo_qwen.json`

#### Run Individual Frameworks

```bash
# Transformers Transformers (both models)
python3 -u run_tran_plain.py

# NeMo (both models)
python3 -u run_nemo_plain.py

# Run specific model only
python3 -u run_tran_plain.py llama
python3 -u run_nemo_plain.py llama
```

#### In Docker

```bash
# NVD
./run_docker_nvd.sh ./run_nvidia_all.sh
./run_docker_nvd.sh python3 -u run_tran_plain.py llama

# AMD
./run_docker_amd.sh ./run_amd_all.sh
./run_docker_amd.sh python3 -u run_nemo_plain.py qwen
```

**Hardcoded Configuration (All Frameworks):**
- Training iterations: **10** (fast testing)
- Parallelism: **TP=1, PP=1, DP=8** (minimal communication) for NeMo; **DDP** for HF
- Batch size: **64** global (micro=1, grad_accum=8)
- Sequence length: **2048** tokens
- Learning rate: **0.0003** with **1** warmup step
- Precision: **BFloat16** (HF), **FP8 Hybrid** (NeMo)
- Seed: **42**

**Note:** All scripts use `-u` flag for unbuffered output to ensure real-time log visibility.

---

## Logging & Output

### Logging Configuration

All plain scripts use proper Python logging with real-time output:

**Features:**
- ✅ Proper Python `logging` module (not print statements)
- ✅ Timestamps on all log messages
- ✅ Unbuffered output (logs appear immediately)
- ✅ Rank information for distributed training
- ✅ Framework and model identification in output files

**Log Format:**
```
2026-01-22 10:30:45,123 - __main__ - INFO - Loading model: meta-llama/Llama-3.1-8B
2026-01-22 10:30:45,456 - __main__ - INFO - CUDA devices available: 8
2026-01-22 10:31:00,789 - __main__ - INFO - Starting training...
```

**For distributed training, rank information is included:**
```
2026-01-22 10:30:45,123 - [Rank 0] - INFO - Loading model: meta-llama/Llama-3.1-8B
2026-01-22 10:30:46,456 - [Rank 0] - INFO - World size: 8, Rank: 0, Local rank: 0
```

### Output File Naming Convention

All benchmark results follow a consistent naming pattern:

```
output/train_<framework>_<model>.json
```

**Framework names:**
- `tran` - Transformers (Transformers)
- `nemo` - NVD NeMo
- `primus` - AMD Prim (AMD-specific)

**Examples:**
- `output/train_tran_llama.json` - Transformers Llama
- `output/train_nemo_llama.json` - NeMo Llama
- `output/train_prim_qwen.json` - Prim Qwen (AMD)

**Benefits:**
- Framework and model immediately identifiable from filename
- Easy to compare results across frameworks for the same model
- Simple to parse and process programmatically
- Consistent across all platforms (NVD/AMD)

### Troubleshooting Logs

#### Not seeing logs?

**Always use `-u` flag for unbuffered output:**
```bash
python3 -u run_tran_plain.py llama
```

**Or set environment variable:**
```bash
export PYTHONUNBUFFERED=1
python3 run_tran_plain.py llama
```

**In Docker, logs appear in real-time:**
```bash
./run_docker_nvd.sh python3 -u run_tran_plain.py llama
```

**Save logs while viewing:**
```bash
python3 -u run_tran_plain.py llama 2>&1 | tee output/hf_llama.log
```


Only rank 0 prints most logs (this is normal and prevents duplicate output).

---

## Run Scripts Reference

### Complete Run Script Matrix

| Script | Purpose | Frameworks | Output Files |
|--------|---------|------------|--------------|
| `run_nvidia_all.sh` | All frameworks (NVD) | HF, NeMo | 4 files |
| `run_amd_all.sh` | All frameworks (AMD) | HF, NeMo | 4 files |
| `run_tran_plain.py` | Transformers only | HF | 2 files |
| `run_nemo_plain.py` | NeMo only | NeMo | 2 files |
| `run_prim_all.sh` | Prim (AMD-optimized) | Prim | 2 files |
| `run_prim_plain.sh` | Prim direct | Prim | Logs + 2 files |

### run_nvidia_all.sh

**Purpose:** Run all training frameworks sequentially on NVD hardware.

**Usage:**
```bash
# Direct execution
./run_nvidia_all.sh

# In Docker
./run_docker_nvd.sh ./run_nvidia_all.sh
```

**What it does:**
1. ✅ Runs Transformers Transformers (Llama + Qwen)
2. ✅ Runs NeMo (Llama + Qwen)
3. ✅ Lists all generated files

**Output:** 4 JSON files in `output/`

**Time estimate:** ~20-40 minutes depending on hardware

### run_amd_all_plain.sh

**Purpose:** Run all training frameworks sequentially on AMD/ROCm hardware.

**Usage:**
```bash
# Direct execution
./run_amd_all.sh

# In Docker
./run_docker_amd.sh ./run_amd_all.sh
```

**What it does:** Same as `run_nvidia_all.sh` but with AMD-specific environment variables:
- `HSA_NO_SCRATCH_RECLAIM=1`
- `HSA_ENABLE_SDMA=1`
- `HSA_FORCE_FINE_GRAIN_PCIE=1`

### Individual Framework Scripts

#### run_tran_plain.py - Transformers Transformers

**Features:**
- Standard Transformers Trainer API
- Flash Attention 2 support (if available)
- Gradient checkpointing for memory efficiency
- Works on both NVD and AMD

**Usage:**
```bash
# Both models
python3 -u run_tran_plain.py

# Single model
python3 -u run_tran_plain.py llama
python3 -u run_tran_plain.py qwen
```

**Output:**
- `output/train_tran_llama.json`
- `output/train_tran_qwen.json`

#### run_nemo_plain.py - NVD NeMo

**Features:**
- NeMo + Mega-LM backend
- FP8 training (hybrid mode) on H100
- Selective recomputation (activation checkpointing)
- Advanced parallelism strategies (TP, PP, CP)
- Works on both NVD (native) and AMD (via NeMo-ROCm)

**Usage:**
```bash
# Both models
python3 -u run_nemo_plain.py

# Single model
python3 -u run_nemo_plain.py llama
python3 -u run_nemo_plain.py qwen
```

**Output:**
- `output/train_nemo_llama.json`
- `output/train_nemo_qwen.json`

### AMD-Specific Scripts

#### run_prim_all.sh

**Purpose:** Run Prim framework training for all models (AMD-optimized).

**Usage:**
```bash
./run_prim_all.sh
```

**What it does:**
1. Runs `run_prim_llama.sh`
2. Runs `run_prim_qwen.sh`
3. Provides summary with success/failure status

**Output:**
- `output/train_prim_llama.json`
- `output/train_prim_qwen.json`

**Note:** Prim is AMD's optimized training framework, specifically designed for ROCm/MI300X.

#### run_prim_plain.sh

**Purpose:** Run Prim training directly (lower-level access).

**Usage:**
```bash
./run_prim_plain.sh
```

**What it does:**
- Runs Llama 3.1 8B via Prim
- Runs Qwen 2.5 7B via Prim
- Uses Prim config files from `$PRIMUS_PATH`
- Extracts metrics from training logs

**Output:**
- `output/training_main_llama.log` (raw logs)
- `output/training_main_qwen.log` (raw logs)
- `output/train_prim_llama.json` (extracted metrics)
- `output/train_prim_qwen.json` (extracted metrics)

### Docker Integration

#### run_docker_nvd.sh

**Usage:**
```bash
# Interactive shell
./run_docker_nvd.sh

# Run specific script
./run_docker_nvd.sh ./run_nvidia_all.sh
./run_docker_nvd.sh python3 -u run_tran_plain.py llama

# Custom command
./run_docker_nvd.sh bash -c "python3 -u run_tran_plain.py && ls -lh output/"
```

**Features:**
- Mounts current directory to `/workspace/tprimat`
- Mounts `/data` for datasets
- GPU access via `--gpus all`
- Loads Transformers token from `secrets.env`

#### run_docker_amd.sh

**Usage:** Same syntax as `run_docker_nvd.sh`

**Features:**
- Mounts home directory and workspace
- Sets up ROCm device access (`/dev/dri`, `/dev/kfd`)
- Privileged mode for AMD hardware access
- Loads Transformers token from `secrets.env`

### Complete Workflow Examples

#### Quick Test (Single Framework)

```bash
# Test Transformers on one model
python3 -u run_tran_plain.py llama

# Check results
cat output/train_tran_llama.json
```

#### Framework Comparison

```bash
# Run all frameworks
./run_nvidia_all.sh

# Compare throughput
jq '.performance_metrics.tokens_per_second_per_gpu' output/train_*_llama.json
```

#### AMD vs NVD Comparison

```bash
# On NVD system
./run_docker_nvd.sh ./run_nvidia_all.sh

# On AMD system
./run_docker_amd.sh ./run_amd_all.sh

# Compare results
python3 compare.py --results-dir output/
```

#### Incremental Testing

```bash
# Test one model at a time
python3 -u run_tran_plain.py llama
python3 -u run_nemo_plain.py llama

# Review results before continuing
ls -lh output/train_*_llama.json

# If good, run Qwen
python3 -u run_tran_plain.py qwen
python3 -u run_nemo_plain.py qwen
```

### Environment Variables

All scripts respect these environment variables:

**Common:**
- `PYTHONUNBUFFERED=1` - Force unbuffered Python output
- `PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'` - Memory allocator
- `PYTHONHASHSEED="42"` - Reproducibility
- `HF_TOKEN` - Transformers authentication

**AMD-Specific (set by `run_amd_all.sh`):**
- `HSA_NO_SCRATCH_RECLAIM=1` - ROCm memory management
- `HSA_ENABLE_SDMA=1` - Enable SDMA engines
- `HSA_FORCE_FINE_GRAIN_PCIE=1` - PCIe optimization
- `RCCL_DEBUG=INFO` - ROCm collective communications debug

**NVD-Specific:**
- `NCCL_DEBUG=INFO` - NVD collective communications debug
- `CUDA_VISIBLE_DEVICES` - GPU selection (if needed)

---

## Installation

### Prerequisites

```bash
# Python 3.10+
python3 --version

# For NVD
nvidia-smi  # Verify CUDA GPUs

# For AMD
rocm-smi    # Verify ROCm GPUs
```

### Docker Setup (Recommended)

#### 1. Setup secrets

```bash
cp secrets.env.example secrets.env
# Edit secrets.env and add your Hugging Face token
```

The `secrets.env` file is git-ignored and will not be tracked.

#### 2. Build the Docker image

**For AMD ROCm (MI300X):**
```bash
docker build -f Dockerfile.amd -t primat:amd .
```

**For NVD CUDA (H100):**
```bash
docker build -f Dockerfile.nvd -t primat:nvd .
```

#### 3. Run the container

**For AMD:**
```bash
./run_docker_amd.sh
```

**For NVD:**
```bash
./run_docker_nvd.sh
```

The scripts automatically load your HF token from `secrets.env`.

**What's included in both Docker images:**
- **fish** - Modern shell with syntax highlighting
- **neovim** - Text editor
- **ranger** - File manager
- **zip** - Archive utility
- **tmux** - Terminal multiplexer
- Platform-specific optimizations (ROCm or CUDA)
- Profiling tools enabled

**Docker usage examples:**

```bash
# Interactive shell (AMD)
./run_docker_amd.sh

# Interactive shell (NVD)
./run_docker_nvd.sh

# Run training directly
./run_docker_amd.sh ./run_prim_all.sh

# Override HF token
HF_TOKEN=another_token ./run_docker_amd.sh
```

### Install Dependencies (Local)

```bash
pip install -r requirements.txt
```

**Dependencies:**
- PyTorch (with CUDA or ROCm)
- NeMo (NVD) or Prim (AMD)
- PyYAML, matplotlib, numpy

---

## Remote Server Deployment

Run long-running benchmarks on remote servers without interruption when SSH disconnects.

### Prerequisites

1. **Build the Docker image** (do this once):
   ```bash
   # For NVD
   docker build -f Dockerfile.nvd -t primat:nvd .
   
   # For AMD
   docker build -f Dockerfile.amd -t primat:amd .
   ```

2. **Set up secrets** (optional, if using Transformers):
   ```bash
   echo "export HF_TOKEN=your_token_here" > secrets.env
   ```

### Method 1: Docker Detached Mode ⭐ (RECOMMENDED)

**Best for:** Fully automated runs with no interaction needed. The container runs in the background and survives SSH disconnection.

#### Start the benchmark:

**For NVD:**
```bash
./run_docker_nvd_detached.sh
```

**For AMD:**
```bash
./run_docker_amd_detached.sh
```

#### Monitor progress:

```bash
# View live logs
docker logs -f primat

# Check container status
docker ps | grep primat

# Execute commands inside running container
docker exec -it primat /bin/bash

# View output files
docker exec primat ls -lh /workspace/tprimat/output/

# Stop and cleanup
docker stop primat && docker rm primat
```

**Advantages:**
- ✅ Survives SSH disconnection
- ✅ Automatic logging
- ✅ No manual interaction needed
- ✅ Easy to monitor remotely

### Method 2: Tmux Inside Container

**Best for:** Interactive debugging and monitoring with the ability to detach/reattach.

#### Start container with interactive session:

```bash
# For NVD
./run_docker_nvd_tmux.sh

# For AMD
./run_docker_amd_tmux.sh
```

#### Inside container:

```bash
# Create tmux session
tmux new -s benchmark

# Run your benchmark
./run_nemo_all.sh  # or ./run_prim_all.sh

# Detach from tmux: Press Ctrl+B, then D
# Exit container: exit
```

#### Reconnect later:

```bash
# Reattach to container
docker exec -it primat /bin/bash

# Reattach to tmux session
tmux attach -t benchmark

# List all tmux sessions
tmux ls
```

**Advantages:**
- ✅ Interactive control
- ✅ Can detach and reattach anytime
- ✅ Multiple windows/panes for monitoring
- ✅ Great for debugging

### Method 3: Host Tmux + Docker

**Best for:** Maximum flexibility and control from the host system.

#### On remote server:

```bash
# Create host tmux session
tmux new -s docker-benchmark

# Inside tmux, run container
./run_docker_nvd.sh  # or ./run_docker_amd.sh

# Inside container, run benchmark
./run_nemo_all.sh  # or ./run_prim_all.sh

# Detach from tmux: Press Ctrl+B, then D
# Now you can safely disconnect from SSH
```

#### Reconnect:

```bash
# SSH back to server
ssh your-server

# Reattach to tmux session
tmux attach -t docker-benchmark

# Or list all sessions first
tmux ls
```

**Advantages:**
- ✅ Control from host level
- ✅ Can manage multiple containers
- ✅ Easiest to debug
- ✅ No Docker-specific commands needed

### Monitoring & Logs

#### Check Container Status:
```bash
# List running containers
docker ps | grep primat

# Detailed container info
docker inspect primat

# Container resource usage
docker stats primat
```

#### View Logs:
```bash
# Docker logs
docker logs primat
docker logs -f primat --tail 100

# Application logs
docker exec primat cat /workspace/tprimat/output/training_llama.log
docker exec primat tail -f /workspace/tprimat/output/training_llama.log

# Benchmark output
docker exec primat cat /workspace/tprimat/output/train_prim_llama.json
```

#### Monitor Resources:
```bash
# GPU usage (NVD)
nvidia-smi -l 1
watch -n 1 nvidia-smi

# GPU usage (AMD)
rocm-smi -l 1
watch -n 1 rocm-smi

# Inside container
docker exec -it primat nvidia-smi  # or rocm-smi
```

### Troubleshooting Remote Runs

**Container exited unexpectedly:**
```bash
# Check exit code and logs
docker ps -a | grep primat
docker logs primat
```

**Out of memory:**
```bash
# Increase shared memory in run script
# Edit --shm-size parameter (default is 64g)
```

**Port/name already in use:**
```bash
# Remove existing container
docker rm -f primat
```

**GPU not available:**
```bash
# Verify GPU access (NVD)
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi

# Verify GPU access (AMD)
docker run --rm --device=/dev/kfd --device=/dev/dri rocm/pytorch:latest rocm-smi
```

### Best Practices

1. **Always use detached mode or tmux** for long-running jobs
2. **Monitor disk space** in `/workspace/tprimat/output/`
3. **Set up alerts** for job completion or failures
4. **Back up checkpoints** regularly to external storage
5. **Use meaningful container names** when running multiple benchmarks
6. **Check logs periodically** to catch errors early

### Quick Reference

| Method | Survives SSH Drop | Easy Monitoring | Auto-Restart | Best For |
|--------|------------------|-----------------|--------------|----------|
| Detached Docker | ✅ | ✅ | ❌ | Automated runs |
| Tmux (Container) | ✅ | ⭐⭐⭐ | ❌ | Interactive debugging |
| Tmux (Host) | ✅ | ⭐⭐ | ❌ | Development |

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
# On NVD machine
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
- **NVD (H100):**
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

| Config | Name | Llama (NVD) | Qwen (NVD) | Use Case |
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

TPrimat includes integrated NVD Nsight Systems profiling for detailed GPU performance analysis. Nsight captures kernel-level traces, memory allocations, CUDA operations, and CPU activity.

### Enabling Nsight Profiler

**1. Edit `config.yaml` to enable profiling:**

```yaml
profiling:
  enabled: true                           # Enable Nsight profiling
  trace: "cuda,nvtx,osrt,cudnn,cublas"    # What to trace
  cuda_memory_usage: true                 # Track CUDA memory allocations
  capture_range: "cudaProfilerApi"        # Capture range method
  stats: true                             # Generate summary statistics
  export_json: true                       # Export to JSON for Chrome tracing
```

**2. Run your benchmark:**

```bash
python3 train_llama.py   # Automatically wraps with nsys when profiling enabled
# or
./benchmark.py --parallel minimal_communication
```

**3. Find profiler output in your output directory:**

```
output/
├── profile_cuda_llama_balanced.nsys-rep    # Nsight profile (binary)
├── train_nemo_llama.json                   # Training metrics
```

### Profiler Configuration

**Trace Options:**

- `cuda`: CUDA API calls and kernel launches
- `nvtx`: NVD Tools Extension markers
- `osrt`: OS runtime libraries
- `cudnn`: cuDNN operations
- `cublas`: cuBLAS operations

**Other Options:**

- `cuda_memory_usage`: Track GPU memory allocations
- `capture_range`: How to control profiling (`cudaProfilerApi` uses code markers)
- `stats`: Generate summary statistics after profiling
- `export_json`: Export timeline to JSON format

### Viewing Profile Results

#### Method 1: Nsight Systems UI (Recommended)

```bash
# Open profile in Nsight Systems GUI
nsys-ui output/profile_cuda_llama_balanced.nsys-rep
```

**What you'll see:**
- GPU kernel timeline with detailed timing
- CUDA memory allocations and transfers
- CPU thread activity
- Distributed training communication patterns (NCCL)
- Per-GPU utilization and occupancy

#### Method 2: Export to JSON (Chrome Tracing)

```bash
# Export to JSON format
nsys export --type=json output/profile_cuda_llama_balanced.nsys-rep

# Open Chrome and navigate to: chrome://tracing
# Click "Load" and select the exported .json file
```

#### Method 3: Command-Line Statistics

```bash
# View profiling statistics
nsys stats output/profile_cuda_llama_balanced.nsys-rep

# Export statistics to CSV
nsys stats --report cuda_gpu_kern_sum output/profile.nsys-rep --format csv
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
python3 train_llama.py
sleep 30
python3 train_qwen.py
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

# Manually extract metrics (for AMD/Prim)
python3 extract_metrics.py \
    --log-file output/training_llama.log \
    --model-name llama \
    --output output/train_prim_llama.json \
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

#### NVD Issues

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

**2. Prim configuration:**
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
├── train_nemo_llama.json          # NVD Llama metrics
├── train_nemo_qwen.json           # NVD Qwen metrics
├── train_prim_llama.json        # AMD Llama metrics (if available)
├── train_prim_qwen.json         # AMD Qwen metrics (if available)
├── training_llama.log             # Complete training logs
└── training_qwen.log              # Complete training logs
```

**Backward Compatibility:**

The comparison tools support both new and old naming conventions:
- **New format**: `train_{framework}_{model}.json` (e.g., `train_nemo_llama.json`, `train_prim_qwen.json`)
- **Old format**: `benchmark_{platform}_{model}.json` (e.g., `benchmark_cuda_llama.json`, `benchmark_rocm_qwen.json`)

Both formats can be used interchangeably with `compare.py` and other analysis tools.

### Benchmark JSON Structure

```json
{
  "platform": "nvd",
  "gpu_info": {
    "device_name": "NVD H100 80GB HBM3",
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
  "raw_loss_values": [...]
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

## Advanced Topics

### Environment Variables

Instead of command-line flags, you can use environment variables:

```bash
# Set parallelism strategy
export PARALLEL="minimal_communication"
python3 train_llama.py

# Set output directory
export OUTPUT_DIR="./my-results"
python3 benchmark.py
```

### Running Individual Models

```bash
# Direct script execution
export PARALLEL="balanced"
export OUTPUT_DIR="./output"

python3 train_llama.py
# Wait for GPU memory to clear
sleep 30
python3 train_qwen.py
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

### extract_metrics.py

Extract metrics from AMD/Prim training logs.

```bash
python3 extract_metrics.py \
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
  platforms:                   # NVD and AMD configs
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
platform_optimizations:        # NVD/AMD specific settings
benchmarking:                  # Metrics and output config
```

### File Structure

```
tprimat/
├── benchmark.py               # Main entry point
├── train_llama.py         # Llama training script
├── train_qwen.py          # Qwen training script
├── compare.py                # Comparison plot generator
├── extract_metrics.py # AMD log parser
├── config_loader.py          # Configuration loader
├── utils.py                  # Benchmark utilities
├── config.yaml               # Main configuration
├── requirements.txt          # Python dependencies
├── run_all_configs.sh        # Run all configurations
├── compare_all_configs.sh    # Generate all plots
└── README.md                 # This file
```

---

## Credits & License

**TPrimat** - Tensor Parallelism AMD/NVD Training Benchmark Suite

Developed for fair, reproducible LLM training benchmarks across GPU platforms.

### Frameworks Used

- **NeMo**: NVD's neural module framework
- **Prim**: AMD's training framework
- **Megatron**: Model-parallel transformers

### Models

- **Llama 3.1 8B**: Meta AI
- **Qwen 2.5 7B**: Alibaba Cloud

---

## Quick Reference Card

### Plain Scripts (Most Common)

```bash
# ========================================
# RUN ALL FRAMEWORKS (RECOMMENDED)
# ========================================

# NVD: Run HF, NeMo
./run_nvidia_all.sh

# AMD: Run HF, NeMo  
./run_amd_all.sh

# In Docker
./run_docker_nvd.sh ./run_nvidia_all.sh
./run_docker_amd.sh ./run_amd_all.sh

# ========================================
# RUN INDIVIDUAL FRAMEWORKS
# ========================================

# Transformers (both models)
python3 -u run_tran_plain.py

# NeMo (both models)
python3 -u run_nemo_plain.py

# Single model only
python3 -u run_tran_plain.py llama

# ========================================
# VIEW RESULTS
# ========================================

# List all results
ls -lh output/train_*.json

# View specific result
cat output/train_tran_llama.json

# Compare throughput across frameworks
for f in output/train_*_llama.json; do
  echo "$f: $(jq -r '.performance_metrics.tokens_per_second_per_gpu' $f) tok/s/GPU"
done

# ========================================
# MONITORING
# ========================================

# Check GPUs
nvidia-smi              # NVD
rocm-smi                # AMD
watch -n 1 nvidia-smi   # Continuous monitoring

# View logs in real-time
tail -f output/training_llama.log
docker logs -f primat
```

### Benchmark Suite Commands

```bash
# ========================================
# TRADITIONAL BENCHMARK SUITE
# ========================================

# Quick single benchmark
./benchmark.py --parallel minimal_communication

# All 5 configurations
./run_all_configs.sh

# Specific model
./benchmark.py --model llama --parallel balanced

# Custom output directory
./benchmark.py --output-dir ./my-results

# Multiple runs for statistics
./benchmark.py --runs 3

# Generate comparison plots
python3 compare.py --results-dir ./output

# Help
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

### Framework Quick Guide

| Framework | Script | Output Files | When to Use |
|-----------|--------|--------------|-------------|
| Transformers | `run_tran_plain.py` | `train_tran_*.json` | Standard training, easy debugging |
| NeMo | `run_nemo_plain.py` | `train_nemo_*.json` | Advanced features, FP8, best performance |
| Prim (AMD) | `run_prim_all.sh` | `train_prim_*.json` | AMD-optimized, MI300X |

### Common Troubleshooting

```bash
# Not seeing logs? Use -u flag
python3 -u run_tran_plain.py llama

# Kill all Python processes
pkill -f python

# Reset GPUs (NVD, requires sudo)
sudo nvidia-smi --gpu-reset

# Check GPU memory
nvidia-smi --query-gpu=memory.free --format=csv

# Save logs to file
python3 -u run_tran_plain.py llama 2>&1 | tee output/hf_llama.log
```

### Output Files Reference

```bash
output/
├── train_tran_llama.json          # Transformers Llama
├── train_tran_qwen.json           # Transformers Qwen
├── train_nemo_llama.json        # NeMo Llama
├── train_nemo_qwen.json         # NeMo Qwen
├── train_prim_llama.json      # Prim Llama (AMD)
└── train_prim_qwen.json       # Prim Qwen (AMD)
```

---

**For questions, issues, or contributions, please check the project repository.**

