# AMD Framework Implementation Guide

This document describes the AMD framework options available in TPrimat for benchmarking LLM training on AMD MI300X GPUs with ROCm.

## Available Frameworks

TPrimat now supports **5 frameworks** on AMD GPUs, matching the variety available on NVIDIA:

| Framework | Identifier | Scripts | Description |
|-----------|-----------|---------|-------------|
| **Primus** | `prim` | `train_amd_prim_llama.sh`, `train_amd_prim_qwen.sh` | AMD's ROCm-optimized Megatron framework |
| **NeMo** | `nemo` | `train_amd_nemo_llama.sh`, `train_amd_nemo_qwen.sh` | NVIDIA NeMo with ROCm support |
| **Transformers** | `tran` | `train_amd_tran_llama.sh`, `train_amd_tran_qwen.sh` | HuggingFace Transformers with DDP |
| **DeepSpeed** | `deep` | `train_amd_deep_llama.sh`, `train_amd_deep_qwen.sh` | Microsoft DeepSpeed with ZeRO-3 |
| **FSDP** | `fsdp` | `train_amd_fsdp_llama.sh`, `train_amd_fsdp_qwen.sh` | PyTorch Fully Sharded Data Parallel |
| **Megatron-DS** | `mgds` | `train_amd_mgds_llama.sh`, `train_amd_mgds_qwen.sh` | Megatron-DeepSpeed (optional) |

## Quick Start

### Run All Frameworks

```bash
# Run all AMD frameworks (Primus, NeMo, Transformers, DeepSpeed, FSDP)
./train_amd.sh

# Or with config loading
./train_amd.sh
```

### Run Specific Framework

```bash
# Run only specific framework(s)
FRAMEWORKS="nemo" ./train_amd.sh
FRAMEWORKS="transformers,deepspeed" ./train_amd.sh
FRAMEWORKS="fsdp" ./train_amd.sh
```

### Run Individual Scripts

```bash
# NeMo
./train_amd_nemo_llama.sh
./train_amd_nemo_qwen.sh

# Transformers
./train_amd_tran_llama.sh
./train_amd_tran_qwen.sh

# DeepSpeed
./train_amd_deep_llama.sh
./train_amd_deep_qwen.sh

# FSDP
./train_amd_fsdp_llama.sh
./train_amd_fsdp_qwen.sh

# Megatron-DeepSpeed (requires separate installation)
./train_amd_mgds_llama.sh
./train_amd_mgds_qwen.sh
```

## Framework Details

### 1. Primus (AMD-Optimized)

**Description:** AMD's fork of Megatron-LM optimized for ROCm and MI300X.

**Features:**
- Native ROCm optimization
- FP8 precision support
- Tensor, pipeline, and data parallelism
- Best performance on MI300X

**Requirements:**
- Primus installation (from base Docker image)
- Set `PRIMUS_PATH` environment variable

**Output:** `train_amd_prim_llama.json`, `train_amd_prim_qwen.json`

### 2. NeMo with ROCm

**Description:** NVIDIA NeMo Framework with ROCm support for cross-platform comparison.

**Features:**
- Direct comparison with NVIDIA NeMo results
- Tensor, pipeline, and data parallelism
- ROCm backend support

**Requirements:**
- NeMo with ROCm support (may need to build from source)
- Same API as NVIDIA version

**Output:** `train_nemo_llama.json`, `train_nemo_qwen.json`

### 3. HuggingFace Transformers

**Description:** Standard HuggingFace Transformers library with DistributedDataParallel.

**Features:**
- Pure data parallelism (DP=8)
- Easy to use and portable
- Gradient checkpointing for memory efficiency
- Works on both NVIDIA and AMD

**Requirements:**
- `transformers>=4.36.0`
- `accelerate>=0.25.0`

**Output:** `train_tran_llama.json`, `train_tran_qwen.json`

### 4. DeepSpeed

**Description:** Microsoft DeepSpeed with ZeRO-3 optimization for memory efficiency.

**Features:**
- ZeRO Stage 3 (full parameter sharding)
- CPU offloading for optimizer and parameters
- ROCm compatibility (v0.12.0+)
- Memory-efficient for large models

**Requirements:**
- `deepspeed>=0.12.0` with ROCm support

**Output:** `train_deep_llama.json`, `train_deep_qwen.json`

### 5. PyTorch FSDP

**Description:** PyTorch's native Fully Sharded Data Parallel implementation.

**Features:**
- Native PyTorch (no external dependencies)
- Full parameter sharding
- Gradient checkpointing
- Works seamlessly with ROCm

**Requirements:**
- PyTorch with ROCm (from base image)

**Output:** `train_fsdp_llama.json`, `train_fsdp_qwen.json`

### 6. Megatron-DeepSpeed (Optional)

**Description:** Combines Megatron's model parallelism with DeepSpeed's memory optimizations.

**Features:**
- Hybrid parallelism (TP + PP + ZeRO-1)
- Good for very large models
- Requires separate installation

**Requirements:**
- Clone: `git clone https://github.com/microsoft/Megatron-DeepSpeed.git`
- Set `MEGATRON_DEEPSPEED_PATH` environment variable

**Output:** `train_mgds_llama.json`, `train_mgds_qwen.json`

## Configuration

Framework-specific settings are in `config.yaml`:

```yaml
frameworks:
  nemo:
    description: "NVIDIA NeMo Framework (supports ROCm)"
    amd:
      enabled: true
      default_precision: "bf16"
      supports_fp8: true
  
  transformers:
    description: "HuggingFace Transformers with DDP"
    amd:
      enabled: true
      gradient_checkpointing: true
  
  deepspeed:
    description: "Microsoft DeepSpeed with ZeRO"
    amd:
      enabled: true
      default_zero_stage: 3
      cpu_offload: true
  
  fsdp:
    description: "PyTorch FSDP"
    amd:
      enabled: true
      default_sharding_strategy: "FULL_SHARD"
```

## Results Comparison

After running benchmarks, compare results:

```bash
# Compare all results (NVIDIA vs AMD, all frameworks)
python3 compare.py

# AMD-only comparison
python3 compare.py --platform amd

# Specific frameworks
python3 compare.py --frameworks nemo,transformers,deepspeed
```

## Output Files

All frameworks produce unified JSON output in `output/`:

```
output/
├── train_amd_prim_llama.json      # Primus + Llama
├── train_amd_prim_qwen.json       # Primus + Qwen
├── train_nemo_llama.json      # NeMo + Llama
├── train_nemo_qwen.json       # NeMo + Qwen
├── train_tran_llama.json      # Transformers + Llama
├── train_tran_qwen.json       # Transformers + Qwen
├── train_deep_llama.json      # DeepSpeed + Llama
├── train_deep_qwen.json       # DeepSpeed + Qwen
├── train_fsdp_llama.json      # FSDP + Llama
└── train_fsdp_qwen.json       # FSDP + Qwen
```

## Performance Expectations

**Expected throughput on MI300X (8 GPUs):**

| Framework | Llama 3.1 8B | Qwen 2.5 7B | Notes |
|-----------|--------------|-------------|-------|
| Primus | Best | Best | AMD-optimized, FP8 support |
| NeMo | Good | Good | Cross-platform comparison |
| Transformers | Moderate | Moderate | Simple, portable |
| DeepSpeed | Good | Good | Memory-efficient |
| FSDP | Good | Good | Native PyTorch |

## Docker Usage

```bash
# Build AMD container
docker build -f amd.Dockerfile -t primat:amd .

# Run all AMD frameworks
./train_docker_amd.sh ./train_amd.sh

# Run specific framework
./train_docker_amd.sh ./train_amd_nemo_llama.sh
```

## Environment Variables

```bash
# Number of GPUs (default: 8)
export CONFIG_AMD_NUM_GPUS=8

# Output directory
export CONFIG_OUTPUT_DIR=./output

# Framework paths
export PRIMUS_PATH=/workspace/Primus
export MEGATRON_DEEPSPEED_PATH=/workspace/Megatron-DeepSpeed
```

## Troubleshooting

### NeMo ROCm Compatibility

If NeMo doesn't work with ROCm out of the box:

```bash
# Build NeMo from source with ROCm support
git clone https://github.com/NVIDIA/NeMo.git
cd NeMo
pip install -e .
```

### DeepSpeed ROCm Support

Ensure DeepSpeed is installed with ROCm support:

```bash
pip install deepspeed>=0.12.0
```

### Memory Issues

If you encounter OOM errors:

1. **Transformers/FSDP:** Reduce `per_device_batch_size`
2. **DeepSpeed:** Enable CPU offloading (already enabled by default)
3. **NeMo/Primus:** Increase TP size or reduce batch size

## Comparison with NVIDIA

You can now do apples-to-apples comparisons:

| NVIDIA Framework | AMD Equivalent |
|------------------|----------------|
| Megatron | Primus |
| NeMo | NeMo (ROCm) |
| Transformers | Transformers |
| DeepSpeed | DeepSpeed |
| N/A | FSDP |

## Next Steps

1. Run benchmarks: `./train_amd.sh`
2. Compare results: `python3 compare.py`
3. Analyze metrics in `output/` directory
4. Generate comparison plots

For more details, see the main [README.md](README.md).
