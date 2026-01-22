# Training Alternatives Without NeMo

This document compares different approaches for training LLMs without NeMo.

## Overview

| Script | Framework | Parallelism | Memory Efficiency | FP8 Support | Ease of Use | Performance |
|--------|-----------|-------------|-------------------|-------------|-------------|-------------|
| `run_nemo_standalone.py` (original) | NeMo + Megatron | Tensor, Pipeline, Sequence | ⭐⭐⭐⭐⭐ | ✅ Yes | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| `run_hf_standalone.py` | HuggingFace Transformers | DDP | ⭐⭐⭐ | ❌ No | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| `run_fsdp_standalone.py` | PyTorch FSDP | FSDP | ⭐⭐⭐⭐ | ❌ No | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| `run_deepspeed_standalone.py` | DeepSpeed ZeRO | ZeRO | ⭐⭐⭐⭐⭐ | ⚠️ Limited | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

## Option 1: HuggingFace Transformers (`run_hf_standalone.py`)

### Pros
- **Simplest to use** - minimal code, intuitive API
- **Most portable** - works on any hardware (NVIDIA, AMD, etc.)
- **Great ecosystem** - extensive documentation and community support
- **Easy debugging** - straightforward stack traces
- **Flash Attention 2** support for optimized attention computation

### Cons
- **No FP8 support** - limited to BF16/FP16
- **No advanced parallelism** - only DDP (Data Parallel)
- **Higher memory usage** - no model sharding by default
- **Lower performance** - compared to Megatron-optimized training

### How to Run
```bash
# Single GPU
python run_hf_standalone.py llama

# Multi-GPU (uses DDP automatically)
torchrun --nproc_per_node=8 run_hf_standalone.py llama
```

### Best For
- Smaller models (< 13B parameters)
- Single-node training
- Development and experimentation
- Non-NVIDIA hardware (AMD, Intel)
- When portability is more important than performance

## Option 2: PyTorch FSDP (`run_fsdp_standalone.py`)

### Pros
- **Native PyTorch** - no external dependencies
- **Good memory efficiency** - full model sharding
- **Flexible** - fine-grained control over sharding strategy
- **Good performance** - optimized for PyTorch ecosystem
- **Active development** - latest PyTorch features

### Cons
- **No FP8 support** - limited to BF16/FP16
- **More complex** - requires understanding of FSDP concepts
- **No built-in recipes** - need to configure everything manually
- **NVIDIA-focused** - best performance on NVIDIA GPUs

### How to Run
```bash
# Must use torchrun for distributed training
torchrun --nproc_per_node=8 run_fsdp_standalone.py llama
```

### Best For
- Medium to large models (7B-70B parameters)
- Single-node or small multi-node training
- When you want native PyTorch without extra dependencies
- Memory-constrained scenarios
- When you need fine-grained control over parallelism

## Option 3: DeepSpeed ZeRO (`run_deepspeed_standalone.py`)

### Pros
- **Excellent memory efficiency** - ZeRO optimization stages
- **Scalable** - works well for very large models
- **Well-tested** - used by many large-scale projects
- **Flexible configuration** - JSON-based config
- **Good optimization** - many optimizations built-in

### Cons
- **Limited FP8 support** - experimental, not as mature as NeMo
- **More dependencies** - requires DeepSpeed installation
- **Configuration complexity** - many tuning parameters
- **Different launcher** - requires deepspeed command

### How to Run
```bash
# Using DeepSpeed launcher (recommended)
deepspeed --num_gpus=8 run_deepspeed_standalone.py llama

# Or using torchrun (may have issues)
torchrun --nproc_per_node=8 run_deepspeed_standalone.py llama
```

### Best For
- Large models (13B+ parameters)
- Multi-node training
- When memory efficiency is critical
- When you need ZeRO-3 for extremely large models
- Azure and Microsoft ecosystem

## Key Differences from NeMo

### What You Lose
1. **FP8 Training** - NeMo has mature FP8 support via Megatron; alternatives are BF16 only
2. **Tensor Parallelism** - NeMo supports tensor parallel (split layers across GPUs); alternatives use data parallel or FSDP
3. **Pipeline Parallelism** - NeMo can pipeline model stages; alternatives don't have this
4. **Sequence Parallelism** - NeMo can split long sequences; alternatives process full sequences
5. **Pre-built Recipes** - NeMo has optimized recipes for common models; alternatives need manual tuning
6. **Indexed Datasets** - NeMo's efficient `.idx` format; alternatives use standard datasets

### What You Gain
1. **Simplicity** - Less abstraction, more direct control
2. **Portability** - HuggingFace works everywhere
3. **Flexibility** - Easier to customize and experiment
4. **Community** - Larger ecosystem and more examples
5. **Debugging** - Simpler stack traces and error messages

## Performance Comparison (Expected)

For 8B parameter model on 8x NVIDIA H100 GPUs:

| Framework | Throughput | Memory/GPU | Training Time (10 steps) |
|-----------|------------|------------|--------------------------|
| NeMo (FP8) | ~100% | ~40GB | Baseline |
| NeMo (BF16) | ~60% | ~60GB | +66% |
| HuggingFace | ~40% | ~70GB | +150% |
| FSDP | ~50% | ~50GB | +100% |
| DeepSpeed | ~55% | ~45GB | +80% |

*Note: Actual performance depends on many factors including GPU, model size, sequence length, etc.*

## Recommendations

### Use NeMo if:
- You're on NVIDIA GPUs (especially H100/A100)
- You need maximum performance
- You want FP8 training
- You're training models > 30B parameters
- You need tensor/pipeline parallelism

### Use HuggingFace if:
- You're just getting started
- You need maximum portability
- You're on non-NVIDIA hardware
- You're training models < 13B parameters
- You want the simplest code

### Use FSDP if:
- You want native PyTorch
- You need good memory efficiency
- You're training models 7B-70B parameters
- You want minimal dependencies

### Use DeepSpeed if:
- You're training very large models (> 70B)
- You need maximum memory efficiency
- You're using Azure/Microsoft infrastructure
- You want ZeRO-3 for extreme model sizes

## Installation Requirements

### HuggingFace
```bash
pip install transformers accelerate torch flash-attn
```

### FSDP
```bash
pip install torch transformers
# Ensure PyTorch 2.0+ for best FSDP support
```

### DeepSpeed
```bash
pip install deepspeed transformers torch
# May need to compile from source for latest features
```

## Migration Path

If you're currently using NeMo and want to migrate:

1. **Start with HuggingFace** - Get familiar with the model and data
2. **Move to FSDP** - When you need better memory efficiency
3. **Try DeepSpeed** - When scaling to very large models
4. **Stay with NeMo** - If you need maximum performance on NVIDIA

Each step up in complexity brings better memory efficiency and scalability, but NeMo still offers the best performance for NVIDIA GPUs with FP8 support.
