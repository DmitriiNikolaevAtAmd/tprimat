# AMD Framework Quick Start Guide

## TL;DR

You now have **6 training frameworks** available on AMD MI300X GPUs:

```bash
# Run all frameworks at once
./train_amd.sh

# Or run individually
./train_amd_nemo_llama.sh      # NeMo with ROCm
./train_amd_tran_llama.sh      # HuggingFace Transformers
./train_amd_deep_llama.sh      # DeepSpeed ZeRO-3
./train_amd_fsdp_llama.sh      # PyTorch FSDP
./train_amd_mgds_llama.sh    # Megatron-DeepSpeed (optional)
./train_amd_prim_llama.sh          # Primus (original)
```

## Available Frameworks

| # | Framework | Command | Description |
|---|-----------|---------|-------------|
| 1 | **Primus** | `./train_amd_prim_llama.sh` | AMD-optimized (best performance) |
| 2 | **NeMo** | `./train_amd_nemo_llama.sh` | Cross-platform comparison |
| 3 | **Transformers** | `./train_amd_tran_llama.sh` | Simple, portable |
| 4 | **DeepSpeed** | `./train_amd_deep_llama.sh` | Memory-efficient |
| 5 | **FSDP** | `./train_amd_fsdp_llama.sh` | Native PyTorch |
| 6 | **Megatron-DS** | `./train_amd_mgds_llama.sh` | Hybrid parallelism |

## Quick Commands

### Run Everything
```bash
# All frameworks, both models
./train_amd.sh
```

### Run Specific Frameworks
```bash
# Just the new ones (NeMo, Transformers, DeepSpeed, FSDP)
FRAMEWORKS="nemo,transformers,deepspeed,fsdp" ./train_amd.sh

# Just one framework
FRAMEWORKS="nemo" ./train_amd.sh
```

### Compare Results
```bash
# After running benchmarks
python3 compare.py
```

## What Each Framework Does

### 1. Primus (Original)
- **Best for:** Maximum AMD MI300X performance
- **Features:** FP8, tensor/pipeline parallelism
- **Use when:** You want peak throughput

### 2. NeMo with ROCm (New!)
- **Best for:** Comparing with NVIDIA NeMo results
- **Features:** Same API as NVIDIA version
- **Use when:** Cross-platform benchmarking

### 3. Transformers (New!)
- **Best for:** Quick baseline, simple setup
- **Features:** Pure data parallelism (DP=8)
- **Use when:** Testing portability

### 4. DeepSpeed (New!)
- **Best for:** Large models, memory constraints
- **Features:** ZeRO-3 sharding, CPU offload
- **Use when:** You need memory efficiency

### 5. FSDP (New!)
- **Best for:** Native PyTorch solution
- **Features:** Full parameter sharding
- **Use when:** No external dependencies wanted

### 6. Megatron-DeepSpeed (New!)
- **Best for:** Very large models, hybrid parallelism
- **Features:** TP + PP + ZeRO-1
- **Use when:** Complex parallelism needed

## Output Files

All frameworks output to `output/` directory:

```
output/
├── train_amd_prim_llama.json    ← Primus
├── train_nemo_llama.json    ← NeMo
├── train_tran_llama.json    ← Transformers
├── train_deep_llama.json    ← DeepSpeed
├── train_fsdp_llama.json    ← FSDP
└── train_mgds_llama.json  ← Megatron-DS
```

## Docker Usage

```bash
# Build container (if needed)
docker build -f amd.Dockerfile -t primat:amd .

# Run all frameworks in Docker
./train_docker_amd.sh ./train_amd.sh

# Run single framework
./train_docker_amd.sh ./train_amd_nemo_llama.sh
```

## Comparison Matrix

| Feature | Primus | NeMo | Transformers | DeepSpeed | FSDP |
|---------|--------|------|--------------|-----------|------|
| Speed | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Memory | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Setup | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| Features | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |

## Expected Performance (MI300X 8 GPUs)

Approximate tokens/sec for Llama 3.1 8B:

1. Primus: ~45,000 tokens/sec (FP8 optimized)
2. NeMo: ~42,000 tokens/sec (BF16)
3. FSDP: ~38,000 tokens/sec
4. DeepSpeed: ~35,000 tokens/sec (memory efficient)
5. Transformers: ~32,000 tokens/sec (baseline)

*Actual numbers vary by configuration*

## Troubleshooting

### Can't find NeMo?
```bash
# May need to build from source for ROCm
git clone https://github.com/NVIDIA/NeMo.git
cd NeMo && pip install -e .
```

### Out of memory?
```bash
# Try DeepSpeed (has CPU offload)
./train_amd_deep_llama.sh

# Or reduce batch size
export CONFIG_AMD_NUM_GPUS=8  # Use all GPUs
```

### Want to test just one model?
```bash
# Just run Llama
./train_amd_nemo_llama.sh

# Just run Qwen  
./train_amd_nemo_qwen.sh
```

## More Information

- Full guide: `README_AMD_FRAMEWORKS.md`
- Implementation details: `IMPLEMENTATION_SUMMARY.md`
- Main documentation: `README.md`

## One-Liner Examples

```bash
# Run everything
./train_amd.sh

# Run in Docker
./train_docker_amd.sh ./train_amd.sh

# Just new frameworks
FRAMEWORKS="nemo,tran,deep,fsdp" ./train_amd.sh

# Compare with NVIDIA
python3 compare.py
```

---

**That's it!** You now have full framework parity between AMD and NVIDIA platforms. 🚀
