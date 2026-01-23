# Multi-GPU Training Guide

## Problem: Script Only Using One GPU

If you see only one GPU being utilized (e.g., GPU 0 at 100%, others at 0%), it's because the script is running in **single-process mode** without distributed training.

## Solution: Use `torchrun` for Multi-GPU Training

### Quick Start (Recommended)

Use the provided launcher scripts that automatically detect and use all available GPUs:

```bash
# For NVIDIA GPUs
bash run_nvidia_all.sh

# For AMD/ROCm GPUs
bash run_amd_all.sh
```

### Manual Launch with torchrun

To manually run with all GPUs:

```bash
# Auto-detect number of GPUs
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)  # NVIDIA
# or
NUM_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())")  # AMD/ROCm

# Run HuggingFace standalone with all GPUs
torchrun --nproc_per_node=$NUM_GPUS run_hf_standalone.py

# Run FSDP standalone with all GPUs
torchrun --nproc_per_node=$NUM_GPUS run_fsdp_standalone.py
```

### Specific Number of GPUs

To use a specific number of GPUs (e.g., 4 out of 8):

```bash
torchrun --nproc_per_node=4 run_hf_standalone.py
```

## How It Works

### Without torchrun (Single GPU - ❌ Wrong)
```bash
python3 run_hf_standalone.py
```
- Only spawns 1 process
- Uses only GPU 0
- Ignores other GPUs

### With torchrun (All GPUs - ✅ Correct)
```bash
torchrun --nproc_per_node=8 run_hf_standalone.py
```
- Spawns 8 processes (one per GPU)
- Each process uses a different GPU
- Automatic DDP (DistributedDataParallel) setup
- Near-linear speedup with multiple GPUs

## Batch Size Calculation

The scripts automatically adjust batch sizes based on GPU count:

```python
global_batch_size = 64
num_gpus = 8
per_device_batch_size = 1
gradient_accumulation_steps = 64 // (1 * 8) = 8

# Effective batch size = per_device_batch_size × num_gpus × gradient_accumulation_steps
# = 1 × 8 × 8 = 64 ✓
```

## Verifying Multi-GPU Usage

Check GPU utilization while training:

```bash
# Watch GPU usage (updates every 1 second)
watch -n 1 nvidia-smi

# Or for a snapshot
nvidia-smi
```

You should see:
- **All GPUs** showing ~100% utilization
- **Similar memory usage** across all GPUs
- **Compute processes** running on each GPU

## Environment Variables

The launcher scripts set these automatically:

```bash
export PYTHONUNBUFFERED=1                          # Real-time output
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'  # Memory efficiency
export PYTHONHASHSEED="42"                        # Reproducibility

# AMD/ROCm specific
export HSA_NO_SCRATCH_RECLAIM=1
export HSA_ENABLE_SDMA=1
export HSA_FORCE_FINE_GRAIN_PCIE=1
```

## Troubleshooting

### Issue: "NCCL error" or "Connection refused"

**Solution**: Ensure all GPUs are visible and accessible:
```bash
nvidia-smi
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```

### Issue: Out of Memory (OOM)

**Solution**: Reduce per-device batch size or increase gradient accumulation:
```python
per_device_train_batch_size=1  # Try reducing to 1
gradient_accumulation_steps=16  # Increase to maintain global batch size
```

### Issue: "Address already in use"

**Solution**: torchrun uses random ports. Specify explicitly:
```bash
torchrun --nproc_per_node=8 --master_port=29500 run_hf_standalone.py
```

## Performance Expectations

With proper multi-GPU setup on 8× H100 GPUs:

- **Single GPU**: ~1.5 steps/sec → 6-7 seconds/step
- **8 GPUs (DDP)**: ~10-12 steps/sec → <1 second/step
- **Expected speedup**: 6-7× (near-linear)

## Summary

✅ **DO**: Use `torchrun` or the launcher scripts
❌ **DON'T**: Run with plain `python3` command for multi-GPU training

The launcher scripts (`run_nvidia_all.sh`, `run_amd_all.sh`) handle everything automatically!
