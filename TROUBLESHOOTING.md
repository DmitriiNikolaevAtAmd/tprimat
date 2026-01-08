# TensorPrimat Troubleshooting Guide

## Common Issues and Solutions

---

## CUDA Out of Memory Error

### Symptom
```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate X.XX GiB. 
GPU N has a total capacity of 79.18 GiB of which only X.XX GiB is free.
```

### Root Causes

1. **Lingering processes from previous runs**
   - Previous training didn't clean up properly
   - GPU memory still allocated

2. **Memory fragmentation**
   - PyTorch allocator fragmented memory
   - Can't allocate contiguous blocks

3. **Configuration too aggressive**
   - Batch size too large for available memory
   - Model parallelism settings incompatible

4. **Multiple users/processes**
   - Other users running on same GPUs
   - Competing for memory resources

### Quick Fix

Run the memory cleanup script:

```bash
./fix_gpu_memory.sh
```

This will:
- ✅ Show current GPU memory usage
- ✅ Find and kill lingering Python processes
- ✅ Clear PyTorch cache
- ✅ Verify memory is freed

### Manual Solutions

#### 1. Kill lingering processes

```bash
# Check what's using GPUs
nvidia-smi

# Kill all Python processes on GPU
nvidia-smi | grep python | awk '{print $5}' | xargs -r kill -9

# Or kill specific process by PID
kill -9 <PID>
```

#### 2. Clear PyTorch cache

```bash
python3 -c "import torch; torch.cuda.empty_cache(); print('Cache cleared')"
```

#### 3. Set memory allocator

**Already fixed in pretrain scripts**, but you can also set manually:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

#### 4. Reduce memory usage

Edit the pretrain script to use less memory:

```python
# Reduce batch size
recipe.data.global_batch_size = 64  # Instead of 128

# Or use fewer GPUs
recipe = llm.llama31_8b.pretrain_recipe(
    num_gpus_per_node=4,  # Instead of 8
)

# Or increase tensor parallelism (splits model more)
recipe.trainer.strategy.tensor_model_parallel_size = 8  # Instead of 4
```

#### 5. Free up all GPU memory (requires root)

```bash
# Restart the system
sudo reboot

# Or restart the GPU (dangerous!)
sudo nvidia-smi --gpu-reset
```

---

## NeMo Not Found Error

### Symptom
```
ModuleNotFoundError: No module named 'nemo'
```

### Solution

This is expected on AMD systems. The benchmark automatically:
1. Detects no NeMo available
2. Switches to log extraction mode
3. Searches for Primus training logs

Just run Primus training first, then:
```bash
./benchmark.py  # Will auto-extract from logs
```

---

## No Logs Found Error

### Symptom
```
⚠️  No log file found for llama
```

### Solutions

1. **Provide log paths explicitly:**
```bash
LLAMA_LOG=/path/to/llama.log ./benchmark.py
```

2. **Copy logs to current directory:**
```bash
cp /path/to/logs/*.log .
./benchmark.py
```

3. **Name logs correctly:**
```bash
# Script looks for these patterns:
training_llama.log
training_mistral.log
training_qwen.log
```

See `AMD_COMPARISON_GUIDE.md` for details.

---

## Platform Detection Issues

### Symptom
```
❌ No AMD benchmark results found!
```
But you have ROCm benchmark files.

### Solution

Already fixed! The issue was conflicting `platform` and `software_stack` fields. 

If you still see this:
1. Check your JSON files have `"software_stack": "rocm"` or `"cuda"`
2. Re-run: `python3 compare_results.py`

---

## Comparison Plot Errors

### Symptom
```
ValueError: Unknown format code 'f' for object of type 'str'
```

### Solution

Already fixed! This happened when GPU info had "N/A" values. The comparison now handles these gracefully.

---

## Memory Usage Questions

### "Why is NVIDIA using 22GB but AMD using 118GB?"

**Answer:** Different parallelism strategies:
- **NVIDIA**: TP=4 (model split across 4 GPUs) → each GPU has 1/4 of model
- **AMD**: TP=1 (full model per GPU) → each GPU has full model

This is **intentional** for maximum performance comparison. See `COMPARISON_METHODOLOGY.md`.

### "Should I make them match?"

**Depends on your goal:**
- **Real-world performance**: Keep different (current approach) ✅
- **Hardware-only comparison**: Match TP=4 on both (see `AMD_COMPARISON_GUIDE.md`)

---

## Performance Issues

### "Training is slower than expected"

**Checklist:**

1. **Check GPU utilization:**
```bash
watch -n 1 nvidia-smi
```
Should see ~100% GPU utilization during training.

2. **Check MFU (Model FLOPs Utilization):**
```bash
python3 compare_with_enhanced_metrics.py
```
Good MFU: 30-50%, Excellent: 50-65%

3. **Memory not being used effectively:**
- If memory < 50% utilized, try larger batch size
- If memory > 95%, reduce batch size

4. **Check for CPU bottleneck:**
```bash
htop  # or top
```
If CPUs at 100%, data loading is slow.

---

## Best Practices

### Before Running Benchmarks

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

### During Benchmarks

- ✅ Don't run other GPU processes
- ✅ Monitor with `nvidia-smi`
- ✅ Let it run without interruption
- ✅ Check logs for errors

### After Benchmarks

- ✅ Verify JSON files created in `output/`
- ✅ Check metrics make sense
- ✅ Run comparison: `python3 compare_results.py`
- ✅ Review enhanced metrics: `python3 compare_with_enhanced_metrics.py`

---

## Getting Help

### Check Configuration
```bash
./check_primus_config.sh  # For AMD/Primus
```

### Verify Benchmark Output
```bash
ls -lh output/
cat output/benchmark_*.json
```

### Check Logs
Look for errors in terminal output or log files in `output/` directory.

### Documentation
- `README.md` - Main documentation
- `COMPARISON_METHODOLOGY.md` - Comparison approaches
- `AMD_COMPARISON_GUIDE.md` - AMD-specific guide
- `RECOMMENDED_METRICS.md` - Metric explanations

---

*Last Updated: January 2026*
