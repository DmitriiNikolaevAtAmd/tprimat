# What Gets Saved During Benchmarking

## ✅ What IS Saved (Logs & Profiles Only)

### Benchmark Results
```
benchmark_results/
└── benchmark_{platform}_{timestamp}.json
```

**Size**: ~5-10 KB per run  
**Contents**:
- Platform info (CUDA/ROCm)
- GPU specifications
- Performance metrics (step times, throughput)
- Memory usage statistics
- Per-core metrics
- Raw timing data

**Purpose**: Lightweight profiling data for AMD vs NVIDIA comparison

### Console Output
- Real-time training progress
- Step timing information
- Benchmark summary at the end

## ❌ What is NOT Saved

### Model Checkpoints
- ✗ No model weights
- ✗ No optimizer states
- ✗ No training state

### TensorBoard Logs
- ✗ No TensorBoard events
- ✗ No metric histories
- ✗ No computational graphs

### WandB Logs
- ✗ No WandB uploads
- ✗ No experiment tracking

### Validation Results
- ✗ No validation checkpoints
- ✗ No validation metrics

### Other Intermediate Files
- ✗ No distributed checkpoint files
- ✗ No async save states
- ✗ No hparams files

## Configuration

All three training scripts are configured to disable saves:

```python
# DISABLE ALL CHECKPOINTING AND INTERMEDIATE SAVES
recipe.trainer.enable_checkpointing = False
recipe.log.ckpt = None
recipe.resume = None

# Disable TensorBoard and other loggers
recipe.log.tensorboard = None
recipe.log.wandb = None

# Disable validation
recipe.trainer.val_check_interval = None
recipe.trainer.check_val_every_n_epoch = None
```

## What You'll See

### On Disk (After Training)
```
week-02/code/
└── benchmark_results/
    ├── benchmark_cuda_20260105_143022.json    (~8 KB)
    └── benchmark_rocm_20260105_095927.json    (~8 KB)
```

**Total disk usage**: ~16 KB for both platforms

### No Checkpoint Directory
The `/checkpoints` directory specified in the recipe will NOT be created or used.

## Benefits

1. **Fast**: No time wasted saving large checkpoints
2. **Clean**: No clutter from intermediate files
3. **Portable**: Small JSON files easy to transfer
4. **Focused**: Only profiling data for comparison

## If You Need Full Logging

To re-enable full logging and checkpointing, comment out these lines:

```python
# recipe.log.ckpt = None              # Uncomment to enable checkpoints
# recipe.log.tensorboard = None       # Uncomment to enable TensorBoard
# recipe.trainer.enable_checkpointing = False  # Change to True
```

## Verifying What's Saved

### Check Benchmark Results
```bash
ls -lh benchmark_results/
```

### View Benchmark Contents
```bash
cat benchmark_results/benchmark_*.json | python3 -m json.tool
```

### Verify No Checkpoints
```bash
# Should be empty or not exist
ls /checkpoints/ 2>/dev/null || echo "No checkpoint directory (correct!)"
```

## Size Comparison

| Item | Saved? | Typical Size |
|------|--------|--------------|
| Benchmark JSON | ✅ YES | 5-10 KB |
| Model Checkpoint | ❌ NO | 8-16 GB |
| Optimizer State | ❌ NO | 16-32 GB |
| TensorBoard Events | ❌ NO | 100-500 MB |
| Total | | **~8 KB** |

**Space saved**: ~30+ GB per run! 🎉

## Summary

**You save**: Lightweight benchmark profiles (~8 KB JSON files)  
**You skip**: Everything else (checkpoints, logs, intermediate files)  
**Result**: Fast, clean benchmarking runs with just the profiling data you need

---

**Modified files**:
- `pretrain_llama.py`
- `pretrain_qwen.py`
- `pretrain_mistral.py`

All configured to save only benchmark results!

