# Fair Comparison Configuration

## ✅ Updated for Fair AMD vs NVIDIA Comparison

All NeMo training scripts have been updated to match the Primus configuration exactly.

## Configuration Summary

### Identical Settings on Both Platforms

```python
# Training Configuration
num_nodes = 1
num_gpus_per_node = 8
max_steps = 10

# Parallelism Strategy
tensor_model_parallel_size = 4
pipeline_model_parallel_size = 1  # (2 for Qwen)

# Data Configuration
micro_batch_size = 1
global_batch_size = 128  # ✅ NOW MATCHES PRIMUS
sequence_length = 2048   # ✅ NOW MATCHES PRIMUS

# Precision
fp8 = "hybrid"
fp8_param = True
```

### Workload Calculation

Both platforms now process **identical workloads**:

```
Tokens per step = global_batch_size × sequence_length
                = 128 × 2,048
                = 262,144 tokens per step
```

## Before vs After

### ❌ Before (Unfair)

| Platform | Batch Size | Seq Length | Tokens/Step |
|----------|------------|------------|-------------|
| AMD      | 128        | 2,048      | 262,144     |
| NVIDIA   | 8          | 8,192      | 65,536      |
| **Difference** | **16x** | **4x** | **4x** |

### ✅ After (Fair)

| Platform | Batch Size | Seq Length | Tokens/Step |
|----------|------------|------------|-------------|
| AMD      | 128        | 2,048      | 262,144     |
| NVIDIA   | 128        | 2,048      | 262,144     |
| **Difference** | **Identical** | **Identical** | **Identical** |

## What Changed

### Updated Files

1. **pretrain_llama.py**
   - `global_batch_size: 8 → 128`
   - Added `seq_length = 2048`

2. **pretrain_mistral.py**
   - `global_batch_size: 8 → 128`
   - Added `seq_length = 2048`

3. **pretrain_qwen.py**
   - `global_batch_size: 8 → 128`
   - Added `seq_length = 2048`

4. **BENCHMARK_README.md**
   - Updated example configuration

5. **PROFILING_SUMMARY.md**
   - Updated all model configurations

## How to Run Fair Comparison

### On NVIDIA (NeMo)

```bash
# Run with updated configuration
python pretrain_llama.py  # or pretrain_mistral.py, pretrain_qwen.py

# Results will be saved to:
# ./output/benchmark_nvd_TIMESTAMP.json
```

### On AMD (Primus)

```bash
# Ensure your Primus command matches:
EXP=examples/megatron/configs/MI300X/llama3.1_8B-pretrain.yaml \
bash ./examples/run_pretrain.sh \
    --train_iters 10 \
    --fp8 hybrid \
    --micro_batch_size 1 \
    --global_batch_size 128 \
    --seq_length 2048

# Or use the extraction script:
python extract_primus_metrics.py \
    --log-file primus_training.log \
    --output ./output/benchmark_amd.json \
    --num-gpus 8 \
    --global-batch-size 128 \
    --sequence-length 2048
```

### Compare Results

```bash
# After running on both platforms
python compare_results.py

# Verify configurations match:
grep -E "global_batch_size|sequence_length" ./output/benchmark_*.json
```

## Verification Checklist

Before comparing results, ensure:

- [ ] Both runs use `global_batch_size = 128`
- [ ] Both runs use `sequence_length = 2048`
- [ ] Both runs use `num_gpus = 8`
- [ ] Both runs use `max_steps = 10`
- [ ] Both runs use `fp8 = "hybrid"`
- [ ] Both runs use same parallelism settings (TP=4, PP=1 or 2)

## Expected Memory Usage

With the new configuration (batch=128, seq=2048):

- **AMD MI300X**: ~118 GB per GPU (you've already tested this)
- **NVIDIA H100**: ~30-40 GB per GPU (estimated, needs verification)

Both platforms should have sufficient memory for this configuration.

## Notes

- The increased batch size (8 → 128) means more gradient accumulation steps
- This is more representative of real-world training scenarios
- The comparison will now measure actual compute performance, not different workloads
- You may need to re-run the NVIDIA benchmarks with the new configuration

## Next Steps

1. ✅ Configuration updated
2. ⏳ Re-run NVIDIA benchmarks with new config
3. ⏳ Verify both JSON files have matching parameters
4. ⏳ Run `compare_results.py` for fair comparison
5. ⏳ Review updated comparison report

---

*Last updated: 2026-01-07*
*Configuration standardized for fair AMD vs NVIDIA comparison*

