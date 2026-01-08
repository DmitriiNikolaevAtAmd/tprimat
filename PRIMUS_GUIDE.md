# Primus Training Scripts Guide

Complete guide for running Primus training on AMD GPUs with automated benchmarking.

---

## Quick Start

### Run Single Model

```bash
# Llama 3.1 8B
./run_primus_llama.sh

# Mistral 7B
./run_primus_mistral.sh

# Qwen 2.5 7B
./run_primus_qwen.sh
```

### Run All Models

```bash
./run_primus_all.sh
```

This will run Llama, Mistral, and Qwen in sequence with automatic:
- ✅ Log capture
- ✅ Metric extraction
- ✅ Benchmark JSON generation

---

## Configuration

### Environment Variables

```bash
# Primus installation path (default: /workspace/Primus)
export PRIMUS_PATH=/path/to/Primus

# Number of training iterations (default: 10)
export TRAIN_ITERS=10

# Then run:
./run_primus_llama.sh
```

### Script Parameters

Each script automatically sets:
- **Config file**: Model-specific YAML from Primus
- **Output directory**: `./output/`
- **Log files**: `training_<model>.log` + timestamped backup
- **Batch size**: 128 (for metric extraction)
- **Sequence length**: 2048
- **Number of GPUs**: 8

---

## What Happens Automatically

When you run `./run_primus_llama.sh`:

```
1. ✅ Validates Primus installation
2. ✅ Checks config file exists
3. ✅ Creates output directory
4. ✅ Runs Primus training
5. ✅ Captures logs (two copies)
6. ✅ Extracts metrics automatically
7. ✅ Generates benchmark JSON
8. ✅ Shows next steps
```

---

## Output Files

### Log Files

```
output/training_llama.log              # Primary log (overwritten each run)
output/primus_training_llama_<timestamp>.log  # Backup (timestamped)
```

### Benchmark Files

```
output/benchmark_rocm_llama.json       # Extracted metrics
output/benchmark_rocm_mistral.json
output/benchmark_rocm_qwen.json
```

---

## Usage Examples

### Basic Usage

```bash
# Run Llama training (uses defaults)
./run_primus_llama.sh
```

### Custom Training Iterations

```bash
# Run 50 iterations instead of 10
TRAIN_ITERS=50 ./run_primus_llama.sh
```

### Custom Primus Path

```bash
# If Primus is in non-standard location
PRIMUS_PATH=/custom/path/to/Primus ./run_primus_llama.sh
```

### Run All Models with Custom Settings

```bash
# Set globally for all runs
export PRIMUS_PATH=/custom/path/to/Primus
export TRAIN_ITERS=20

# Run all models
./run_primus_all.sh
```

---

## Complete Workflow

### On AMD System

```bash
# 1. Run all Primus models
./run_primus_all.sh

# Output:
# ✅ output/benchmark_rocm_llama.json
# ✅ output/benchmark_rocm_mistral.json
# ✅ output/benchmark_rocm_qwen.json
```

### On NVIDIA System

```bash
# 2. Run all NeMo models
./benchmark.py

# Output:
# ✅ output/benchmark_cuda_llama.json
# ✅ output/benchmark_cuda_mistral.json
# ✅ output/benchmark_cuda_qwen.json
```

### Compare Results

```bash
# 3. Generate comparison (on either system)
python3 compare_results.py

# Output:
# ✅ comparison_plot.png
# ✅ comparison_report.md

# 4. View enhanced metrics
python3 compare_with_enhanced_metrics.py
```

---

## Customization

### Modify Training Configuration

If you need different settings, edit the Primus config files:

```bash
cd /workspace/Primus/examples/megatron/configs/MI300X/

# Edit model configs:
vi llama3.1_8B-pretrain.yaml
vi mistral_7B-pretrain.yaml
vi qwen2.5_7B-pretrain.yaml
```

**Common parameters to adjust:**
```yaml
# Parallelism
tensor_model_parallel_size: 1    # Model parallelism
pipeline_model_parallel_size: 1

# Batch sizes
micro_batch_size: 1
global_batch_size: 128

# Precision
precision: bf16  # or fp16, fp8

# Sequence length
seq_length: 2048
```

### Modify Metric Extraction

If your training uses different settings, update the extraction call in the scripts:

```bash
python3 extract_primus_metrics.py \
    --log-file "$LOG_FILE" \
    --model-name "$MODEL" \
    --num-gpus 16 \              # Change if using different GPU count
    --global-batch-size 256 \     # Change if using different batch size
    --sequence-length 4096        # Change if using different seq length
```

---

## Troubleshooting

### Config File Not Found

```bash
❌ Config file not found: /workspace/Primus/examples/megatron/configs/MI300X/llama3.1_8B-pretrain.yaml
```

**Solutions:**
1. Check Primus installation: `ls /workspace/Primus`
2. Set correct path: `export PRIMUS_PATH=/actual/path`
3. List available configs: `ls /workspace/Primus/examples/megatron/configs/MI300X/`

### Training Fails

**Check the log file:**
```bash
cat output/training_llama.log
```

**Common issues:**
- OOM (out of memory) → Reduce batch size in config
- Missing dependencies → Check Primus installation
- GPU not available → Verify ROCm setup

### Metric Extraction Fails

```bash
❌ Metric extraction failed
   Check the log file manually: output/training_llama.log
```

**Solutions:**
1. Verify log has timing information:
   ```bash
   grep "elapsed time per iteration" output/training_llama.log
   ```

2. Manually extract:
   ```bash
   python3 extract_primus_metrics.py \
       --log-file output/training_llama.log \
       --model-name llama \
       --num-gpus 8 \
       --global-batch-size 128 \
       --sequence-length 2048
   ```

3. Check log format matches expected pattern (see `extract_primus_metrics.py`)

### Logs Not Found for Comparison

If running `./benchmark.py` can't find logs:

```bash
# Logs must be named correctly:
ls -l output/training_*.log

# Or use environment variables:
LLAMA_LOG=output/my_llama.log ./benchmark.py
```

---

## Advanced Usage

### Parallel Training (Multiple GPUs/Nodes)

Edit the Primus config to use multiple nodes:

```yaml
# In config file
num_nodes: 2
devices: 8  # GPUs per node
```

Then update the metric extraction:
```bash
--num-gpus 16  # 2 nodes * 8 GPUs
```

### Fair Comparison with NVIDIA (TP=4)

To match NVIDIA's configuration:

1. Create new config:
```bash
cd /workspace/Primus/examples/megatron/configs/MI300X/
cp llama3.1_8B-pretrain.yaml llama3.1_8B-pretrain-tp4.yaml
```

2. Edit to match NVIDIA:
```yaml
tensor_model_parallel_size: 4  # Match NVIDIA
pipeline_model_parallel_size: 1
micro_batch_size: 1
global_batch_size: 128
precision: bf16  # or fp8 if supported
```

3. Update script to use new config (modify `CONFIG_FILE` in script)

4. Run and compare both configurations

See `COMPARISON_METHODOLOGY.md` for details.

---

## Script Reference

### Available Scripts

| Script | Purpose |
|--------|---------|
| `run_primus_llama.sh` | Run Llama 3.1 8B training |
| `run_primus_mistral.sh` | Run Mistral 7B training |
| `run_primus_qwen.sh` | Run Qwen 2.5 7B training |
| `run_primus_all.sh` | Run all models in sequence |

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PRIMUS_PATH` | `/workspace/Primus` | Primus installation directory |
| `TRAIN_ITERS` | `10` | Number of training iterations |

### Automatic Features

- ✅ **Error checking**: Validates paths and configs
- ✅ **Log capture**: Saves both primary and backup logs
- ✅ **Auto-extraction**: Runs metric extraction automatically
- ✅ **Next steps**: Shows what to do after completion
- ✅ **Exit codes**: Returns proper status for automation

---

## Integration with TensorPrimat

The Primus scripts integrate seamlessly with TensorPrimat:

```bash
# 1. Run Primus (AMD)
./run_primus_all.sh

# 2. Run NeMo (NVIDIA)  
./benchmark.py

# 3. Compare
python3 compare_results.py
python3 compare_with_enhanced_metrics.py
```

All outputs use consistent formats for easy comparison.

---

## Best Practices

### Before Training

1. ✅ Check Primus path: `echo $PRIMUS_PATH`
2. ✅ Verify configs exist: `ls /workspace/Primus/examples/megatron/configs/MI300X/`
3. ✅ Clean output dir: `mkdir -p output`
4. ✅ Check GPU availability: `rocm-smi`

### During Training

- ✅ Monitor progress: `watch -n 1 rocm-smi`
- ✅ Don't interrupt (Ctrl+C leaves GPU memory allocated)
- ✅ Check logs periodically: `tail -f output/training_llama.log`

### After Training

1. ✅ Verify JSON created: `ls output/benchmark_rocm_*.json`
2. ✅ Check metrics look reasonable: `cat output/benchmark_rocm_llama.json`
3. ✅ Run comparison if have both AMD and NVIDIA results

---

*Last Updated: January 2026*
