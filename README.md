# TensorPrimat

**Unified benchmarking toolkit for comparing AMD vs NVIDIA GPU performance with LLM training**

---

## 🚀 Quick Start

```bash
# Run on any platform (NVIDIA or AMD)
./benchmark.py

# Compare results after running on both platforms
python3 compare_results.py
```

**That's it!** Automatically detects your platform and runs all models.

---

## 📋 What You Get

### Automatic Platform Detection
- ✅ **NVIDIA (CUDA)**: Runs NeMo training scripts
- ✅ **AMD (ROCm)**: Extracts metrics from Primus logs
- ✅ **Smart**: Detects NeMo availability and adapts

### Benchmark Results
- 📄 `output/benchmark_cuda_llama.json`
- 📄 `output/benchmark_cuda_mistral.json`
- 📄 `output/benchmark_cuda_qwen.json`
- 📄 `output/benchmark_rocm_llama.json`
- 📄 `output/benchmark_rocm_mistral.json`
- 📄 `output/benchmark_rocm_qwen.json`

### Comparison Reports
- 📊 `comparison_plot.png` - Visual charts
- 📝 `comparison_report.md` - Detailed analysis
- 🏆 Console output - Quick winner summary

---

## 💻 Installation

### Prerequisites

**Already installed with NeMo/Primus:**
- Python 3.8+
- PyTorch 2.x+
- CUDA or ROCm

**For visualization (optional):**
```bash
pip install matplotlib numpy
```

Or use the included requirements:
```bash
pip install -r requirements.txt
```

---

## 🎯 Usage

### Basic Usage

```bash
# Run all models (default)
./benchmark.py

# Run specific model
./benchmark.py --model llama
./benchmark.py --model mistral
./benchmark.py --model qwen

# Run multiple times for better statistics
./benchmark.py --runs 3

# Get help
./benchmark.py --help
```

### AMD/Primus Automatic Log Detection

The script **automatically searches for Primus training logs** in multiple locations:
- Current directory
- `output/` directory
- `/workspace/Primus/`
- `/workspace/tprimat/`

Just run `./benchmark.py` and it will find your logs automatically!

**Optional: Specify log paths explicitly**

```bash
# If needed, you can provide specific log file paths
LLAMA_LOG=/path/to/llama.log \
MISTRAL_LOG=/path/to/mistral.log \
QWEN_LOG=/path/to/qwen.log \
./benchmark.py
```

### Complete AMD vs NVIDIA Workflow

```bash
# 1. Run on NVIDIA system
./benchmark.py

# 2. Run on AMD system  
./benchmark.py

# 3. Compare results (on either system)
python3 compare_results.py

# 4. View results
open comparison_plot.png       # macOS
xdg-open comparison_plot.png   # Linux
cat comparison_report.md
```

---

## 📊 What Gets Measured

### Performance Metrics
- **Tokens/sec/GPU**: Per-GPU efficiency (primary metric)
- **Total Throughput**: System-wide tokens per second
- **Step Time**: Time per training iteration
- **Steps per Second**: Training speed

### Memory Metrics
- **Average Memory**: Typical GPU memory usage
- **Peak Memory**: Maximum allocation
- **Reserved Memory**: Total PyTorch reservation

### System Information
- **Platform**: `nvd` (NVIDIA) or `amd` (AMD)
- **Software Stack**: `cuda` or `rocm`
- **Software Version**: CUDA/ROCm version string
- **PyTorch Version**: Full version
- **GPU Model**: Device name and specs
- **GPU Cores**: CUDA cores or Stream Processors

---

## 🛠️ Core Components

### Main Scripts

| File | Purpose |
|------|---------|
| **`benchmark.py`** | Main entrypoint - runs everything |
| **`compare_results.py`** | Generate comparison reports |
| **`benchmark_utils.py`** | Core benchmarking framework |
| **`extract_primus_metrics.py`** | Extract from Primus logs |

### Training Scripts (NeMo)

| File | Model |
|------|-------|
| `pretrain_llama.py` | Llama 3.1 8B |
| `pretrain_mistral.py` | Mistral 7B |
| `pretrain_qwen.py` | Qwen 2.5 7B |

All include automatic benchmarking via `BenchmarkCallback`.

---

## 🔧 Advanced Usage

### How Automatic Log Detection Works

On AMD platforms without NeMo, the benchmark script automatically searches for Primus training logs:

**Search Strategy:**
1. **Environment Variables**: Checks `LLAMA_LOG`, `MISTRAL_LOG`, `QWEN_LOG`
2. **Standard Filenames**: Looks for `training_llama.log`, `training_mistral.log`, etc.
3. **Pattern Matching**: Searches for files like `*llama*.log`, `primus_*llama*.log`
4. **Multiple Directories**: 
   - Current directory (`.`)
   - Output directory (`output/`)
   - Primus workspace (`/workspace/Primus/`)
   - TensorPrimat workspace (`/workspace/tprimat/`)
5. **Content-Based Search**: As a fallback, scans `.log` and `.txt` files for model-specific keywords

**Result**: You don't need to manually specify log paths in most cases—just run `./benchmark.py`!

### Running Primus Training

If you need to generate new Primus logs:

```bash
# Navigate to Primus
cd /workspace/Primus

# Run Llama training
export EXP=examples/megatron/configs/MI300X/llama3.1_8B-pretrain.yaml
bash ./examples/run_pretrain.sh --train_iters 10 2>&1 | tee /workspace/tprimat/training_llama.log

# Run Mistral training
export EXP=examples/megatron/configs/MI300X/mistral_7B-pretrain.yaml
bash ./examples/run_pretrain.sh --train_iters 10 2>&1 | tee /workspace/tprimat/training_mistral.log

# Run Qwen training
export EXP=examples/megatron/configs/MI300X/qwen2.5_7B-pretrain.yaml
bash ./examples/run_pretrain.sh --train_iters 10 2>&1 | tee /workspace/tprimat/training_qwen.log

# Extract metrics
cd /workspace/tprimat
./benchmark.py
```

### Manual Log Extraction

```bash
python3 extract_primus_metrics.py \
    --log-file training_llama.log \
    --model-name llama \
    --num-gpus 8 \
    --global-batch-size 128 \
    --sequence-length 2048
```

### Customizing Benchmarks

Edit training scripts to change parameters:

```python
# In pretrain_llama.py (or mistral/qwen)

# Change number of steps
recipe.trainer.max_steps = 20  # Default is 10

# Change batch size
recipe.data.global_batch_size = 256  # Default is 8

# Customize output directory
benchmark_callback = BenchmarkCallback(
    output_dir="./my_results",
    model_name="llama"
)
```

---

## 📁 Output Structure

```
output/
├── benchmark_cuda_llama.json      # NVIDIA Llama results
├── benchmark_cuda_mistral.json    # NVIDIA Mistral results
├── benchmark_cuda_qwen.json       # NVIDIA Qwen results
├── benchmark_rocm_llama.json      # AMD Llama results
├── benchmark_rocm_mistral.json    # AMD Mistral results
├── benchmark_rocm_qwen.json       # AMD Qwen results
├── comparison_plot.png            # Visual comparison charts
└── comparison_report.md           # Detailed analysis report
```

**Note**: Each model creates a **single file per platform** that gets overwritten on each run.

---

## 🎨 JSON Output Format

```json
{
  "platform": "nvd",
  "gpu_info": {
    "device_name": "NVIDIA H100 80GB HBM3",
    "device_count": 8,
    "gpu_cores": 16896,
    "software_stack": "cuda",
    "software_version": "12.8",
    "pytorch_version": "2.7.0a0+7c8ec84dab.nv25.03"
  },
  "performance_metrics": {
    "avg_step_time_seconds": 1.602,
    "tokens_per_second": 40918.340,
    "tokens_per_second_per_gpu": 5114.792,
    "steps_per_second": 0.624
  },
  "memory_metrics": {
    "avg_memory_allocated_gb": 22.295,
    "peak_memory_allocated_gb": 22.295
  }
}
```

All float values are automatically rounded to 3 decimal places.

---

## 🤖 How It Works

### On NVIDIA (NeMo)

1. Detects NeMo is installed
2. Runs training scripts (`pretrain_llama.py`, etc.)
3. `BenchmarkCallback` collects metrics during training
4. Saves to `output/benchmark_cuda_{model}.json`

### On AMD (Primus)

1. Detects NeMo is NOT installed
2. Searches for log files:
   - Environment variables (`LLAMA_LOG`, `MISTRAL_LOG`, `QWEN_LOG`)
   - Standard filenames (`training_llama.log`, etc.)
   - Pattern matching (`*llama*.log`)
   - Content-based search (scans all `.log` and `.txt` files)
3. Extracts metrics using `extract_primus_metrics.py`
4. Saves to `output/benchmark_rocm_{model}.json`

---

## 🐛 Troubleshooting

### "No GPU detected"

```bash
# Check GPU availability
nvidia-smi  # NVIDIA
rocm-smi    # AMD

# Verify PyTorch
python3 -c "import torch; print(torch.cuda.is_available())"
```

### "No log files found" (AMD/Primus)

**Solution 1**: Provide explicit paths
```bash
LLAMA_LOG=/path/to/your/log.txt ./benchmark.py
```

**Solution 2**: Copy logs to current directory
```bash
cp /path/to/logs/*.log .
# Name them: training_llama.log, training_mistral.log, training_qwen.log
./benchmark.py
```

**Solution 3**: Run Primus training with log capture
```bash
cd /workspace/Primus
bash ./examples/run_pretrain.sh ... 2>&1 | tee /workspace/tprimat/training_llama.log
```

### "Module not found: matplotlib"

```bash
pip install matplotlib numpy
```

### Out of Memory

Reduce batch size in training scripts:
```python
recipe.data.global_batch_size = 4  # Reduce from 8
```

---

## 📞 Quick Reference

| Command | Purpose |
|---------|---------|
| `./benchmark.py` | Run all models on current platform |
| `./benchmark.py --model llama` | Run single model |
| `./benchmark.py --runs 3` | Run 3 times per model |
| `python3 compare_results.py` | Generate comparison report |
| `./benchmark.py --help` | Show all options |

| Environment Variable | Purpose |
|---------------------|---------|
| `LLAMA_LOG` | Path to Llama training log |
| `MISTRAL_LOG` | Path to Mistral training log |
| `QWEN_LOG` | Path to Qwen training log |

---

## 🎯 Workflow Diagram

```
NVIDIA (NeMo):
  ./benchmark.py  →  pretrain_llama.py    →  benchmark_cuda_llama.json
                  →  pretrain_mistral.py  →  benchmark_cuda_mistral.json
                  →  pretrain_qwen.py     →  benchmark_cuda_qwen.json

AMD (Primus):
  training_llama.log  →  extract_primus_metrics.py  →  benchmark_rocm_llama.json
  training_mistral.log →  extract_primus_metrics.py  →  benchmark_rocm_mistral.json
  training_qwen.log   →  extract_primus_metrics.py  →  benchmark_rocm_qwen.json

Both Platforms:
  output/benchmark_*.json  →  compare_results.py  →  comparison_plot.png
                                                  →  comparison_report.md
```

---

## 💡 Best Practices

1. ✅ **Run multiple times** (3-5 runs) for statistical significance
2. ✅ **Use identical configurations** on both platforms
3. ✅ **Close other GPU applications** during benchmarking
4. ✅ **Let warmup complete** (first step is automatically excluded)
5. ✅ **Save results** before running again (or they'll be overwritten)
6. ✅ **Document your setup** (GPU model, driver versions, etc.)

---

## 📚 Additional Documentation

| File | Content |
|------|---------|
| [QUICK_START.md](QUICK_START.md) | 5-minute tutorial |
| [BENCHMARK_README.md](BENCHMARK_README.md) | Complete reference |
| [WORKFLOW.md](WORKFLOW.md) | Architecture diagrams |
| [PRIMUS_INTEGRATION.md](PRIMUS_INTEGRATION.md) | Primus-specific guide |

---

## 🆕 Features

- ✅ **Single Python entrypoint** - `./benchmark.py` does everything
- ✅ **Automatic platform detection** - works on NVIDIA and AMD
- ✅ **Smart log discovery** - finds logs automatically
- ✅ **Model-based filenames** - single file per model
- ✅ **Automatic rounding** - clean 3-decimal JSON output
- ✅ **Environment variable support** - specify log paths easily
- ✅ **Beautiful CLI output** - colored, formatted, clear
- ✅ **Comprehensive metrics** - tokens/sec/GPU, memory, timing
- ✅ **Fair comparison** - identical configurations guaranteed

---

## 🔄 Version History

**Version 2.0** (Current)
- Python main entrypoint
- Automatic platform detection
- Smart log file discovery
- Model-based output filenames
- Improved metrics (tokens/sec/GPU)
- Environment variable configuration

**Version 1.0**
- Initial shell script implementation
- Basic benchmarking functionality

---

**Version**: 2.0  
**Compatible with**: NeMo 24.x+, PyTorch 2.x+  
**Platforms**: NVIDIA CUDA, AMD ROCm  
**Python**: 3.8+

**Happy Benchmarking! 🎉**
