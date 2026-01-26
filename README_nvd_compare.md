# NVIDIA GPU Benchmark Visualization

The `nvd_compare.py` script visualizes and compares NVIDIA GPU benchmark results across different frameworks.

## Overview

The script analyzes training performance data from multiple frameworks:
- **Megatron** (native Megatron-LM)
- **Transformers** (HuggingFace Transformers with DDP)
- **DeepSpeed** (Microsoft DeepSpeed)
- **NeMo** (NVIDIA NeMo Framework)

## Usage

### Basic Usage

```bash
python3 nvd_compare.py
```

This will:
1. Load all `train_*.json` files with `platform='nvd'` from the `./output` directory
2. Generate visualizations comparing all frameworks and models
3. Save the plot as `nvd_compare.png` in the output directory
4. Print detailed performance comparison tables

### Advanced Usage

```bash
# Specify custom results directory
python3 nvd_compare.py --results-dir /path/to/results

# Specify custom output filename
python3 nvd_compare.py --output my_comparison.png

# Get help
python3 nvd_compare.py --help
```

## Expected Data Format

The script expects JSON files with the following naming convention:
- `train_{framework}_{model}.json`
  - Examples: `train_mega_llama.json`, `train_tran_qwen.json`

Each JSON file should contain:
```json
{
  "platform": "nvd",
  "gpu_info": { ... },
  "training_config": { ... },
  "performance_metrics": {
    "tokens_per_second_per_gpu": float,
    "avg_step_time_seconds": float,
    "total_time_seconds": float,
    ...
  },
  "step_times": [float, ...],
  "loss_values": [float, ...],
  "learning_rates": [float, ...]
}
```

## Output

### Visualization (`nvd_compare.png`)

The script generates a 2x2 grid of plots:
1. **Top-left**: Per-GPU Throughput (bar chart) - Shows tokens/s/GPU for each framework-model combo
2. **Top-right**: Training Loss Convergence - Shows how loss decreases over training steps
3. **Bottom-left**: Step Duration over Time - Shows execution time for each training step
4. **Bottom-right**: Average Step Time (bar chart) - Compares average step times (lower is better)

### Console Output

The script prints:
- List of loaded benchmarks
- Detailed comparison tables grouped by model (Llama vs Qwen)
- Performance summary with winners for:
  - Highest throughput
  - Fastest step time
  - Performance range analysis

## Example Output

```
====================================================================================================
NVIDIA H100 - FRAMEWORK PERFORMANCE COMPARISON
====================================================================================================

  Hardware: NVIDIA H100 80GB HBM3
  GPUs: 8
  PyTorch: 2.7.0a0+7c8ec84dab.nv25.03
  CUDA: 12.8

----------------------------------------------------------------------------------------------------
LLAMA 3.1 8B COMPARISON
----------------------------------------------------------------------------------------------------

Framework               Tokens/s/GPU   Avg Step Time   Final Loss   Total Time
----------------------------------------------------------------------------------------------------
DeepSpeed                   21,924.6           0.747s      11.8994        375.4s
Megatron                     5,315.5           3.082s      11.7970       1546.2s
Transformers                 6,170.1           2.655s      11.7577       1422.8s

====================================================================================================
PERFORMANCE SUMMARY
====================================================================================================

  Highest Throughput: DeepSpeed QWEN
    23,304.6 tokens/s/GPU

  Fastest Step Time: DeepSpeed QWEN
    0.703 seconds/step
```

## Requirements

- Python 3.7+
- matplotlib
- numpy
- Standard library: json, os, pathlib, argparse

Install dependencies:
```bash
pip install matplotlib numpy
```

## Troubleshooting

### Matplotlib Cache Errors

If you see matplotlib cache directory warnings:
```bash
export MPLCONFIGDIR=/tmp/matplotlib
python3 nvd_compare.py
```

### No Data Found

Ensure your JSON files:
1. Are in the results directory
2. Have `"platform": "nvd"` in the JSON
3. Follow the naming convention `train_{framework}_{model}.json`
4. Contain all required fields (performance_metrics, step_times, loss_values)

### Missing Dependencies

```bash
pip install matplotlib numpy
```

## Related Scripts

- `compare.py` - Cross-platform comparison (NVIDIA vs AMD)
- Training scripts in the project root generate the benchmark data
