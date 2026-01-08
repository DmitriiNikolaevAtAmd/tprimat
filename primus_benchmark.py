#!/usr/bin/env python3
"""
Benchmark callback for Primus training runs on AMD ROCm.

Usage:
    Add this to your Primus YAML config or training script to enable benchmarking.
"""
import sys
sys.path.insert(0, '/workspace/support/week-02/code')

from benchmark_utils import BenchmarkCallback

# Import after adding path
try:
    from lightning.pytorch.callbacks import Callback
except ImportError:
    try:
        from pytorch_lightning.callbacks import Callback
    except ImportError:
        print("Warning: PyTorch Lightning not found, using base Callback")
        Callback = object


def get_primus_benchmark_callback(output_dir: str = "./output", model_name: str = None):
    """
    Get benchmark callback configured for Primus training.
    
    Args:
        output_dir: Directory to save benchmark results
        model_name: Name of the model (e.g., 'llama', 'mixtral', 'qwen')
        
    Returns:
        BenchmarkCallback instance
    """
    return BenchmarkCallback(
        output_dir=output_dir,
        platform="auto",  # Auto-detects ROCm
        model_name=model_name
    )


def add_benchmark_to_trainer(trainer, output_dir: str = "./output", model_name: str = None):
    """
    Add benchmark callback to an existing trainer.
    
    Args:
        trainer: PyTorch Lightning Trainer instance
        output_dir: Directory to save benchmark results
        model_name: Name of the model (e.g., 'llama', 'mixtral', 'qwen')
        
    Example:
        >>> trainer = Trainer(...)
        >>> add_benchmark_to_trainer(trainer, model_name='llama')
        >>> trainer.fit(model, datamodule)
    """
    benchmark_callback = get_primus_benchmark_callback(output_dir, model_name)
    
    if not hasattr(trainer, 'callbacks'):
        trainer.callbacks = []
    
    if trainer.callbacks is None:
        trainer.callbacks = []
    
    trainer.callbacks.append(benchmark_callback)
    print(f"✅ Benchmark callback added - results will be saved to {output_dir}")


if __name__ == "__main__":
    print("""
Primus Benchmark Integration
============================

This callback integrates with your Primus training runs.

Method 1: Add to YAML Config
-----------------------------
Add to your YAML file (e.g., llama3.1_8B-pretrain.yaml):

trainer:
  callbacks:
    - class_path: primus_benchmark.get_primus_benchmark_callback
      init_args:
        output_dir: ./output

Method 2: Modify Training Script
---------------------------------
In your training script, after creating the trainer:

from primus_benchmark import add_benchmark_to_trainer

trainer = Trainer(...)
add_benchmark_to_trainer(trainer, output_dir="./output")
trainer.fit(model, datamodule)

Method 3: Environment Variable
-------------------------------
Set before running:

export BENCHMARK_OUTPUT_DIR="./output"

Then the callback will auto-detect and save results.

Output
------
Results will be saved to:
  ./output/benchmark_rocm_<model>.json  (on AMD/ROCm, e.g., benchmark_rocm_llama.json)
  ./output/benchmark_cuda_<model>.json  (on NVIDIA/CUDA, e.g., benchmark_cuda_llama.json)

With metrics:
  - Platform: amd or nvd
  - GPU info: Device details
  - Software stack: rocm or cuda
  - Throughput: tokens/sec and tokens/sec/GPU
  - Memory usage
  - Step timings

Compare Results
--------------
After running on both AMD and NVIDIA:

  python3 compare_results.py

This will generate:
  - comparison_plot.png (visual charts)
  - comparison_report.md (detailed analysis)
""")

