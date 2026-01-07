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


def get_primus_benchmark_callback(output_dir: str = "./outs"):
    """
    Get benchmark callback configured for Primus training.
    
    Args:
        output_dir: Directory to save benchmark results
        
    Returns:
        BenchmarkCallback instance
    """
    return BenchmarkCallback(
        output_dir=output_dir,
        platform="auto"  # Auto-detects ROCm
    )


def add_benchmark_to_trainer(trainer, output_dir: str = "./outs"):
    """
    Add benchmark callback to an existing trainer.
    
    Args:
        trainer: PyTorch Lightning Trainer instance
        output_dir: Directory to save benchmark results
        
    Example:
        >>> trainer = Trainer(...)
        >>> add_benchmark_to_trainer(trainer)
        >>> trainer.fit(model, datamodule)
    """
    benchmark_callback = get_primus_benchmark_callback(output_dir)
    
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
        output_dir: ./outs

Method 2: Modify Training Script
---------------------------------
In your training script, after creating the trainer:

from primus_benchmark import add_benchmark_to_trainer

trainer = Trainer(...)
add_benchmark_to_trainer(trainer, output_dir="./outs")
trainer.fit(model, datamodule)

Method 3: Environment Variable
-------------------------------
Set before running:

export BENCHMARK_OUTPUT_DIR="./outs"

Then the callback will auto-detect and save results.

Output
------
Results will be saved to:
  ./outs/benchmark_rocm_TIMESTAMP.json

With metrics:
  - Platform: ROCm
  - GPU info: MI300X details
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

