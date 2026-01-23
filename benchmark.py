#!/usr/bin/env python3
"""
TensorPrimat - LLM Benchmark Suite
"""

import argparse
import glob
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple


def detect_platform() -> Tuple[str, str]:
    """Detect GPU platform. Returns (platform_name, software_stack)"""
    try:
        import torch
        if not torch.cuda.is_available():
            print("⚠️  No CUDA/GPU available")
            return "Unknown", "rocm"
        is_rocm = hasattr(torch.version, 'hip') and torch.version.hip is not None
        platform = "rocm" if is_rocm else "cuda"
        print(f"✓ Detected platform: {platform.upper()}")
        return (platform, platform)
    except ImportError:
        print("⚠️  PyTorch not found")
        return "Unknown", "rocm"


def check_nemo() -> bool:
    try:
        import nemo
        print("✓ NeMo framework detected")
        return True
    except ImportError:
        print("ℹ️  NeMo framework not found")
        return False


def find_log_file(model: str, output_dir: str = "./output") -> Optional[str]:
    """Find log file for a model."""
    env_var = f"{model.upper()}_LOG"
    if env_var in os.environ and os.path.isfile(os.environ[env_var]):
        print(f"  Using log file from ${env_var}: {os.environ[env_var]}")
        return os.environ[env_var]
    
    search_dirs = [d for d in [output_dir, ".", "output", "/workspace/Primus", "/workspace/tprimat"] if os.path.isdir(d)]
    patterns = [f"training_{model}.log", f"{model}_training.log", f"primus_{model}.log", f"*{model}*.log"]
    
    for search_dir in search_dirs:
        for pattern in patterns:
            matches = sorted(glob.glob(os.path.join(search_dir, pattern)), key=os.path.getmtime, reverse=True)
            if matches:
                print(f"  Found log file: {matches[0]}")
                return matches[0]
    print(f"  ⚠️  No log file found for {model}")
    return None


def extract_metrics(log_file: str, model: str, parallel_strategy: str, output_dir: str) -> bool:
    """Extract metrics from Primus log file."""
    from config_loader import load_config
    config = load_config()
    
    output_path = os.path.join(output_dir, f"train_prim_{model}.json")
    print(f"  Extracting metrics to: {output_path}")
    
    # Get num_gpus based on detected platform (default to nvidia if detection fails)
    platform = "nvidia"
    try:
        import torch
        is_rocm = hasattr(torch.version, 'hip') and torch.version.hip is not None
        platform = "amd" if is_rocm else "nvidia"
    except ImportError:
        pass  # Default to nvidia if torch not available
    
    num_gpus = config.hardware.platforms[platform].num_gpus
    
    cmd = [
        "python3", "extract_metrics.py",
        "--log-file", log_file,
        "--model-name", model,
        "--output", output_path,
        "--parallel-strategy", parallel_strategy,
        "--num-gpus", str(num_gpus),
        "--global-batch-size", str(config.training.data.global_batch_size),
        "--sequence-length", str(config.training.data.seq_length),
    ]
    result = subprocess.run(cmd)
    if result.returncode == 0:
        print(f"  ✓ Metrics extracted successfully")
    else:
        print(f"  ✗ Metrics extraction failed")
    return result.returncode == 0


def run_nemo_training(model: str, output_dir: str) -> bool:
    """Run NeMo training script."""
    script = {"llama": "pretrain_llama.py", "qwen": "pretrain_qwen.py"}.get(model)
    if not script or not os.path.isfile(script):
        print(f"  ✗ Training script not found: {script}")
        return False
    
    print(f"  Running NeMo training script: {script}")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    log_file = os.path.join(output_dir, f"training_{model}.log")
    
    with open(log_file, 'w') as log:
        result = subprocess.run(["python3", script], stdout=log, stderr=subprocess.STDOUT)
    
    if result.returncode == 0:
        print(f"  ✓ Training completed successfully")
        try:
            os.remove(log_file)
        except OSError:
            pass
    else:
        print(f"  ✗ Training failed (exit code: {result.returncode})")
    return result.returncode == 0


def run_primus_training(model: str, output_dir: str) -> bool:
    """Run Primus training script."""
    script = f"./run_primus_{model}.sh"
    if not os.path.isfile(script):
        print(f"  ✗ Primus script not found: {script}")
        return False
    
    print(f"  Running Primus training script: {script}")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    result = subprocess.run(["bash", script])
    
    if result.returncode == 0:
        print(f"  ✓ Training completed successfully")
    else:
        print(f"  ✗ Training failed (exit code: {result.returncode})")
    return result.returncode == 0


def run_benchmarks(models: List[str], runs: int, output_dir: str, use_nemo: bool) -> Tuple[List[str], List[str]]:
    """Run training benchmarks."""
    successful, failed = [], []
    framework = "NeMo" if use_nemo else "Primus"
    
    print(f"\n{'='*60}")
    print(f"Running {framework} Training Benchmarks")
    print(f"{'='*60}")
    
    for model_idx, model in enumerate(models):
        print(f"\n[{model_idx + 1}/{len(models)}] Model: {model.upper()}")
        for run in range(1, runs + 1):
            print(f"  Run {run}/{runs}:")
            if use_nemo:
                ok = run_nemo_training(model, output_dir)
            else:
                ok = run_primus_training(model, output_dir)
            
            if ok:
                if run == runs:
                    successful.append(model)
            else:
                failed.append(model)
                break
    
    return successful, failed


def run_extraction(models: List[str], output_dir: str) -> Tuple[List[str], List[str]]:
    """Extract metrics from existing logs."""
    successful, failed = [], []
    parallel_strategy = os.environ.get('PARALLEL', 'unknown')
    
    print(f"\n{'='*60}")
    print(f"Extracting Metrics from Logs")
    print(f"{'='*60}")
    
    for model_idx, model in enumerate(models):
        print(f"\n[{model_idx + 1}/{len(models)}] Model: {model.upper()}")
        log_file = find_log_file(model, output_dir)
        if log_file and extract_metrics(log_file, model, parallel_strategy, output_dir):
            successful.append(model)
        else:
            failed.append(model)
    
    return successful, failed


def main():
    parser = argparse.ArgumentParser(description="TensorPrimat - LLM Benchmark Suite")
    parser.add_argument('--model', choices=['llama', 'qwen', 'all'], default='all')
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--parallel', choices=['maximum_performance', 'truly_identical', 'memory_optimized', 'minimal_communication', 'balanced'], default='minimal_communication')
    parser.add_argument('--output-dir', default='./output')
    args = parser.parse_args()
    
    print("=" * 60)
    print("TensorPrimat - LLM Benchmark Suite")
    print("=" * 60)
    print(f"Models: {args.model}")
    print(f"Runs: {args.runs}")
    print(f"Parallel Strategy: {args.parallel}")
    print(f"Output Directory: {args.output_dir}")
    print()
    
    os.environ['PARALLEL'] = args.parallel
    os.environ['OUTPUT_DIR'] = args.output_dir
    
    models = ['llama', 'qwen'] if args.model == 'all' else [args.model]
    platform, software_stack = detect_platform()
    gpu_available = platform != "Unknown"
    has_nemo = check_nemo() if gpu_available else False
    
    if not gpu_available:
        print("\nMode: Metrics extraction (no GPU detected)")
        successful, failed = run_extraction(models, args.output_dir)
    elif has_nemo:
        print("\nMode: NeMo training benchmarks")
        successful, failed = run_benchmarks(models, args.runs, args.output_dir, use_nemo=True)
    elif software_stack == "rocm":
        print("\nMode: Primus/ROCm training benchmarks")
        successful, failed = run_benchmarks(models, args.runs, args.output_dir, use_nemo=False)
    else:
        print("\nMode: Metrics extraction (CUDA without NeMo)")
        successful, failed = run_extraction(models, args.output_dir)
    
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"✓ Successful: {len(successful)}")
    if successful:
        print(f"  Models: {', '.join(successful)}")
    print(f"✗ Failed: {len(failed)}")
    if failed:
        print(f"  Models: {', '.join(failed)}")
    print()
    
    sys.exit(0 if len(failed) == 0 else 1)


if __name__ == "__main__":
    main()
