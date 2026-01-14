#!/usr/bin/env python3
"""
TensorPrimat - Main Benchmark Entrypoint

Automatically benchmarks LLM training on NVIDIA (NeMo) or AMD (Primus).

Usage:
    ./benchmark.py                  # Run all models (default)
    ./benchmark.py --model llama    # Run single model
    ./benchmark.py --runs 3         # Run 3 times
    
Environment variables (for AMD/Primus):
    LLAMA_LOG=/path/to/llama.log
    QWEN_LOG=/path/to/qwen.log
"""

import argparse
import glob
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple


class Colors:
    """ANSI color codes for terminal output"""
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    CYAN = '\033[0;36m'
    NC = '\033[0m'  # No Color


def detect_platform() -> Tuple[str, str, str]:
    """
    Detect GPU platform and software stack.
    
    Works for both NVIDIA (CUDA) and AMD (ROCm) GPUs.
    Note: torch.cuda.is_available() returns True for both CUDA and ROCm
    thanks to HIP compatibility layer.
    
    If no GPU is detected, defaults to log analysis mode.
    
    Returns:
        (platform_name, software_stack, color)
        e.g., ("AMD (ROCm)", "rocm", Colors.RED) or ("NVD (CUDA)", "cuda", Colors.GREEN)
    """
    try:
        import torch
        
        # torch.cuda.is_available() works for both CUDA and ROCm
        if not torch.cuda.is_available():
            # No GPU detected - will use log analysis mode
            print(f"{Colors.YELLOW}⚠️  No GPU detected - Log analysis mode only{Colors.NC}")
            # Try to infer platform from existing logs or default to rocm
            return "Unknown", "rocm", Colors.YELLOW
        
        # Check if ROCm or CUDA
        # ROCm sets torch.version.hip, CUDA sets torch.version.cuda
        is_rocm = hasattr(torch.version, 'hip') and torch.version.hip is not None
        
        if is_rocm:
            return "AMD (ROCm)", "rocm", Colors.RED
        else:
            return "NVD (CUDA)", "cuda", Colors.GREEN
            
    except ImportError:
        print(f"{Colors.YELLOW}⚠️  PyTorch not found - Log analysis mode only{Colors.NC}")
        # Default to rocm for log analysis
        return "Unknown", "rocm", Colors.YELLOW


def check_nemo() -> bool:
    """Check if NeMo is available"""
    try:
        import nemo
        return True
    except ImportError:
        return False


def find_log_file(model: str) -> Optional[str]:
    """
    Find log file for a given model.
    
    Searches for:
    1. Environment variable (LLAMA_LOG, QWEN_LOG)
    2. Standard filenames (training_llama.log, etc.)
    3. Pattern matching in current directory (*llama*.log)
    4. Pattern matching in output directory
    5. Pattern matching in /workspace/Primus (if exists)
    6. Content-based search in all .log/.txt files
    
    Args:
        model: Model name (llama, qwen)
        
    Returns:
        Path to log file or None
    """
    # Check environment variable
    env_var = f"{model.upper()}_LOG"
    if env_var in os.environ:
        log_path = os.environ[env_var]
        if os.path.isfile(log_path):
            print(f"{Colors.GREEN}✓{Colors.NC} Using log from ${env_var}: {log_path}")
            return log_path
    
    # Define search directories
    search_dirs = [
        ".",                                    # Current directory
        "output",                               # Output directory
        "/workspace/Primus",                    # Primus workspace
        "/workspace/Primus/logs",               # Primus logs directory
        "/workspace/tprimat",                   # TensorPrimat workspace
        "/workspace/tprimat/output",            # TensorPrimat output directory
    ]
    
    # Filter to existing directories
    search_dirs = [d for d in search_dirs if os.path.isdir(d)]
    
    # Try standard filenames in all search directories
    patterns = [
        f"training_{model}.log",
        f"{model}_training.log",
        f"primus_{model}.log",
        f"primus_training_*{model}*.log",
        f"*{model}*.log",
    ]
    
    for search_dir in search_dirs:
        for pattern in patterns:
            full_pattern = os.path.join(search_dir, pattern)
            matches = sorted(glob.glob(full_pattern), key=os.path.getmtime, reverse=True)
            if matches:
                print(f"{Colors.GREEN}✓{Colors.NC} Found log by pattern: {matches[0]}")
                return matches[0]
    
    # Search all .log and .txt files by content
    print(f"{Colors.YELLOW}→{Colors.NC} Searching log files by content for {model} training...")
    
    all_log_files = []
    for search_dir in search_dirs:
        all_log_files.extend(glob.glob(os.path.join(search_dir, "*.log")))
        all_log_files.extend(glob.glob(os.path.join(search_dir, "*.txt")))
    
    # Remove duplicates and sort by modification time (newest first)
    all_log_files = sorted(set(all_log_files), key=os.path.getmtime, reverse=True)
    
    for log_file in all_log_files:
        if not os.path.isfile(log_file):
            continue
            
        try:
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(10000)  # Read first 10KB
                
                # Check for model-specific strings
                content_lower = content.lower()
                
                # More specific checks
                if model == "llama" and "llama" in content_lower and ("8b" in content_lower or "pretrain" in content_lower):
                    print(f"{Colors.GREEN}✓{Colors.NC} Found by content: {log_file}")
                    return log_file
                elif model == "qwen" and "qwen" in content_lower and ("pretrain" in content_lower or "2.5" in content_lower):
                    print(f"{Colors.GREEN}✓{Colors.NC} Found by content: {log_file}")
                    return log_file
        except Exception:
            continue
    
    return None


def extract_metrics(log_file: str, model: str, num_gpus: int = 8, 
                   global_batch_size: int = 128, sequence_length: int = 2048,
                   parallel_strategy: str = "unknown", output_dir: str = "./output") -> bool:
    """
    Extract metrics from Primus log file.
    
    Args:
        log_file: Path to log file
        model: Model name
        num_gpus: Number of GPUs
        global_batch_size: Global batch size
        sequence_length: Sequence length
        parallel_strategy: Parallelism strategy name
        output_dir: Output directory for JSON results
        
    Returns:
        True if successful, False otherwise
    """
    # Determine output path
    output_path = os.path.join(output_dir, f"benchmark_rocm_{model}.json")
    
    cmd = [
        "python3", "extract_primus_metrics.py",
        "--log-file", log_file,
        "--model-name", model,
        "--output", output_path,
        "--num-gpus", str(num_gpus),
        "--global-batch-size", str(global_batch_size),
        "--sequence-length", str(sequence_length),
        "--parallel-strategy", parallel_strategy,
    ]
    
    result = subprocess.run(cmd)
    return result.returncode == 0


def run_nemo_training(model: str, output_dir: str = "./output") -> bool:
    """
    Run NeMo training script for a model.
    
    Args:
        model: Model name (llama, qwen)
        output_dir: Directory to save logs and results
        
    Returns:
        True if successful, False otherwise
    """
    scripts = {
        "llama": "pretrain_llama.py",
        "qwen": "pretrain_qwen.py",
    }
    
    script = scripts.get(model)
    if not script:
        print(f"{Colors.RED}❌ Unknown model: {model}{Colors.NC}")
        return False
    
    if not os.path.isfile(script):
        print(f"{Colors.RED}❌ Script not found: {script}{Colors.NC}")
        return False
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Log file path
    log_file = os.path.join(output_dir, f"training_{model}.log")
    
    # Run training and capture output to log file
    print(f"{Colors.CYAN}→ Logging to: {log_file}{Colors.NC}")
    with open(log_file, 'w') as log:
        result = subprocess.run(
            ["python3", script],
            stdout=log,
            stderr=subprocess.STDOUT
        )
    
    return result.returncode == 0


def run_primus_extraction(models: List[str], software_stack: str) -> Tuple[List[str], List[str]]:
    """
    Extract metrics from Primus logs for all models.
    
    Args:
        models: List of model names
        software_stack: Software stack (rocm/cuda)
        
    Returns:
        (successful_models, failed_models)
    """
    successful = []
    failed = []
    
    print(f"{Colors.CYAN}{'='*60}{Colors.NC}")
    print(f"{Colors.BLUE}Primus Log Extraction Mode{Colors.NC}")
    print(f"{Colors.CYAN}{'='*60}{Colors.NC}")
    print(f"{Colors.YELLOW}→{Colors.NC} Automatically searching for training logs...")
    print()
    
    for model in models:
        print(f"{Colors.CYAN}{'─'*60}{Colors.NC}")
        print(f"{Colors.BLUE}Model: {Colors.GREEN}{model}{Colors.NC}")
        print(f"{Colors.CYAN}{'─'*60}{Colors.NC}")
        
        log_file = find_log_file(model)
        
        if not log_file:
            print(f"{Colors.YELLOW}⚠️  No log file found for {model}{Colors.NC}")
            print()
            failed.append(model)
            continue
        
        print(f"{Colors.BLUE}→{Colors.NC} Extracting metrics from: {log_file}")
        print()
        
        # Get parallel strategy from environment or default to "unknown"
        parallel_strategy = os.environ.get('TPRIMAT_PARALLEL', 'unknown')
        
        # Get output directory from environment or default
        output_dir = os.environ.get('OUTPUT_DIR', './output')
        
        if extract_metrics(log_file, model, parallel_strategy=parallel_strategy, output_dir=output_dir):
            successful.append(model)
            print(f"{Colors.GREEN}✅ {model} metrics extracted successfully{Colors.NC}")
        else:
            failed.append(model)
            print(f"{Colors.RED}❌ {model} extraction failed{Colors.NC}")
        
        print()
    
    return successful, failed


def run_nemo_benchmarks(models: List[str], runs: int, software_stack: str, output_dir: str = "./output") -> Tuple[List[str], List[str]]:
    """
    Run NeMo training benchmarks for all models.
    
    Args:
        models: List of model names
        runs: Number of runs per model
        software_stack: Software stack (rocm/cuda)
        output_dir: Directory to save logs and results
        
    Returns:
        (successful_models, failed_models)
    """
    successful = []
    failed = []
    
    for model_idx, model in enumerate(models):
        for run in range(1, runs + 1):
            print(f"{Colors.CYAN}{'='*60}{Colors.NC}")
            print(f"{Colors.BLUE}Starting: {Colors.GREEN}{model}{Colors.NC} (Run {run}/{runs})")
            print(f"{Colors.CYAN}{'='*60}{Colors.NC}")
            print()
            
            if run_nemo_training(model, output_dir):
                if run == runs:  # Only add to successful after all runs complete
                    successful.append(model)
                print(f"{Colors.GREEN}✅ {model} run {run} completed successfully{Colors.NC}")
            else:
                failed.append(model)
                print(f"{Colors.RED}❌ {model} run {run} failed{Colors.NC}")
                break  # Stop further runs if one fails
            
            print()
            
            # Cooldown between runs of the same model
            if run < runs:
                print(f"{Colors.YELLOW}⏳ Cooling down for 10 seconds...{Colors.NC}")
                import time
                time.sleep(10)
        
        # Cooldown between different models to allow GPU memory to clear
        if model_idx < len(models) - 1:  # Not the last model
            print()
            print(f"{Colors.YELLOW}⏳ Waiting 20 seconds for GPU memory to clear before next model...{Colors.NC}")
            import time
            time.sleep(20)
            print()
    
    return successful, failed


def print_summary(successful: List[str], failed: List[str], software_stack: str, is_primus: bool, output_dir: str = "./output"):
    """Print summary of benchmark results"""
    print(f"{Colors.CYAN}╔{'='*60}╗{Colors.NC}")
    print(f"{Colors.CYAN}║{Colors.NC}{'Summary':^60}{Colors.CYAN}║{Colors.NC}")
    print(f"{Colors.CYAN}╚{'='*60}╝{Colors.NC}")
    print()
    
    if successful:
        print(f"{Colors.GREEN}✅ Successful ({len(successful)}): {', '.join(successful)}{Colors.NC}")
        print()
        for model in successful:
            print(f"   📄 {output_dir}/benchmark_{software_stack}_{model}.json")
        print()
    
    if failed:
        print(f"{Colors.RED}❌ Failed ({len(failed)}): {', '.join(failed)}{Colors.NC}")
        
        if is_primus:
            print()
            if len(successful) == 0:
                # No logs found at all - provide full guidance
                print(f"{Colors.CYAN}{'='*60}{Colors.NC}")
                print(f"{Colors.YELLOW}No Primus training logs found!{Colors.NC}")
                print(f"{Colors.CYAN}{'='*60}{Colors.NC}")
                print()
                print(f"{Colors.BLUE}To generate benchmark results, choose one of:{Colors.NC}")
                print()
                print(f"{Colors.GREEN}Option 1: Provide log paths via environment variables{Colors.NC}")
                print()
                print("  LLAMA_LOG=/path/to/llama.log \\")
                print("  QWEN_LOG=/path/to/qwen.log \\")
                print("  ./benchmark.py")
                print()
                print(f"{Colors.GREEN}Option 2: Copy logs to current directory{Colors.NC}")
                print()
                print("  cp /path/to/logs/*.log .")
                print("  # Name them: training_llama.log, training_qwen.log")
                print("  ./benchmark.py")
                print()
                print(f"{Colors.GREEN}Option 3: Run Primus training and capture logs{Colors.NC}")
                print()
                print("  cd /workspace/Primus")
                print("  export EXP=examples/megatron/configs/MI300X/llama3.1_8B-pretrain.yaml")
                print("  bash ./examples/run_pretrain.sh --train_iters 10 2>&1 | tee /workspace/tprimat/training_llama.log")
                print()
                print("  Then: cd /workspace/tprimat && ./benchmark.py")
                print()
            else:
                # Some logs found, some missing
                print()
                print(f"{Colors.YELLOW}→{Colors.NC} Missing logs for: {', '.join(failed)}")
                print(f"{Colors.BLUE}Tip:{Colors.NC} Place logs in current directory or output/ directory")
                print(f"      Or use environment variables: {', '.join([f'{m.upper()}_LOG' for m in failed])}")
                print()
    
    if successful or (is_primus and len(successful) > 0):
        print(f"{Colors.BLUE}Next Steps:{Colors.NC}")
        if successful:
            print("  1. Run on the other platform (AMD/NVD)")
            print(f"  2. Compare results: {Colors.GREEN}python3 compare_results.py{Colors.NC}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="TensorPrimat - LLM Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  ./benchmark.py                                    # Run all models (default: minimal_communication)
  ./benchmark.py --model llama                      # Run only llama
  ./benchmark.py --runs 3                           # Run all models 3 times
  
  # Parallelism strategies:
  ./benchmark.py --parallel minimal_communication # TP=1 (default, fastest if fits in memory)
  ./benchmark.py --parallel maximum_performance  # Platform-optimized for speed
  ./benchmark.py --parallel identical_config     # Same config on both platforms
  ./benchmark.py --parallel memory_optimized     # TP=4,PP=2 (save memory)
  ./benchmark.py --parallel balanced             # TP=2 (balanced)
  
  # For AMD/Primus with log files:
  LLAMA_LOG=/path/to/llama.log ./benchmark.py
        """
    )
    
    parser.add_argument(
        '--model',
        choices=['llama', 'qwen', 'all'],
        default='all',
        help='Model to benchmark (default: all)'
    )
    
    parser.add_argument(
        '--runs',
        type=int,
        default=1,
        help='Number of runs per model (default: 1)'
    )
    
    parser.add_argument(
        '--parallel',
        choices=['maximum_performance', 'identical_config', 'memory_optimized', 
                 'minimal_communication', 'balanced'],
        default='minimal_communication',
        help='Parallelism strategy (default: minimal_communication)'
    )
    
    parser.add_argument(
        '--output-dir',
        default='./output',
        help='Output directory for benchmark results (default: ./output)'
    )
    
    args = parser.parse_args()
    
    # Set parallelism strategy environment variable (always has a default)
    os.environ['TPRIMAT_PARALLEL'] = args.parallel
    
    # Set output directory environment variable
    os.environ['OUTPUT_DIR'] = args.output_dir
    
    # Determine models to run
    if args.model == 'all':
        models = ['llama', 'qwen']
    else:
        models = [args.model]
    
    # Detect platform
    platform_name, software_stack, color = detect_platform()
    
    # Check if GPU is available
    gpu_available = platform_name != "Unknown"
    
    # Get parallelism strategy being used (always set from args.parallel)
    parallel_strategy = args.parallel
    
    # Print header
    if len(models) > 1:
        print(f"{Colors.CYAN}╔{'='*60}╗{Colors.NC}")
        print(f"{Colors.CYAN}║{Colors.NC}{f'{Colors.BLUE}TensorPrimat - All Models{Colors.NC}':^68}{Colors.CYAN}║{Colors.NC}")
        print(f"{Colors.CYAN}╚{'='*60}╝{Colors.NC}")
        print()
        print(f"Platform:     {color}{platform_name}{Colors.NC}")
        print(f"Models:       {Colors.GREEN}{', '.join(models)}{Colors.NC}")
        print(f"Runs each:    {Colors.GREEN}{args.runs}{Colors.NC}")
        print(f"Parallelism:  {Colors.GREEN}{parallel_strategy}{Colors.NC}")
        print(f"Output dir:   {Colors.GREEN}{args.output_dir}{Colors.NC}")
        print()
    else:
        print(f"{Colors.BLUE}{'='*60}{Colors.NC}")
        print(f"{Colors.BLUE}TensorPrimat{Colors.NC}")
        print(f"{Colors.BLUE}{'='*60}{Colors.NC}")
        print(f"Model:        {Colors.GREEN}{args.model}{Colors.NC}")
        print(f"Runs:         {Colors.GREEN}{args.runs}{Colors.NC}")
        print(f"Platform:     {color}{platform_name}{Colors.NC}")
        print(f"Parallelism:  {Colors.GREEN}{parallel_strategy}{Colors.NC}")
        print(f"Output dir:   {Colors.GREEN}{args.output_dir}{Colors.NC}")
        print()
    
    # If no GPU, skip training and go straight to log analysis
    if not gpu_available:
        print(f"{Colors.BLUE}→ Log analysis mode (no GPU required){Colors.NC}")
        print()
        successful, failed = run_primus_extraction(models, software_stack)
        has_nemo = False  # No GPU means no NeMo training possible
    else:
        # Check for NeMo (only if GPU is available)
        has_nemo = check_nemo()
        
        if has_nemo:
            print(f"{Colors.GREEN}✓ NeMo detected - Running training mode{Colors.NC}")
            print()
            
            successful, failed = run_nemo_benchmarks(models, args.runs, software_stack, args.output_dir)
        else:
            print(f"{Colors.YELLOW}⚠️  NeMo not detected{Colors.NC}")
            print(f"{Colors.BLUE}→ Using Primus log extraction mode{Colors.NC}")
            print()
            
            successful, failed = run_primus_extraction(models, software_stack)
    
    # Print summary
    print_summary(successful, failed, software_stack, not has_nemo, args.output_dir)
    
    # Exit with appropriate code
    sys.exit(0 if len(failed) == 0 else 1)


if __name__ == "__main__":
    main()
