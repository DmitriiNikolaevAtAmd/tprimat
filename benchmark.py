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
    MISTRAL_LOG=/path/to/mistral.log
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
    
    Returns:
        (platform_name, software_stack, color)
        e.g., ("AMD (ROCm)", "rocm", Colors.RED)
    """
    try:
        import torch
        
        if not torch.cuda.is_available():
            print(f"{Colors.RED}❌ No GPU detected!{Colors.NC}")
            print("Please ensure CUDA or ROCm is properly installed.")
            sys.exit(1)
        
        # Check if ROCm or CUDA
        is_rocm = hasattr(torch.version, 'hip') and torch.version.hip is not None
        
        if is_rocm:
            return "AMD (ROCm)", "rocm", Colors.RED
        else:
            return "NVD (CUDA)", "cuda", Colors.GREEN
            
    except ImportError:
        print(f"{Colors.RED}❌ PyTorch not found!{Colors.NC}")
        print("Please install PyTorch.")
        sys.exit(1)


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
    1. Environment variable (LLAMA_LOG, MISTRAL_LOG, QWEN_LOG)
    2. Standard filenames (training_llama.log, etc.)
    3. Pattern matching (*llama*.log)
    4. Content-based search in all .log/.txt files
    
    Args:
        model: Model name (llama, mistral, qwen)
        
    Returns:
        Path to log file or None
    """
    # Check environment variable
    env_var = f"{model.upper()}_LOG"
    if env_var in os.environ:
        log_path = os.environ[env_var]
        if os.path.isfile(log_path):
            return log_path
    
    # Try standard filenames
    patterns = [
        f"training_{model}.log",
        f"{model}_training.log",
        f"primus_{model}.log",
        f"*{model}*.log",
    ]
    
    for pattern in patterns:
        matches = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
        if matches:
            return matches[0]
    
    # Search all .log and .txt files by content
    print(f"{Colors.YELLOW}→{Colors.NC} Searching all log files for {model} training...")
    
    for log_file in glob.glob("*.log") + glob.glob("*.txt"):
        if not os.path.isfile(log_file):
            continue
            
        try:
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(10000)  # Read first 10KB
                
                # Check for model-specific strings
                if model in content.lower():
                    print(f"{Colors.GREEN}→{Colors.NC} Found potential match: {log_file}")
                    return log_file
                    
                # More specific checks
                if model == "llama" and "llama" in content.lower() and "8b" in content.lower():
                    print(f"{Colors.GREEN}→{Colors.NC} Found potential match: {log_file}")
                    return log_file
                elif model == "mistral" and "mistral" in content.lower() and "7b" in content.lower():
                    print(f"{Colors.GREEN}→{Colors.NC} Found potential match: {log_file}")
                    return log_file
                elif model == "qwen" and "qwen" in content.lower():
                    print(f"{Colors.GREEN}→{Colors.NC} Found potential match: {log_file}")
                    return log_file
        except Exception:
            continue
    
    return None


def extract_metrics(log_file: str, model: str, num_gpus: int = 8, 
                   global_batch_size: int = 128, sequence_length: int = 2048) -> bool:
    """
    Extract metrics from Primus log file.
    
    Args:
        log_file: Path to log file
        model: Model name
        num_gpus: Number of GPUs
        global_batch_size: Global batch size
        sequence_length: Sequence length
        
    Returns:
        True if successful, False otherwise
    """
    cmd = [
        "python3", "extract_primus_metrics.py",
        "--log-file", log_file,
        "--model-name", model,
        "--num-gpus", str(num_gpus),
        "--global-batch-size", str(global_batch_size),
        "--sequence-length", str(sequence_length),
    ]
    
    result = subprocess.run(cmd)
    return result.returncode == 0


def run_nemo_training(model: str) -> bool:
    """
    Run NeMo training script for a model.
    
    Args:
        model: Model name (llama, mistral, qwen)
        
    Returns:
        True if successful, False otherwise
    """
    scripts = {
        "llama": "pretrain_llama.py",
        "mistral": "pretrain_mistral.py",
        "qwen": "pretrain_qwen.py",
    }
    
    script = scripts.get(model)
    if not script:
        print(f"{Colors.RED}❌ Unknown model: {model}{Colors.NC}")
        return False
    
    if not os.path.isfile(script):
        print(f"{Colors.RED}❌ Script not found: {script}{Colors.NC}")
        return False
    
    result = subprocess.run(["python3", script])
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
    
    for model in models:
        print(f"{Colors.CYAN}{'='*60}{Colors.NC}")
        print(f"{Colors.BLUE}Searching for {Colors.GREEN}{model}{Colors.NC} logs...")
        print(f"{Colors.CYAN}{'='*60}{Colors.NC}")
        
        log_file = find_log_file(model)
        
        if not log_file:
            print(f"{Colors.YELLOW}⚠️  No log file found for {model}{Colors.NC}")
            print()
            failed.append(model)
            continue
        
        print(f"{Colors.GREEN}✓{Colors.NC} Found log: {log_file}")
        print(f"{Colors.BLUE}→{Colors.NC} Extracting metrics...")
        print()
        
        if extract_metrics(log_file, model):
            successful.append(model)
            print(f"{Colors.GREEN}✅ {model} extracted successfully{Colors.NC}")
        else:
            failed.append(model)
            print(f"{Colors.RED}❌ {model} extraction failed{Colors.NC}")
        
        print()
    
    return successful, failed


def run_nemo_benchmarks(models: List[str], runs: int, software_stack: str) -> Tuple[List[str], List[str]]:
    """
    Run NeMo training benchmarks for all models.
    
    Args:
        models: List of model names
        runs: Number of runs per model
        software_stack: Software stack (rocm/cuda)
        
    Returns:
        (successful_models, failed_models)
    """
    successful = []
    failed = []
    
    for model in models:
        for run in range(1, runs + 1):
            print(f"{Colors.CYAN}{'='*60}{Colors.NC}")
            print(f"{Colors.BLUE}Starting: {Colors.GREEN}{model}{Colors.NC} (Run {run}/{runs})")
            print(f"{Colors.CYAN}{'='*60}{Colors.NC}")
            print()
            
            if run_nemo_training(model):
                if run == runs:  # Only add to successful after all runs complete
                    successful.append(model)
                print(f"{Colors.GREEN}✅ {model} run {run} completed successfully{Colors.NC}")
            else:
                failed.append(model)
                print(f"{Colors.RED}❌ {model} run {run} failed{Colors.NC}")
                break  # Stop further runs if one fails
            
            print()
            
            # Cooldown between runs
            if run < runs:
                print(f"{Colors.YELLOW}Cooling down for 10 seconds...{Colors.NC}")
                import time
                time.sleep(10)
    
    return successful, failed


def print_summary(successful: List[str], failed: List[str], software_stack: str, is_primus: bool):
    """Print summary of benchmark results"""
    print(f"{Colors.CYAN}╔{'='*60}╗{Colors.NC}")
    print(f"{Colors.CYAN}║{Colors.NC}{'Summary':^60}{Colors.CYAN}║{Colors.NC}")
    print(f"{Colors.CYAN}╚{'='*60}╝{Colors.NC}")
    print()
    
    if successful:
        print(f"{Colors.GREEN}✅ Successful ({len(successful)}): {', '.join(successful)}{Colors.NC}")
        for model in successful:
            print(f"   📄 output/benchmark_{software_stack}_{model}.json")
        print()
    
    if failed:
        print(f"{Colors.RED}❌ Failed ({len(failed)}): {', '.join(failed)}{Colors.NC}")
        print()
        
        if is_primus and len(successful) == 0:
            # No logs found - provide guidance
            print(f"{Colors.CYAN}{'='*60}{Colors.NC}")
            print(f"{Colors.YELLOW}No Primus training logs found!{Colors.NC}")
            print(f"{Colors.CYAN}{'='*60}{Colors.NC}")
            print()
            print(f"{Colors.BLUE}Choose one of the following options:{Colors.NC}")
            print()
            print(f"{Colors.GREEN}Option 1: Provide log paths via environment variables{Colors.NC}")
            print()
            print("  LLAMA_LOG=/path/to/llama.log \\")
            print("  MISTRAL_LOG=/path/to/mistral.log \\")
            print("  QWEN_LOG=/path/to/qwen.log \\")
            print("  ./benchmark.py")
            print()
            print(f"{Colors.GREEN}Option 2: Copy logs to current directory{Colors.NC}")
            print()
            print("  cp /path/to/logs/*.log .")
            print("  # Name them: training_llama.log, training_mistral.log, training_qwen.log")
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
    
    print(f"{Colors.BLUE}Next Steps:{Colors.NC}")
    print("  1. Run on the other platform (AMD/NVD)")
    print(f"  2. Compare: {Colors.GREEN}python3 compare_results.py{Colors.NC}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="TensorPrimat - LLM Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  ./benchmark.py                    # Run all models
  ./benchmark.py --model llama      # Run only llama
  ./benchmark.py --runs 3           # Run all models 3 times
  
  # For AMD/Primus with log files:
  LLAMA_LOG=/path/to/llama.log ./benchmark.py
        """
    )
    
    parser.add_argument(
        '--model',
        choices=['llama', 'mistral', 'qwen', 'all'],
        default='all',
        help='Model to benchmark (default: all)'
    )
    
    parser.add_argument(
        '--runs',
        type=int,
        default=1,
        help='Number of runs per model (default: 1)'
    )
    
    args = parser.parse_args()
    
    # Determine models to run
    if args.model == 'all':
        models = ['llama', 'mistral', 'qwen']
    else:
        models = [args.model]
    
    # Detect platform
    platform_name, software_stack, color = detect_platform()
    
    # Print header
    if len(models) > 1:
        print(f"{Colors.CYAN}╔{'='*60}╗{Colors.NC}")
        print(f"{Colors.CYAN}║{Colors.NC}{f'{Colors.BLUE}TensorPrimat - All Models{Colors.NC}':^68}{Colors.CYAN}║{Colors.NC}")
        print(f"{Colors.CYAN}╚{'='*60}╝{Colors.NC}")
        print()
        print(f"Platform:  {color}{platform_name}{Colors.NC}")
        print(f"Models:    {Colors.GREEN}{', '.join(models)}{Colors.NC}")
        print(f"Runs each: {Colors.GREEN}{args.runs}{Colors.NC}")
        print()
    else:
        print(f"{Colors.BLUE}{'='*60}{Colors.NC}")
        print(f"{Colors.BLUE}TensorPrimat{Colors.NC}")
        print(f"{Colors.BLUE}{'='*60}{Colors.NC}")
        print(f"Model: {Colors.GREEN}{args.model}{Colors.NC}")
        print(f"Runs:  {Colors.GREEN}{args.runs}{Colors.NC}")
        print()
        print(f"Platform: {color}{platform_name}{Colors.NC}")
        print()
    
    # Check for NeMo
    has_nemo = check_nemo()
    
    if has_nemo:
        print(f"{Colors.GREEN}✓ NeMo detected - Running training mode{Colors.NC}")
        print()
        
        successful, failed = run_nemo_benchmarks(models, args.runs, software_stack)
    else:
        print(f"{Colors.YELLOW}⚠️  NeMo not detected{Colors.NC}")
        print(f"{Colors.BLUE}→ Using Primus log extraction mode{Colors.NC}")
        print()
        
        successful, failed = run_primus_extraction(models, software_stack)
    
    # Print summary
    print_summary(successful, failed, software_stack, not has_nemo)
    
    # Exit with appropriate code
    sys.exit(0 if len(failed) == 0 else 1)


if __name__ == "__main__":
    main()
