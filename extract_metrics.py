#!/usr/bin/env python3
"""
Extract benchmark metrics from Primus training logs.

Usage:
    python3 extract_metrics.py \
        --log-file primus_training.log \
        --output benchmark_results/train_primus_manual.json \
        --num-gpus 8 \
        --global-batch-size 128 \
        --sequence-length 2048
"""
import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from config_loader import load_config
    CONFIG_LOADER_AVAILABLE = True
except ImportError:
    CONFIG_LOADER_AVAILABLE = False
    print("[!] config_loader not available - parallelism details will be limited")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[!] PyTorch not available - running in log-only mode")


def round_floats(obj: Any, precision: int = 5) -> Any:
    """
    Recursively round all float values in a nested structure to specified precision.
    
    Args:
        obj: Dictionary, list, or value to process
        precision: Number of decimal places (default: 5)
    
    Returns:
        Object with all floats rounded
    """
    if isinstance(obj, float):
        # Use higher precision for very small numbers (like learning rates)
        # to avoid rounding 5e-6 and 1e-5 to the same value
        if abs(obj) < 0.001 and obj != 0:
            # For learning rates and other small values, preserve more precision
            return round(obj, 10)  # 10 decimal places for small values
        return round(obj, precision)
    elif isinstance(obj, dict):
        return {key: round_floats(value, precision) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [round_floats(item, precision) for item in obj]
    else:
        return obj


def get_gpu_core_count(device_name: str, device_props) -> int:
    """
    Get approximate GPU core count based on device name.
    
    NVIDIA GPUs use CUDA cores, AMD GPUs use Stream Processors.
    """
    device_name_lower = device_name.lower()
    
    # NVIDIA GPUs (CUDA cores)
    nvidia_cores = {
        "h100": 16896,
        "h100 sxm5": 16896,
        "h100 pcie": 14592,
        "a100": 6912,
        "v100": 5120,
        "a40": 10752,
        "a30": 10752,
        "a10": 9216,
        "rtx 4090": 16384,
        "rtx 3090": 10496,
    }
    
    # AMD GPUs (Stream Processors)
    amd_cores = {
        "mi300x": 19456,
        "mi300a": 19456,
        "mi250x": 14080 * 2,  # 2 GCDs
        "mi250": 13312 * 2,
        "mi210": 13312,
        "mi100": 7680,
        "instinct mi300x": 19456,
        "instinct mi250x": 14080 * 2,
        "instinct mi210": 13312,
    }
    
    # Try to match device name
    for gpu_name, cores in nvidia_cores.items():
        if gpu_name in device_name_lower:
            return cores
    
    for gpu_name, cores in amd_cores.items():
        if gpu_name in device_name_lower:
            return cores
    
    # Try to get from device properties
    if hasattr(device_props, 'multi_processor_count'):
        return device_props.multi_processor_count * 128
    
    return 0


def detect_gpu_info():
    """
    Auto-detect GPU information from PyTorch.
    
    Works for both NVIDIA (CUDA) and AMD (ROCm) GPUs.
    torch.cuda.* APIs are compatible with both platforms.
    
    If no GPU is available, returns placeholder info for log analysis.
    """
    gpu_info = {}
    
    if not TORCH_AVAILABLE:
        # PyTorch not available - use placeholder info for log analysis
        print("ℹ️  PyTorch not available - using log data only")
        gpu_info = {
            "device_count": "N/A",
            "device_name": "AMD GPU (from log)",
            "total_memory_gb": 192,  # MI300X default
            "gpu_cores": 19456,  # MI300X default
            "pytorch_version": "N/A",
            "software_stack": "rocm",
            "software_version": "N/A",
        }
        return gpu_info
    
    if torch.cuda.is_available():  # Works for both CUDA and ROCm
        device_props = torch.cuda.get_device_properties(0)
        device_name = torch.cuda.get_device_name(0)
        
        # Detect GPU cores (approximate based on known models)
        gpu_cores = get_gpu_core_count(device_name, device_props)
        
        # Detect software stack and version
        # ROCm sets torch.version.hip, CUDA sets torch.version.cuda
        is_rocm = hasattr(torch.version, 'hip') and torch.version.hip is not None
        software_stack = "rocm" if is_rocm else "cuda"
        software_version = torch.version.hip if is_rocm else torch.version.cuda
        
        gpu_info = {
            "device_count": torch.cuda.device_count(),
            "device_name": device_name,
            "total_memory_gb": device_props.total_memory / 1e9,
            "gpu_cores": gpu_cores,
            "pytorch_version": torch.__version__,
            "software_stack": software_stack,
            "software_version": software_version,
        }
    else:
        # No GPU available - use placeholder info for log analysis
        print("ℹ️  No GPU detected - using log data only")
        gpu_info = {
            "device_count": "N/A",
            "device_name": "Unknown (from logs)",
            "total_memory_gb": "N/A",
            "gpu_cores": 0,
            "pytorch_version": torch.__version__,
            "software_stack": "rocm",  # Default to rocm for log analysis
            "software_version": "N/A",
        }
    
    return gpu_info


def extract_step_times_from_log(log_file):
    """
    Extract step timing and loss from Primus/Megatron logs.
    
    Primus format:
    elapsed time per iteration (ms): 9836.3/21761.7
    lm loss: 1.189761E+01
    learning rate: 5.000000E-06
    """
    step_times = []
    tokens_per_gpu_values = []
    loss_values = []
    learning_rates = []
    
    with open(log_file, 'r') as f:
        for line in f:
            # Primus format: elapsed time per iteration (ms): 9836.3/21761.7
            # First value is current iteration, second is average
            match = re.search(r'elapsed time per iteration \(ms\):\s*([0-9.]+)/([0-9.]+)', line)
            if match:
                try:
                    # Get current iteration time in ms, convert to seconds
                    step_time_ms = float(match.group(1))
                    step_time_s = step_time_ms / 1000.0
                    
                    if 0.001 < step_time_s < 1000:  # Sanity check
                        step_times.append(step_time_s)
                except (ValueError, IndexError):
                    continue
            
            # Also extract tokens per GPU if available
            # tokens per GPU (tokens/s/GPU): 13325.3/8608.1
            tokens_match = re.search(r'tokens per GPU \(tokens/s/GPU\):\s*([0-9.]+)/([0-9.]+)', line)
            if tokens_match:
                try:
                    tokens_per_gpu = float(tokens_match.group(1))
                    if 0 < tokens_per_gpu < 1000000:  # Sanity check
                        tokens_per_gpu_values.append(tokens_per_gpu)
                except (ValueError, IndexError):
                    continue
            
            # Extract loss values
            # lm loss: 1.189761E+01 or lm loss: 11.89761
            loss_match = re.search(r'lm loss:\s*([0-9.Ee+-]+)', line)
            if loss_match:
                try:
                    loss = float(loss_match.group(1))
                    if 0 < loss < 10000:  # Sanity check
                        loss_values.append(loss)
                except (ValueError, IndexError):
                    continue
            
            # learning rate: 5.000000E-06 or learning rate: 0.000005
            lr_match = re.search(r'learning rate:\s*([0-9.Ee+-]+)', line)
            if lr_match:
                try:
                    lr = float(lr_match.group(1))
                    if 0 < lr < 1:  # Sanity check (learning rates are typically 1e-7 to 1e-3)
                        learning_rates.append(lr)
                except (ValueError, IndexError):
                    continue
    
    return step_times, tokens_per_gpu_values, loss_values, learning_rates


def extract_memory_from_log(log_file):
    """
    Extract GPU memory usage from log.
    
    Supports multiple formats:
    - Primus/Megatron: hip mem usage/free/total/usage_ratio: 117.99GB/74.00GB/191.98GB/61.46%
    - Megatron-LM: allocated: 60.2GB, max allocated: 60.3GB, reserved: 62.1GB
    - Generic: memory usage: 60.5 GB
    
    Note: "hip" refers to AMD's HIP (Heterogeneous Interface for Portability),
    which provides CUDA compatibility on ROCm.
    """
    memory_values = []
    
    with open(log_file, 'r') as f:
        for line in f:
            # Format 1: Primus/ROCm - hip mem usage/free/total/usage_ratio: 117.99GB/...
            match = re.search(r'hip mem (?:usage|allocated)[^:]*:\s*([0-9.]+)\s*GB', line, re.IGNORECASE)
            if match:
                try:
                    memory_gb = float(match.group(1))
                    if 0 < memory_gb < 1000:  # Sanity check
                        memory_values.append(memory_gb)
                        continue
                except (ValueError, IndexError):
                    pass
            
            # Format 2: Megatron-LM - allocated: 60.2GB or max allocated: 60.3GB
            match = re.search(r'(?:max )?allocated:\s*([0-9.]+)\s*GB', line, re.IGNORECASE)
            if match:
                try:
                    memory_gb = float(match.group(1))
                    if 0 < memory_gb < 1000:  # Sanity check
                        memory_values.append(memory_gb)
                        continue
                except (ValueError, IndexError):
                    pass
            
            # Format 3: Generic - memory usage: 60.5 GB or memory: 60.5GB
            match = re.search(r'memory\s+(?:usage)?:?\s*([0-9.]+)\s*GB', line, re.IGNORECASE)
            if match:
                try:
                    memory_gb = float(match.group(1))
                    if 0 < memory_gb < 1000:  # Sanity check
                        memory_values.append(memory_gb)
                        continue
                except (ValueError, IndexError):
                    pass
    
    return memory_values


def get_parallelism_config(strategy: str, model: str, platform: str) -> Dict[str, Any]:
    """
    Get parallelism configuration from config.yaml.
    
    Args:
        strategy: Parallelism strategy name
        model: Model name (llama, qwen)
        platform: Platform (amd, nvidia/nvd)
        
    Returns:
        Dictionary with parallelism parameters or minimal info if unavailable
    """
    if not CONFIG_LOADER_AVAILABLE or not strategy or strategy == "unknown":
        return {"strategy": strategy or "unknown"}
    
    try:
        config = load_config()
        # Normalize platform name (nvd -> nvidia)
        platform_normalized = "nvidia" if platform in ["nvd", "nvidia"] else "amd"
        
        # Get parallelism config for this strategy/model/platform
        params = config.get_parallelism(model, platform_normalized, methodology=strategy)
        
        return {
            "strategy": strategy,
            "tensor_parallel_size": params.get("tensor_model_parallel_size", 1),
            "pipeline_parallel_size": params.get("pipeline_model_parallel_size", 1),
            "data_parallel_size": params.get("data_parallel_size", 1),
            "gradient_accumulation_steps": params.get("gradient_accumulation_steps", 1),
        }
    except Exception as e:
        print(f"[!] Could not load parallelism config: {e}")
        return {"strategy": strategy or "unknown"}


def extract_metrics_from_log(log_file, num_gpus, global_batch_size, seq_length, parallel_strategy=None, model_name=None):
    """Extract comprehensive metrics from Primus training log."""
    
    print(f"Analyzing log file: {log_file}")
    
    # Extract step times, tokens per GPU, loss values, and learning rates
    step_times, tokens_per_gpu_values, loss_values, learning_rates = extract_step_times_from_log(log_file)
    
    if not step_times:
        print("[!] No timing data found in log file")
        print("     Log file might use different format.")
        print("     Check the log manually and adjust regex patterns.")
        print("\n     Expected format: 'elapsed time per iteration (ms): X/Y'")
        return None
    
    print(f"  + Found {len(step_times)} step timing entries")
    
    if tokens_per_gpu_values:
        print(f"  + Found {len(tokens_per_gpu_values)} tokens/GPU entries (using Primus native metrics)")
    
    if loss_values:
        print(f"  + Found {len(loss_values)} loss values")
    
    # Extract memory (optional)
    memory_values = extract_memory_from_log(log_file)
    
    if memory_values:
        print(f"  + Found {len(memory_values)} memory usage entries")
    
    # Auto-detect GPU info
    gpu_info = detect_gpu_info()
    if not gpu_info:
        gpu_info = {
            "device_count": num_gpus,
            "device_name": "AMD GPU (from log)",
            "gpu_cores": 0,
            "pytorch_version": torch.__version__,
            "software_stack": "rocm",
            "software_version": "unknown",
        }
    
    # Detect platform from GPU info
    # Use software_stack as the primary indicator
    software_stack = gpu_info.get("software_stack", "rocm")
    
    # Also check device name for additional validation
    device_name_lower = gpu_info.get("device_name", "").lower()
    is_amd = "amd" in device_name_lower or "mi" in device_name_lower or software_stack == "rocm"
    is_nvidia = "nvidia" in device_name_lower or "h100" in device_name_lower or "a100" in device_name_lower or software_stack == "cuda"
    
    # Set platform based on software stack (more reliable than device name)
    if software_stack == "rocm":
        platform = "amd"
    elif software_stack == "cuda":
        platform = "nvd"
    elif is_amd:
        platform = "amd"
    elif is_nvidia:
        platform = "nvd"
    else:
        # Default to amd for Primus logs (Primus is AMD-focused)
        platform = "amd"
    
    # Calculate metrics (skip first step as warmup)
    step_times_no_warmup = step_times[1:] if len(step_times) > 1 else step_times
    avg_step_time = sum(step_times_no_warmup) / len(step_times_no_warmup)
    min_step_time = min(step_times_no_warmup)
    max_step_time = max(step_times_no_warmup)
    
    # Calculate token-based throughput
    # Always calculate from batch size and step time for consistency with NeMo/NVIDIA
    # Note: Primus native metrics report per-microbatch throughput, not per-training-step
    tokens_per_step = global_batch_size * seq_length
    tokens_per_second = tokens_per_step / avg_step_time
    tokens_per_second_per_gpu = tokens_per_second / num_gpus
    print(f"  * Calculated tokens/s/GPU: {tokens_per_second_per_gpu:.1f} (consistent with NeMo)")
    
    steps_per_second = 1.0 / avg_step_time
    
    # Get parallelism configuration from config.yaml
    import os
    # Priority: explicit argument > environment variable
    if not parallel_strategy:
        parallel_strategy = os.environ.get('PARALLEL', None)
    
    # Load full parallelism config from config.yaml
    parallelism_config = get_parallelism_config(
        parallel_strategy or "unknown",
        model_name or "llama",  # Default to llama if not specified
        platform
    )
    
    results = {
        "platform": platform,
        "gpu_info": gpu_info,
        "timestamp": datetime.now().isoformat(),
        "parallelism": parallelism_config,
        "training_config": {
            "max_steps": len(step_times),
            "global_batch_size": global_batch_size,
            "sequence_length": seq_length,
            "num_gpus": num_gpus,
        },
        "performance_metrics": {
            "total_steps": len(step_times),
            "total_time_seconds": sum(step_times),
            "avg_step_time_seconds": avg_step_time,
            "min_step_time_seconds": min_step_time,
            "max_step_time_seconds": max_step_time,
            "steps_per_second": steps_per_second,
            "tokens_per_second": tokens_per_second,
            "tokens_per_second_per_gpu": tokens_per_second_per_gpu,
        },
        "raw_step_times": step_times,
        "raw_loss_values": loss_values if loss_values else [],
        "raw_learning_rates": learning_rates if learning_rates else [],
        "source_log_file": str(Path(log_file).absolute())
    }
    
    return results


def print_summary(results):
    """Print benchmark summary."""
    print(f"\n{'='*60}")
    print(f"BENCHMARK SUMMARY - Platform: {results['platform'].upper()}")
    print(f"{'='*60}")
    print(f"Device: {results['gpu_info'].get('device_name', 'Unknown')}")
    print(f"GPUs: {results['training_config']['num_gpus']}")
    print(f"Total Steps: {results['performance_metrics']['total_steps']}")
    print(f"Avg Step Time: {results['performance_metrics']['avg_step_time_seconds']:.3f}s")
    
    if results['performance_metrics'].get('tokens_per_second'):
        print(f"\nThroughput Metrics:")
        print(f"  Total Throughput: {results['performance_metrics']['tokens_per_second']:,.0f} tokens/sec")
        print(f"  Per-GPU Throughput: {results['performance_metrics']['tokens_per_second_per_gpu']:,.0f} tokens/sec/GPU")
        print(f"  Peak Memory: {max(memory_values):.2f}GB")
    
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Extract benchmark metrics from Primus training logs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # With model name (auto-generates filename)
  python3 extract_metrics.py \\
      --log-file primus_training.log \\
      --model-name llama \\
      --num-gpus 8 \\
      --global-batch-size 128 \\
      --sequence-length 2048
  
  # With explicit output path
  python3 extract_metrics.py \\
      --log-file primus_training.log \\
      --output output/train_primus_llama.json \\
      --num-gpus 8 \\
      --global-batch-size 128 \\
      --sequence-length 2048
        """
    )
    
    parser.add_argument('--log-file', required=True, 
                       help='Primus/Megatron training log file')
    parser.add_argument('--output', 
                       help='Output JSON file path (if not specified, uses output/train_<framework>_<model>.json)')
    parser.add_argument('--model-name', 
                       help='Model name (e.g., llama, qwen) - used in auto-generated filename')
    parser.add_argument('--num-gpus', type=int, required=True,
                       help='Number of GPUs used in training')
    parser.add_argument('--global-batch-size', type=int, required=True,
                       help='Global batch size (total across all GPUs)')
    parser.add_argument('--sequence-length', type=int, default=2048,
                       help='Sequence length (default: 2048)')
    parser.add_argument('--parallel-strategy', 
                       help='Parallelism strategy name (e.g., balanced, minimal_communication)')
    parser.add_argument('--verbose', action='store_true',
                       help='Print verbose output')
    
    args = parser.parse_args()
    
    # Check if log file exists, fallback to backup logs if needed
    log_file = Path(args.log_file)
    if not log_file.exists():
        print(f"[!] Primary log file not found: {log_file}")
        
        # Try to find backup logs in the same directory
        log_dir = log_file.parent
        # Look for backup logs with new naming (primus_training_MODEL.log) and old naming (primus_training_MODEL_*.log)
        log_pattern = f"primus_training_{args.model_name}*.log" if args.model_name else "primus_training*.log"
        
        backup_logs = sorted(log_dir.glob(log_pattern), key=lambda p: p.stat().st_mtime, reverse=True)
        
        if backup_logs:
            log_file = backup_logs[0]
            print(f"[+] Found backup log: {log_file}")
            print(f"  Using most recent backup from: {datetime.fromtimestamp(log_file.stat().st_mtime)}")
            print()
        else:
            print(f"  x No backup logs found matching pattern: {log_pattern}")
            print(f"   Searched in: {log_dir}")
            return 1
    
    # Extract metrics
    results = extract_metrics_from_log(
        str(log_file),
        args.num_gpus,
        args.global_batch_size,
        args.sequence_length,
        args.parallel_strategy,
        args.model_name
    )
    
    if results:
        # Determine output path
        if args.output:
            output_path = Path(args.output).resolve()
        else:
            # Auto-generate filename with model name
            software_stack = results['gpu_info'].get('software_stack', 'unknown')
            model_suffix = f"_{args.model_name}" if args.model_name else ""
            output_path = Path(f"./output/train_{software_stack}{model_suffix}.json").resolve()
        
        # Save results (round all floats to 3 decimal places)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Round all float values to 5 decimal places
        results_rounded = round_floats(results, precision=5)
        
        with open(output_path, 'w') as f:
            json.dump(results_rounded, f, indent=2)
        
        print(f"  + Metrics saved to: {output_path}")
        
        # Print summary
        print_summary(results)
        
        # Print next steps
        print("Next Steps:")
        print("  1. Run on NVIDIA GPU and collect results")
        print("  2. Compare with: python3 compare_results.py")
        print()
        
    else:
        print("  x Failed to extract metrics from log file")
        print("\nTroubleshooting:")
        print("  1. Check log file format")
        print("  2. Verify log contains timing information")
        print("  3. Try opening the log and looking for step times manually")
        print("  4. Adjust regex patterns in extract_metrics.py if needed")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

