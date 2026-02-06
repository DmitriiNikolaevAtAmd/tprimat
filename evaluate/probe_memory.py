#!/usr/bin/env python3
"""
Probe GPU memory usage on AMD/NVIDIA GPUs.

This script queries PyTorch's memory statistics to get actual GPU memory usage.
Can be run standalone or integrated into training pipelines.

Usage:
    python3 probe_gpu_memory.py [--output memory.json]
"""
import argparse
import json
import os
import sys


def get_gpu_memory_stats():
    """Get GPU memory statistics using PyTorch."""
    try:
        import torch
    except ImportError:
        print("Error: PyTorch not available", file=sys.stderr)
        return None
    
    if not torch.cuda.is_available():
        print("Error: CUDA/ROCm not available", file=sys.stderr)
        return None
    
    num_gpus = torch.cuda.device_count()
    
    memory_stats = {
        "num_gpus": num_gpus,
        "per_gpu": [],
        "total": {
            "allocated_gb": 0,
            "reserved_gb": 0,
            "max_allocated_gb": 0,
            "max_reserved_gb": 0,
        }
    }
    
    for i in range(num_gpus):
        try:
            # Get memory stats for this GPU
            allocated = torch.cuda.memory_allocated(i) / 1e9
            reserved = torch.cuda.memory_reserved(i) / 1e9
            max_allocated = torch.cuda.max_memory_allocated(i) / 1e9
            max_reserved = torch.cuda.max_memory_reserved(i) / 1e9
            
            gpu_stats = {
                "device_id": i,
                "device_name": torch.cuda.get_device_name(i),
                "allocated_gb": round(allocated, 3),
                "reserved_gb": round(reserved, 3),
                "max_allocated_gb": round(max_allocated, 3),
                "max_reserved_gb": round(max_reserved, 3),
            }
            
            memory_stats["per_gpu"].append(gpu_stats)
            memory_stats["total"]["allocated_gb"] += allocated
            memory_stats["total"]["reserved_gb"] += reserved
            memory_stats["total"]["max_allocated_gb"] = max(
                memory_stats["total"]["max_allocated_gb"], max_allocated
            )
            memory_stats["total"]["max_reserved_gb"] = max(
                memory_stats["total"]["max_reserved_gb"], max_reserved
            )
            
        except Exception as e:
            print(f"Warning: Could not get memory stats for GPU {i}: {e}", file=sys.stderr)
    
    # Round totals
    memory_stats["total"]["allocated_gb"] = round(memory_stats["total"]["allocated_gb"], 3)
    memory_stats["total"]["reserved_gb"] = round(memory_stats["total"]["reserved_gb"], 3)
    memory_stats["total"]["max_allocated_gb"] = round(memory_stats["total"]["max_allocated_gb"], 3)
    memory_stats["total"]["max_reserved_gb"] = round(memory_stats["total"]["max_reserved_gb"], 3)
    
    return memory_stats


def print_memory_summary(stats):
    """Print a human-readable memory summary."""
    if not stats:
        return
    
    print(f"\n{'='*60}")
    print("GPU MEMORY USAGE")
    print(f"{'='*60}")
    
    for gpu in stats["per_gpu"]:
        print(f"\nGPU {gpu['device_id']}: {gpu['device_name']}")
        print(f"  Current allocated: {gpu['allocated_gb']:.2f} GB")
        print(f"  Current reserved:  {gpu['reserved_gb']:.2f} GB")
        print(f"  Peak allocated:    {gpu['max_allocated_gb']:.2f} GB")
        print(f"  Peak reserved:     {gpu['max_reserved_gb']:.2f} GB")
    
    print(f"\n{'-'*60}")
    print(f"PEAK MEMORY (max across GPUs): {stats['total']['max_allocated_gb']:.2f} GB allocated")
    print(f"{'='*60}\n")


def parse_memory_log(log_file, return_values=False):
    """Parse rocm-smi or nvidia-smi memory log to find memory usage.
    
    Args:
        log_file: Path to the memory log file
        return_values: If True, include full memory_values array in result
    
    Returns:
        Dict with peak_memory_gb, avg_memory_gb, samples, and optionally memory_values
    """
    import re
    
    if not os.path.exists(log_file):
        return None
    
    memory_values = []
    
    with open(log_file, 'r') as f:
        for line in f:
            # rocm-smi formats:
            # "VRAM Total Used Memory (B): 73014444032"
            # "Used: 69632 MB"
            # nvidia-smi format: "0, 65432" (index, memory_mb)
            
            # Match bytes format
            match = re.search(r'Used.*\(B\)[:\s]+(\d+)', line)
            if match:
                bytes_val = int(match.group(1))
                memory_values.append(round(bytes_val / 1e9, 2))  # Convert to GB
                continue
            
            # Match MB format
            match = re.search(r'Used[:\s]+(\d+)\s*MB', line, re.IGNORECASE)
            if match:
                mb_val = int(match.group(1))
                memory_values.append(round(mb_val / 1024, 2))  # Convert to GB
                continue
            
            # Match nvidia-smi CSV format (index, memory_mb)
            match = re.search(r'^\d+,\s*(\d+)', line)
            if match:
                mb_val = int(match.group(1))
                memory_values.append(round(mb_val / 1024, 2))  # Convert to GB
                continue
    
    if memory_values:
        result = {
            "peak_memory_gb": round(max(memory_values), 2),
            "avg_memory_gb": round(sum(memory_values) / len(memory_values), 2),
            "min_memory_gb": round(min(memory_values), 2),
            "samples": len(memory_values),
        }
        if return_values:
            result["memory_values"] = memory_values
        return result
    
    return None


def main():
    parser = argparse.ArgumentParser(description='Probe GPU memory usage')
    parser.add_argument('--output', '-o', help='Output JSON file path')
    parser.add_argument('--quiet', '-q', action='store_true', help='Suppress output')
    parser.add_argument('--json', action='store_true', help='Output as JSON to stdout')
    parser.add_argument('--parse-log', help='Parse rocm-smi/nvidia-smi log file instead of probing')
    parser.add_argument('--output-values', help='Output memory values JSON file (for line charts)')
    
    args = parser.parse_args()
    
    # If parsing a log file, use that instead of probing
    if args.parse_log:
        # Get full values if output-values is requested
        log_stats = parse_memory_log(args.parse_log, return_values=bool(args.output_values))
        if log_stats:
            if not args.quiet:
                print(f"Parsed memory log: {args.parse_log}")
                print(f"  Peak memory: {log_stats['peak_memory_gb']:.2f} GB")
                print(f"  Avg memory:  {log_stats['avg_memory_gb']:.2f} GB")
                print(f"  Samples:     {log_stats['samples']}")
            
            # Write memory values to JSON file if requested
            if args.output_values and 'memory_values' in log_stats:
                with open(args.output_values, 'w') as f:
                    json.dump({
                        "memory_values": log_stats['memory_values'],
                        "peak_memory_gb": log_stats['peak_memory_gb'],
                        "avg_memory_gb": log_stats['avg_memory_gb'],
                        "min_memory_gb": log_stats['min_memory_gb'],
                    }, f, indent=2)
                if not args.quiet:
                    print(f"  Memory values saved to: {args.output_values}")
            
            print(f"PEAK_MEMORY_GB={log_stats['peak_memory_gb']}")
            return 0
        else:
            print(f"Error: Could not parse memory from log: {args.parse_log}", file=sys.stderr)
            print("PEAK_MEMORY_GB=0")
            return 1
    
    stats = get_gpu_memory_stats()
    
    if not stats:
        return 1
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(stats, f, indent=2)
        if not args.quiet:
            print(f"Memory stats saved to: {args.output}")
    
    if args.json:
        print(json.dumps(stats, indent=2))
    elif not args.quiet:
        print_memory_summary(stats)
    
    # Output key metric for shell scripts to capture
    print(f"PEAK_MEMORY_GB={stats['total']['max_allocated_gb']}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
