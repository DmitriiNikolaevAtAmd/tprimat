#!/usr/bin/env python3
"""
Peak Throughput Analyzer

Analyzes all benchmark results to identify peak throughput across different
configurations and platforms.

Usage:
    python3 analyze_peak_throughput.py
    python3 analyze_peak_throughput.py --show-all  # Show all configs
"""

import argparse
import glob
import json
from pathlib import Path
from typing import Dict, List, Tuple
import sys


# Hardware theoretical peak specs (TFLOPS for FP16/BF16)
THEORETICAL_PEAKS = {
    "NVIDIA H100 80GB HBM3": {
        "tflops_fp16": 1979,  # Tensor Core peak
        "memory_bandwidth_gbps": 3352,  # HBM3
        "total_memory_gb": 80,
    },
    "AMD Instinct MI300X": {
        "tflops_fp16": 1307,  # Matrix Core peak
        "memory_bandwidth_gbps": 5200,  # HBM3
        "total_memory_gb": 192,
    }
}

# Model FLOPs per token (6 × num_parameters for forward + backward)
MODEL_FLOPS_PER_TOKEN = {
    'llama': 6 * 8.0e9,   # 48 billion FLOPs per token (Llama 3.1 8B)
    'qwen': 6 * 7.6e9,    # 45.6 billion FLOPs per token (Qwen 2.5 7B)
}


def find_all_benchmark_files() -> List[Path]:
    """Find all benchmark JSON files in the project."""
    patterns = [
        "output/benchmark_*.json",
        "all_outputs/*/benchmark_*.json",
        "nvd-output/*/benchmark_*.json",
    ]
    
    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern))
    
    return [Path(f) for f in sorted(files)]


def parse_benchmark_file(file_path: Path) -> Dict:
    """Parse a benchmark file and extract relevant information."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Extract model name from filename
        filename = file_path.stem  # e.g., "benchmark_cuda_llama"
        parts = filename.split('_')
        model_name = parts[2] if len(parts) >= 3 else 'unknown'
        
        # Get configuration directory (if in subdirectory)
        config_name = file_path.parent.name if file_path.parent.name != 'output' else 'default'
        
        # Calculate TFLOPS
        perf = data.get('performance_metrics', {})
        tps = perf.get('tokens_per_second', 0)
        tps_per_gpu = perf.get('tokens_per_second_per_gpu', 0)
        
        flops_per_token = MODEL_FLOPS_PER_TOKEN.get(model_name, 6 * 7.8e9)
        
        # Total TFLOPS
        total_tflops = (tps * flops_per_token) / 1e12 if tps else 0
        
        # TFLOPS per GPU
        tflops_per_gpu = (tps_per_gpu * flops_per_token) / 1e12 if tps_per_gpu else 0
        
        # Get hardware info
        gpu_info = data.get('gpu_info', {})
        device_name = gpu_info.get('device_name', 'Unknown')
        device_count = gpu_info.get('num_gpus', gpu_info.get('device_count', 8))
        
        # Calculate hardware utilization
        theoretical_specs = THEORETICAL_PEAKS.get(device_name, {})
        theoretical_peak = theoretical_specs.get('tflops_fp16', 0)
        
        if theoretical_peak > 0 and device_count > 0:
            total_theoretical = theoretical_peak * device_count
            hw_utilization = (total_tflops / total_theoretical) * 100
        else:
            hw_utilization = 0
            total_theoretical = 0
        
        # Get training config
        training_config = data.get('training_config', {})
        
        # Get parallelism info if available
        parallel_config = data.get('parallel_config', {})
        tp = parallel_config.get('tensor_parallel', 'N/A')
        pp = parallel_config.get('pipeline_parallel', 'N/A')
        
        return {
            'file_path': str(file_path),
            'config_name': config_name,
            'model_name': model_name,
            'platform': data.get('platform', 'unknown'),
            'device_name': device_name,
            'device_count': device_count,
            'tokens_per_second': tps,
            'tokens_per_second_per_gpu': tps_per_gpu,
            'total_tflops': total_tflops,
            'tflops_per_gpu': tflops_per_gpu,
            'theoretical_peak_tflops': theoretical_peak,
            'total_theoretical_tflops': total_theoretical,
            'hw_utilization_pct': hw_utilization,
            'avg_step_time': perf.get('avg_step_time_seconds', 0),
            'min_step_time': perf.get('min_step_time_seconds', 0),
            'global_batch_size': training_config.get('global_batch_size', 0),
            'sequence_length': training_config.get('sequence_length', 0),
            'tensor_parallel': tp,
            'pipeline_parallel': pp,
            'timestamp': data.get('timestamp', ''),
        }
    except Exception as e:
        print(f"⚠️  Error parsing {file_path}: {e}")
        return None


def analyze_peak_throughput(show_all: bool = False):
    """Analyze all benchmarks and identify peak throughput."""
    
    print("="*100)
    print("PEAK THROUGHPUT ANALYSIS")
    print("="*100)
    print()
    
    # Find all benchmark files
    files = find_all_benchmark_files()
    
    if not files:
        print("❌ No benchmark files found!")
        return
    
    print(f"Found {len(files)} benchmark file(s)")
    print()
    
    # Parse all files
    results = []
    for file_path in files:
        result = parse_benchmark_file(file_path)
        if result:
            results.append(result)
    
    if not results:
        print("❌ No valid benchmark data found!")
        return
    
    # Group by platform and model
    platforms = {}
    for r in results:
        key = f"{r['device_name']} - {r['model_name']}"
        if key not in platforms:
            platforms[key] = []
        platforms[key].append(r)
    
    # Find peak for each platform-model combination
    print("🏆 PEAK THROUGHPUT BY CONFIGURATION")
    print("="*100)
    print()
    
    peak_results = []
    for key, configs in sorted(platforms.items()):
        # Find peak by tokens per second
        peak = max(configs, key=lambda x: x['tokens_per_second'])
        peak_results.append(peak)
        
        print(f"📊 {key}")
        print(f"   Configuration: {peak['config_name']}")
        print(f"   Peak Tokens/s: {peak['tokens_per_second']:,.0f}")
        print(f"   Per-GPU:       {peak['tokens_per_second_per_gpu']:,.0f} tokens/s")
        print(f"   Total TFLOPS:  {peak['total_tflops']:,.1f}")
        print(f"   Per-GPU:       {peak['tflops_per_gpu']:.1f} TFLOPS")
        print(f"   Step Time:     {peak['min_step_time']:.2f}s (min), {peak['avg_step_time']:.2f}s (avg)")
        
        if peak['total_theoretical_tflops'] > 0:
            print(f"   HW Utilization: {peak['hw_utilization_pct']:.1f}% of theoretical peak")
            print(f"   Theoretical:   {peak['total_theoretical_tflops']:,.0f} TFLOPS ({peak['device_count']} × {peak['theoretical_peak_tflops']} TFLOPS)")
        
        if peak['tensor_parallel'] != 'N/A':
            print(f"   Parallelism:   TP={peak['tensor_parallel']}, PP={peak['pipeline_parallel']}")
        
        print()
        
        # Show other configs if requested
        if show_all and len(configs) > 1:
            print(f"   Other configurations:")
            for cfg in sorted(configs, key=lambda x: x['tokens_per_second'], reverse=True):
                if cfg != peak:
                    print(f"     • {cfg['config_name']}: {cfg['tokens_per_second']:,.0f} tokens/s "
                          f"({cfg['total_tflops']:,.1f} TFLOPS, {cfg['hw_utilization_pct']:.1f}% util)")
            print()
    
    # Overall peak throughput
    print("="*100)
    print("🥇 OVERALL PEAK THROUGHPUT")
    print("="*100)
    print()
    
    # Best overall
    overall_peak = max(peak_results, key=lambda x: x['tokens_per_second'])
    
    print(f"Best Configuration: {overall_peak['device_name']} - {overall_peak['model_name']}")
    print(f"Configuration:      {overall_peak['config_name']}")
    print(f"Peak Throughput:    {overall_peak['tokens_per_second']:,.0f} tokens/s")
    print(f"                    {overall_peak['total_tflops']:,.1f} TFLOPS total")
    print(f"                    {overall_peak['tflops_per_gpu']:.1f} TFLOPS per GPU")
    print(f"Hardware:           {overall_peak['device_count']} × {overall_peak['device_name']}")
    print(f"HW Utilization:     {overall_peak['hw_utilization_pct']:.1f}%")
    print()
    
    # Best per platform
    print("-"*100)
    print("Best by Platform:")
    print()
    
    by_platform = {}
    for r in peak_results:
        platform = r['device_name']
        if platform not in by_platform or r['tokens_per_second'] > by_platform[platform]['tokens_per_second']:
            by_platform[platform] = r
    
    for platform, r in sorted(by_platform.items()):
        print(f"  {platform}:")
        print(f"    Model:          {r['model_name']}")
        print(f"    Configuration:  {r['config_name']}")
        print(f"    Peak:           {r['tokens_per_second']:,.0f} tokens/s ({r['total_tflops']:,.1f} TFLOPS)")
        print(f"    Per-GPU:        {r['tokens_per_second_per_gpu']:,.0f} tokens/s ({r['tflops_per_gpu']:.1f} TFLOPS)")
        print(f"    HW Utilization: {r['hw_utilization_pct']:.1f}%")
        print()
    
    # Scaling comparison
    print("="*100)
    print("📈 THROUGHPUT SCALING ANALYSIS")
    print("="*100)
    print()
    
    print(f"Batch Processing Capacity (at peak):")
    print(f"  Tokens/second:  {overall_peak['tokens_per_second']:,.0f}")
    print(f"  Samples/second: {overall_peak['tokens_per_second'] / overall_peak['sequence_length']:.1f} "
          f"(seq_len={overall_peak['sequence_length']})")
    print()
    
    tokens_per_day = overall_peak['tokens_per_second'] * 86400
    print(f"Projected Training Capacity (24h at peak):")
    print(f"  Tokens:         {tokens_per_day:,.0f} ({tokens_per_day/1e9:.1f}B tokens)")
    print(f"  Samples:        {tokens_per_day / overall_peak['sequence_length']:,.0f}")
    print()
    
    # Multi-node projection
    if overall_peak['device_count'] == 8:
        print(f"Multi-Node Scaling Projection (assuming linear scaling):")
        for nodes in [2, 4, 8]:
            total_gpus = nodes * 8
            projected_tps = overall_peak['tokens_per_second'] * nodes
            projected_tflops = overall_peak['total_tflops'] * nodes
            print(f"  {nodes} nodes ({total_gpus} GPUs): {projected_tps:,.0f} tokens/s ({projected_tflops:,.0f} TFLOPS)")
        print()
    
    print("="*100)
    
    # Summary table
    print()
    print("📋 COMPLETE RESULTS TABLE")
    print("="*100)
    print(f"{'Platform':<30} {'Model':<10} {'Config':<15} {'Tokens/s':>12} {'TFLOPS':>10} {'Util%':>8}")
    print("-"*100)
    
    for r in sorted(results, key=lambda x: x['tokens_per_second'], reverse=True):
        platform_short = r['device_name'].replace('NVIDIA ', '').replace('AMD ', '').replace(' 80GB HBM3', '')[:29]
        print(f"{platform_short:<30} {r['model_name']:<10} {r['config_name']:<15} "
              f"{r['tokens_per_second']:>12,.0f} {r['total_tflops']:>10,.1f} {r['hw_utilization_pct']:>7.1f}%")
    
    print("="*100)
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze peak throughput across all benchmark configurations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        '--show-all',
        action='store_true',
        help='Show all configurations, not just peak for each platform-model'
    )
    
    args = parser.parse_args()
    
    analyze_peak_throughput(show_all=args.show_all)


if __name__ == "__main__":
    main()
