#!/usr/bin/env python3
"""
GPU Benchmark Comparison Script

Compares AMD and NVIDIA GPU training performance with:
- Performance metrics (throughput, step time, memory)
- Visual plots and analysis

Usage:
    python3 compare.py [--results-dir ./output]
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np


# ============================================================================
# DATA LOADING
# ============================================================================

def load_all_benchmark_results(results_dir: str) -> Dict[str, Dict]:
    """Load all benchmark results with model and platform information."""
    results_path = Path(results_dir)
    
    benchmarks = {}
    
    # Load training results: train_{framework}_{model}.json
    for json_file in sorted(results_path.glob("train_*.json")):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Extract model name from filename
            # Format: train_{framework}_{model}.json (e.g., "train_nemo_llama")
            filename = json_file.stem
            parts = filename.split('_')
                if len(parts) >= 3:
                    framework = parts[1]  # mega/tran/deep/nemo/prim
                    model_name = parts[2]  # llama or qwen
                    
                    # Map framework to platform
                    if framework in ['mega', 'tran', 'deep', 'nemo']:
                        platform = "nvidia"
                    elif framework in ['prim']:
                        platform = "amd"
                    else:
                        # Fallback: check platform field in JSON
                        platform = data.get('platform', 'unknown')
                    
                    # Store with key like "nvidia-llama"
                    key = f"{platform}-{model_name}"
                    data['model_name'] = model_name
                    data['platform_key'] = platform
                    benchmarks[key] = data
                    
                    print(f"[+] Loaded: {key} from {json_file.name}")
            except Exception as e:
                print(f"[!] Error loading {json_file}: {e}")
    
    return benchmarks


# ============================================================================
# PLOTTING
# ============================================================================

def create_comparison_plot(benchmarks: Dict[str, Dict], output_file: str = "compare.png"):
    """Create visual comparison of all platform-model combinations."""
    
    # Detect which platforms are available
    has_nvidia = any(key.startswith('nvidia-') for key in benchmarks.keys())
    has_amd = any(key.startswith('amd-') for key in benchmarks.keys())
    
    # Create dynamic title based on available data
    if has_nvidia and has_amd:
        title = 'NVIDIA H100 vs AMD Instinct Mi300X'
    elif has_nvidia:
        title = 'NVIDIA H100 Benchmark Results'
    elif has_amd:
        title = 'AMD Instinct Mi300X Benchmark Results'
    else:
        title = 'GPU Benchmark Results'
    
    # Create 2x2 grid for comprehensive comparison with elegant styling
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor='white')
    fig.suptitle(title, fontsize=18, fontweight='bold', y=0.995, color='#2C3E50')
    
    # Flatten axes for easier indexing
    axes = axes.flatten()
    
    # Setup - elegant pastel colors and markers
    # Order: nvidia-llama, nvidia-qwen, amd-llama, amd-qwen
    style_map = {
        'nvidia-llama': {'color': '#7FB3D5', 'marker': 'o', 'label': 'NVIDIA Llama', 'linestyle': '-'},
        'nvidia-qwen':  {'color': '#85C1E9', 'marker': 's', 'label': 'NVIDIA Qwen', 'linestyle': '--'},
        'amd-llama':    {'color': '#F1948A', 'marker': 'o', 'label': 'AMD Llama', 'linestyle': '-'},
        'amd-qwen':     {'color': '#F5B7B1', 'marker': 's', 'label': 'AMD Qwen', 'linestyle': '--'},
    }
    
    # Extract training config for calculations (use first available)
    first_data = next(iter(benchmarks.values()))
    config = first_data.get('training_config', {})
    global_batch_size = config.get('global_batch_size', 128)
    seq_length = config.get('sequence_length', 2048)
    num_gpus = config.get('num_gpus', 8)
    
    # 1. Per-GPU Throughput (Bar Chart)
    ax1 = axes[0]
    labels = []
    values = []
    colors_list = []
    
    for key in ['nvidia-llama', 'nvidia-qwen', 'amd-llama', 'amd-qwen']:
        if key in benchmarks:
            perf = benchmarks[key]['performance_metrics']
            tps_gpu = perf.get('tokens_per_second_per_gpu')
            if tps_gpu:
                labels.append(style_map[key]['label'])
                values.append(tps_gpu)
                colors_list.append(style_map[key]['color'])
    
    if values:
        bars = ax1.bar(labels, values, color=colors_list, alpha=0.75, edgecolor='#333333', linewidth=1.2)
        ax1.set_ylabel('Tokens/s/GPU', fontweight='bold', fontsize=11)
        ax1.set_title('Average Per-GPU Throughput', fontweight='bold', fontsize=12)
        ax1.grid(axis='y', alpha=0.2, linestyle='--', linewidth=0.5)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:,.0f}',
                    ha='center', va='bottom', fontweight='bold', fontsize=9)
    else:
        ax1.text(0.5, 0.5, 'Throughput data not available', 
                ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('Average Per-GPU Throughput', fontweight='bold', fontsize=12)
    
    # 2. Training Loss over Time
    ax2 = axes[1]
    has_data = False
    
    for key in ['nvidia-llama', 'nvidia-qwen', 'amd-llama', 'amd-qwen']:
        if key in benchmarks and 'loss_values' in benchmarks[key]:
            loss_values = benchmarks[key]['loss_values']
            if loss_values:
                steps = range(len(loss_values))
                style = style_map[key]
                ax2.plot(steps, loss_values, 
                        marker=style['marker'], 
                        linestyle=style['linestyle'],
                        color=style['color'], 
                        label=style['label'], 
                        linewidth=1.5, 
                        markersize=2, 
                        alpha=0.85)
                has_data = True
    
    if has_data:
        ax2.set_xlabel('Step', fontweight='bold', fontsize=10)
        ax2.set_ylabel('Loss', fontweight='bold', fontsize=11)
        ax2.set_title('Training Loss over Time', fontweight='bold', fontsize=12)
        ax2.legend(fontsize=8, loc='best')
        ax2.grid(alpha=0.2, linestyle='--', linewidth=0.5)
    else:
        ax2.text(0.5, 0.5, 'Loss data not available', 
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Training Loss over Time', fontweight='bold', fontsize=12)
    
    # 3. Learning Rate Schedule over Time (from logs)
    ax3 = axes[2]
    has_data = False
    
    for key in ['nvidia-llama', 'nvidia-qwen', 'amd-llama', 'amd-qwen']:
        if key in benchmarks and 'learning_rates' in benchmarks[key]:
            lr_values = benchmarks[key]['learning_rates']
            if lr_values:
                steps = range(len(lr_values))
                style = style_map[key]
                ax3.plot(steps, lr_values, 
                        marker=style['marker'], 
                        linestyle=style['linestyle'],
                        color=style['color'], 
                        label=style['label'], 
                        linewidth=1.5, 
                        markersize=2, 
                        alpha=0.85)
                has_data = True
    
    if has_data:
        ax3.set_xlabel('Step', fontweight='bold', fontsize=10)
        ax3.set_ylabel('Learning Rate', fontweight='bold', fontsize=11)
        ax3.set_title('Learning Rate over Time', fontweight='bold', fontsize=12)
        ax3.legend(fontsize=8, loc='best')
        ax3.grid(alpha=0.2, linestyle='--', linewidth=0.5)
        ax3.ticklabel_format(axis='y', style='scientific', scilimits=(0,0))
    else:
        ax3.text(0.5, 0.5, 'Learning rate data not available', 
                ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Learning Rate over Time', fontweight='bold', fontsize=12)
    
    # 4. Step Duration over Time
    ax4 = axes[3]
    has_data = False
    
    for key in ['nvidia-llama', 'nvidia-qwen', 'amd-llama', 'amd-qwen']:
        if key in benchmarks and 'step_times' in benchmarks[key]:
            step_times = benchmarks[key]['step_times']
            if step_times:
                steps = range(len(step_times))
                style = style_map[key]
                ax4.plot(steps, step_times, 
                        marker=style['marker'], 
                        linestyle=style['linestyle'],
                        color=style['color'], 
                        label=style['label'], 
                        linewidth=1.5, 
                        markersize=2, 
                        alpha=0.85)
                
                # Add average line annotation
                avg_time = sum(step_times) / len(step_times)
                ax4.axhline(y=avg_time, color=style['color'], linestyle=':', alpha=0.3, linewidth=0.8)
                has_data = True
    
    if has_data:
        ax4.set_xlabel('Step', fontweight='bold', fontsize=10)
        ax4.set_ylabel('Time (secs)', fontweight='bold', fontsize=11)
        ax4.set_title('Step Duration over Time', fontweight='bold', fontsize=12)
        ax4.legend(fontsize=8, loc='best')
        ax4.grid(alpha=0.2, linestyle='--', linewidth=0.5)
    else:
        ax4.text(0.5, 0.5, 'Step time data not available', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Step Duration over Time', fontweight='bold', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  + Comparison plot saved to: {output_file}")
    
    return fig


# ============================================================================
# ENHANCED METRICS COMPARISON
# ============================================================================

def print_comparison(nvidia_data: Dict, amd_data: Dict):
    """Print comprehensive comparison of benchmark metrics."""
    
    print("\n" + "="*80)
    print("BENCHMARK COMPARISON")
    print("="*80)
    
    # Extract metrics
    nvidia_perf = nvidia_data['performance_metrics']
    amd_perf = amd_data['performance_metrics']
    nvidia_gpu = nvidia_data['gpu_info']
    amd_gpu = amd_data['gpu_info']
    
    # 1. Performance Metrics
    print("\n  * Performance Metrics")
    print("-" * 80)
    
    nvidia_tps_gpu = nvidia_perf['tokens_per_second_per_gpu']
    amd_tps_gpu = amd_perf['tokens_per_second_per_gpu']
    nvidia_step = nvidia_perf['avg_step_time_seconds']
    amd_step = amd_perf['avg_step_time_seconds']
    
    print(f"  Tokens per Second (Per GPU):")
    print(f"    NVIDIA: {nvidia_tps_gpu:10,.1f}")
    print(f"    AMD:    {amd_tps_gpu:10,.1f}")
    print(f"    → AMD is {amd_tps_gpu/nvidia_tps_gpu:.2f}x faster per GPU")
    
    print(f"\n  Average Step Time:")
    print(f"    NVIDIA: {nvidia_step:7.2f} secs")
    print(f"    AMD:    {amd_step:7.2f} secs")
    print(f"    → AMD is {nvidia_step/amd_step:.2f}x faster per step")
    
    # 2. Hardware Info
    print("\n🖥️  Hardware Configuration")
    print("-" * 80)
    print(f"  NVIDIA: {nvidia_gpu.get('device_name', 'Unknown')} ({nvidia_gpu.get('device_count', 0)} GPUs)")
    print(f"  AMD:    {amd_gpu.get('device_name', 'Unknown')} ({amd_gpu.get('device_count', 0)} GPUs)")
    
    # Summary table
    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY")
    print("="*80)
    
    metrics = [
        ("Per-GPU Throughput (tokens/s)", nvidia_tps_gpu, amd_tps_gpu, "higher"),
        ("Step Time (secs)", nvidia_step, amd_step, "lower"),
    ]
    
    print(f"\n{'Metric':<40} {'NVIDIA':>12} {'AMD':>12} {'Winner':>8}")
    print("-" * 80)
    
    for metric_name, nvidia_val, amd_val, better in metrics:
        if better == "higher":
            winner = "AMD" if amd_val > nvidia_val else "NVIDIA"
            ratio = max(nvidia_val, amd_val) / min(nvidia_val, amd_val) if min(nvidia_val, amd_val) > 0 else 1.0
        else:
            winner = "AMD" if amd_val < nvidia_val else "NVIDIA"
            ratio = max(nvidia_val, amd_val) / min(nvidia_val, amd_val) if min(nvidia_val, amd_val) > 0 else 1.0
        
        print(f"{metric_name:<40} {nvidia_val:12,.2f} {amd_val:12,.2f} {winner:>8} ({ratio:.1f}x)")
    
    print("\n" + "="*80 + "\n")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='GPU benchmark comparison - all models and platforms'
    )
    # Default to OUTPUT_DIR env var if set, otherwise './output'
    default_dir = os.environ.get('OUTPUT_DIR', './output')
    parser.add_argument('--results-dir', default=default_dir,
                       help='Directory containing benchmark JSON files (default: OUTPUT_DIR env var or ./output)')
    parser.add_argument('--output', default='compare.png',
                       help='Output filename for the comparison plot (default: compare.png)')
    
    args = parser.parse_args()
    
    print("Loading benchmark results...")
    benchmarks = load_all_benchmark_results(args.results_dir)
    
    if not benchmarks:
        print("  x No benchmark results found!")
        print(f"Expected files in {args.results_dir}/:")
        print("  NVIDIA frameworks:")
        print("    - train_mega_llama.json, train_mega_qwen.json (Megatron)")
        print("    - train_tran_llama.json, train_tran_qwen.json (Transformers)")
        print("    - train_deep_llama.json, train_deep_qwen.json (DeepSpeed)")
        print("    - train_nemo_llama.json, train_nemo_qwen.json (NeMo)")
        print("  AMD frameworks:")
        print("    - train_prim_llama.json, train_prim_qwen.json (Primus)")
        return 1
    
    # Detect which platforms are available
    has_nvidia = any(key.startswith('nvidia-') for key in benchmarks.keys())
    has_amd = any(key.startswith('amd-') for key in benchmarks.keys())
    
    print(f"\n  * Found {len(benchmarks)} benchmark(s):")
    for key in sorted(benchmarks.keys()):
        data = benchmarks[key]
        print(f"  {key}: {data['gpu_info']['device_name']} ({data['timestamp']})")
    
    # Show platform availability
    print(f"\n  * Platform availability:")
    print(f"  NVIDIA: {'+ Available' if has_nvidia else 'x Not available'}")
    print(f"  AMD:    {'+ Available' if has_amd else 'x Not available'}")
    
    if not has_nvidia and not has_amd:
        print("[!] Warning: No recognized platform data found")
    elif not has_nvidia:
        print("ℹ️  Note: Generating AMD-only comparison (no NVIDIA data)")
    elif not has_amd:
        print("ℹ️  Note: Generating NVIDIA-only comparison (no AMD data)")
    
    # Generate comparison plot with all models
    print(f"\nGenerating comparison plot: {args.output}")
    try:
        create_comparison_plot(benchmarks, args.output)
    except Exception as e:
        print(f"[!] Could not generate plot: {e}")
        import traceback
        traceback.print_exc()
    
    # Print summary table
    print("\n" + "="*100)
    print("PERFORMANCE SUMMARY")
    print("="*100)
    print(f"\n{'Configuration':<25} {'Tokens/sec/GPU':>15} {'Avg Step Time':>15} {'Avg Loss':>12}")
    print("-" * 100)
    
    for key in ['nvidia-llama', 'nvidia-qwen', 'amd-llama', 'amd-qwen']:
        if key in benchmarks:
            data = benchmarks[key]
            perf = data['performance_metrics']
            tps_gpu = perf.get('tokens_per_second_per_gpu', 0)
            step_time = perf.get('avg_step_time_seconds', 0)
            loss_values = data.get('loss_values', [])
            avg_loss = sum(loss_values) / len(loss_values) if loss_values else 0
            
            print(f"{key:<25} {tps_gpu:>15,.1f} {step_time:>15.2f} {avg_loss:>12.2f}")
    
    print("\n" + "="*100 + "\n")
    
    return 0


if __name__ == "__main__":
    exit(main())
