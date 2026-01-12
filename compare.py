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
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np


# ============================================================================
# DATA LOADING
# ============================================================================

def load_benchmark_results(results_dir: str) -> Tuple[List[Dict], List[Dict]]:
    """Load all benchmark results and separate by platform."""
    results_path = Path(results_dir)
    
    nvidia_results = []
    amd_results = []
    
    for json_file in sorted(results_path.glob("*.json")):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Support both old and new platform naming
            platform = data.get('platform', '').lower()
            software_stack = data.get('gpu_info', {}).get('software_stack', '').lower()
            
            # Prioritize software_stack over platform for accurate detection
            # NVIDIA: cuda
            if software_stack == 'cuda':
                nvidia_results.append(data)
            # AMD: rocm
            elif software_stack == 'rocm':
                amd_results.append(data)
            # Fallback to platform field (for older files)
            elif platform in ['cuda', 'nvd', 'nvidia']:
                nvidia_results.append(data)
            elif platform in ['rocm', 'amd']:
                amd_results.append(data)
        except Exception as e:
            print(f"⚠️  Error loading {json_file}: {e}")
    
    return nvidia_results, amd_results


# ============================================================================
# PLOTTING
# ============================================================================

def create_comparison_plot(nvidia_data: Dict, amd_data: Dict, output_file: str = "comparison.png"):
    """Create visual comparison of AMD vs NVIDIA performance."""
    
    # Check if we have tokens/sec/GPU data (primary metric)
    has_tokens_per_gpu = (nvidia_data['performance_metrics'].get('tokens_per_second_per_gpu') is not None and 
                          amd_data['performance_metrics'].get('tokens_per_second_per_gpu') is not None)
    
    # Create 4x2 grid for comprehensive comparison with elegant styling
    fig, axes = plt.subplots(4, 2, figsize=(16, 18), facecolor='white')
    fig.suptitle('GPU Benchmark Comparison', fontsize=18, fontweight='bold', y=0.995, color='#2C3E50')
    
    # Flatten axes for easier indexing
    axes = axes.flatten()
    
    # Setup - elegant pastel colors
    platforms = ['NVIDIA\n' + nvidia_data['gpu_info']['device_name'], 
                 'AMD\n' + amd_data['gpu_info']['device_name']]
    colors = ['#7FB3D5', '#F1948A']  # Soft blue (NVIDIA), Soft coral (AMD)
    
    # Extract training config for calculations
    nvidia_config = nvidia_data.get('training_config', {})
    amd_config = amd_data.get('training_config', {})
    global_batch_size = nvidia_config.get('global_batch_size', 128)
    seq_length = nvidia_config.get('sequence_length', 2048)
    num_gpus = nvidia_config.get('num_gpus', 8)
    
    # 1. Per-GPU Throughput (Bar Chart)
    ax1 = axes[0]
    if has_tokens_per_gpu:
        tokens_per_gpu = [
            nvidia_data['performance_metrics']['tokens_per_second_per_gpu'],
            amd_data['performance_metrics']['tokens_per_second_per_gpu']
        ]
        bars = ax1.bar(platforms, tokens_per_gpu, color=colors, alpha=0.75, edgecolor='#333333', linewidth=1.2)
        ax1.set_ylabel('Tokens/sec/GPU', fontweight='bold', fontsize=11)
        ax1.set_title('Average Per-GPU Throughput', fontweight='bold', fontsize=12)
        ax1.grid(axis='y', alpha=0.2, linestyle='--', linewidth=0.5)
        
        # Add value labels on bars
        for bar, value in zip(bars, tokens_per_gpu):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:,.0f}',
                    ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        # Add performance comparison annotation
        ratio = max(tokens_per_gpu) / min(tokens_per_gpu) if min(tokens_per_gpu) > 0 else 1.0
        winner = "AMD" if tokens_per_gpu[1] > tokens_per_gpu[0] else "NVIDIA"
        ax1.text(0.5, 0.95, f'{winner}: {ratio:.1f}x faster',
                transform=ax1.transAxes, ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor='#F9E79F', alpha=0.5, edgecolor='#E5C100', linewidth=1),
                fontsize=9, fontweight='bold')
    else:
        ax1.text(0.5, 0.5, 'Tokens/sec/GPU data not available', 
                ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('Average Per-GPU Throughput', fontweight='bold', fontsize=12)
    
    # 2. Per-GPU Throughput over Time
    ax2 = axes[1]
    if 'raw_step_times' in nvidia_data and 'raw_step_times' in amd_data:
        nvidia_times = nvidia_data['raw_step_times']
        amd_times = amd_data['raw_step_times']
        steps = range(len(nvidia_times))
        
        # Calculate per-step throughput: (global_batch_size * seq_length) / (step_time * num_gpus)
        nvidia_throughput = [(global_batch_size * seq_length) / (t * num_gpus) for t in nvidia_times]
        amd_throughput = [(global_batch_size * seq_length) / (t * num_gpus) for t in amd_times]
        
        ax2.plot(steps, nvidia_throughput, 'o-', color=colors[0], label='NVIDIA', linewidth=1.5, markersize=2, alpha=0.85)
        ax2.plot(steps, amd_throughput, 's-', color=colors[1], label='AMD', linewidth=1.5, markersize=2, alpha=0.85)
        ax2.set_xlabel('Step', fontweight='bold', fontsize=10)
        ax2.set_ylabel('Tokens/sec/GPU', fontweight='bold', fontsize=11)
        ax2.set_title('Per-GPU Throughput over Time', fontweight='bold', fontsize=12)
        ax2.legend(fontsize=9)
        ax2.grid(alpha=0.2, linestyle='--', linewidth=0.5)
    else:
        ax2.text(0.5, 0.5, 'Step time data not available', 
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Per-GPU Throughput over Time', fontweight='bold', fontsize=12)
    
    # 3. Training Loss over Time
    ax3 = axes[2]
    if 'raw_loss_values' in nvidia_data and 'raw_loss_values' in amd_data:
        nvidia_loss = nvidia_data['raw_loss_values']
        amd_loss = amd_data['raw_loss_values']
        steps = range(len(nvidia_loss))
        
        ax3.plot(steps, nvidia_loss, 'o-', color=colors[0], label='NVIDIA', linewidth=1.5, markersize=2, alpha=0.85)
        ax3.plot(steps, amd_loss, 's-', color=colors[1], label='AMD', linewidth=1.5, markersize=2, alpha=0.85)
        ax3.set_xlabel('Step', fontweight='bold', fontsize=10)
        ax3.set_ylabel('Loss', fontweight='bold', fontsize=11)
        ax3.set_title('Training Loss over Time', fontweight='bold', fontsize=12)
        ax3.legend(fontsize=9)
        ax3.grid(alpha=0.2, linestyle='--', linewidth=0.5)
    else:
        ax3.text(0.5, 0.5, 'Loss data not available', 
                ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Training Loss over Time', fontweight='bold', fontsize=12)
    
    # 4. Cumulative Tokens per GPU over Time
    ax4 = axes[3]
    if 'raw_step_times' in nvidia_data and 'raw_step_times' in amd_data:
        steps = range(len(nvidia_data['raw_step_times']))
        tokens_per_step = (global_batch_size * seq_length) / num_gpus
        
        # Cumulative tokens processed per GPU
        nvidia_tokens = [tokens_per_step * (i + 1) for i in steps]
        amd_tokens = [tokens_per_step * (i + 1) for i in steps]
        
        ax4.plot(steps, nvidia_tokens, 'o-', color=colors[0], label='NVIDIA', linewidth=1.5, markersize=2, alpha=0.85)
        ax4.plot(steps, amd_tokens, 's-', color=colors[1], label='AMD', linewidth=1.5, markersize=2, alpha=0.85)
        ax4.set_xlabel('Step', fontweight='bold', fontsize=10)
        ax4.set_ylabel('Cumulative Tokens', fontweight='bold', fontsize=11)
        ax4.set_title('Cumulative Tokens per GPU over Time', fontweight='bold', fontsize=12)
        ax4.legend(fontsize=9)
        ax4.grid(alpha=0.2, linestyle='--', linewidth=0.5)
        
        # Format y-axis as millions
        ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M'))
    else:
        ax4.text(0.5, 0.5, 'Step data not available', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Cumulative Tokens per GPU over Time', fontweight='bold', fontsize=12)
    
    # 5. Learning Rate Schedule over Time
    ax5 = axes[4]
    if 'raw_step_times' in nvidia_data and 'raw_step_times' in amd_data:
        num_steps = len(nvidia_data['raw_step_times'])
        steps = range(num_steps)
        
        # Calculate learning rate schedule (warmup + cosine decay)
        # Assuming: warmup_steps=10, lr=3e-4, min_lr=3e-5
        warmup_steps = 10
        max_lr = 3.0e-4
        min_lr = 3.0e-5
        
        lr_schedule = []
        for step in steps:
            if step < warmup_steps:
                # Linear warmup
                lr = min_lr + (max_lr - min_lr) * step / warmup_steps
            else:
                # Cosine decay
                progress = (step - warmup_steps) / (num_steps - warmup_steps)
                lr = min_lr + (max_lr - min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
            lr_schedule.append(lr)
        
        # Same schedule for both (same config)
        ax5.plot(steps, lr_schedule, '-', color='#9B59B6', linewidth=2, alpha=0.85)
        ax5.set_xlabel('Step', fontweight='bold', fontsize=10)
        ax5.set_ylabel('Learning Rate', fontweight='bold', fontsize=11)
        ax5.set_title('Learning Rate Schedule', fontweight='bold', fontsize=12)
        ax5.grid(alpha=0.2, linestyle='--', linewidth=0.5)
        ax5.axvline(x=warmup_steps, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax5.text(warmup_steps, max_lr * 0.5, f'Warmup\n({warmup_steps} steps)', 
                ha='right', va='center', fontsize=8, color='gray')
    else:
        ax5.text(0.5, 0.5, 'Step data not available', 
                ha='center', va='center', transform=ax5.transAxes)
        ax5.set_title('Learning Rate Schedule', fontweight='bold', fontsize=12)
    
    # 6. Consumed Samples over Time
    ax6 = axes[5]
    if 'raw_step_times' in nvidia_data and 'raw_step_times' in amd_data:
        steps = range(len(nvidia_data['raw_step_times']))
        
        # Cumulative samples consumed (global_batch_size per step)
        nvidia_samples = [global_batch_size * (i + 1) for i in steps]
        amd_samples = [global_batch_size * (i + 1) for i in steps]
        
        ax6.plot(steps, nvidia_samples, 'o-', color=colors[0], label='NVIDIA', linewidth=1.5, markersize=2, alpha=0.85)
        ax6.plot(steps, amd_samples, 's-', color=colors[1], label='AMD', linewidth=1.5, markersize=2, alpha=0.85)
        ax6.set_xlabel('Step', fontweight='bold', fontsize=10)
        ax6.set_ylabel('Consumed Samples', fontweight='bold', fontsize=11)
        ax6.set_title('Consumed Samples over Time', fontweight='bold', fontsize=12)
        ax6.legend(fontsize=9)
        ax6.grid(alpha=0.2, linestyle='--', linewidth=0.5)
    else:
        ax6.text(0.5, 0.5, 'Step data not available', 
                ha='center', va='center', transform=ax6.transAxes)
        ax6.set_title('Consumed Samples over Time', fontweight='bold', fontsize=12)
    
    # 7. GPU Memory Usage over Time
    ax7 = axes[6]
    nvidia_mem_data = nvidia_data.get('raw_memory_values', [])
    amd_mem_data = amd_data.get('raw_memory_values', [])
    
    if nvidia_mem_data and amd_mem_data:
        # Use actual memory time series
        steps_nvidia = range(len(nvidia_mem_data))
        steps_amd = range(len(amd_mem_data))
        
        ax7.plot(steps_nvidia, nvidia_mem_data, 'o-', color=colors[0], label='NVIDIA', linewidth=1.5, markersize=2, alpha=0.85)
        ax7.plot(steps_amd, amd_mem_data, 's-', color=colors[1], label='AMD', linewidth=1.5, markersize=2, alpha=0.85)
        ax7.set_xlabel('Step', fontweight='bold', fontsize=10)
        ax7.set_ylabel('Memory (GB)', fontweight='bold', fontsize=11)
        ax7.set_title('GPU Memory Usage over Time', fontweight='bold', fontsize=12)
        ax7.legend(fontsize=9)
        ax7.grid(alpha=0.2, linestyle='--', linewidth=0.5)
    else:
        ax7.text(0.5, 0.5, 'Memory time series not available', 
                ha='center', va='center', transform=ax7.transAxes)
        ax7.set_title('GPU Memory Usage over Time', fontweight='bold', fontsize=12)
    
    # 8. Step Duration over Time
    ax8 = axes[7]
    if 'raw_step_times' in nvidia_data and 'raw_step_times' in amd_data:
        nvidia_times = nvidia_data['raw_step_times']
        amd_times = amd_data['raw_step_times']
        steps = range(len(nvidia_times))
        
        ax8.plot(steps, nvidia_times, 'o-', color=colors[0], label='NVIDIA', linewidth=1.5, markersize=2, alpha=0.85)
        ax8.plot(steps, amd_times, 's-', color=colors[1], label='AMD', linewidth=1.5, markersize=2, alpha=0.85)
        ax8.set_xlabel('Step', fontweight='bold', fontsize=10)
        ax8.set_ylabel('Time (seconds)', fontweight='bold', fontsize=11)
        ax8.set_title('Step Duration over Time', fontweight='bold', fontsize=12)
        ax8.legend(fontsize=9)
        ax8.grid(alpha=0.2, linestyle='--', linewidth=0.5)
        
        # Add average line annotations
        nvidia_avg = sum(nvidia_times) / len(nvidia_times)
        amd_avg = sum(amd_times) / len(amd_times)
        ax8.axhline(y=nvidia_avg, color=colors[0], linestyle='--', alpha=0.25, linewidth=0.8)
        ax8.axhline(y=amd_avg, color=colors[1], linestyle='--', alpha=0.25, linewidth=0.8)
    else:
        ax8.text(0.5, 0.5, 'Step time data not available', 
                ha='center', va='center', transform=ax8.transAxes)
        ax8.set_title('Step Duration over Time', fontweight='bold', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Comparison plot saved to: {output_file}")
    
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
    nvidia_mem_series = nvidia_data.get('raw_memory_values', [])
    amd_mem_series = amd_data.get('raw_memory_values', [])
    nvidia_gpu = nvidia_data['gpu_info']
    amd_gpu = amd_data['gpu_info']
    
    # 1. Performance Metrics
    print("\n📊 Performance Metrics")
    print("-" * 80)
    
    nvidia_tps = nvidia_perf['tokens_per_second']
    amd_tps = amd_perf['tokens_per_second']
    nvidia_tps_gpu = nvidia_perf['tokens_per_second_per_gpu']
    amd_tps_gpu = amd_perf['tokens_per_second_per_gpu']
    nvidia_step = nvidia_perf['avg_step_time_seconds']
    amd_step = amd_perf['avg_step_time_seconds']
    
    print(f"  Tokens per Second (Total):")
    print(f"    NVIDIA: {nvidia_tps:10,.1f}")
    print(f"    AMD:    {amd_tps:10,.1f}")
    print(f"    → AMD is {amd_tps/nvidia_tps:.2f}x faster")
    
    print(f"\n  Tokens per Second (Per GPU):")
    print(f"    NVIDIA: {nvidia_tps_gpu:10,.1f}")
    print(f"    AMD:    {amd_tps_gpu:10,.1f}")
    print(f"    → AMD is {amd_tps_gpu/nvidia_tps_gpu:.2f}x faster per GPU")
    
    print(f"\n  Average Step Time:")
    print(f"    NVIDIA: {nvidia_step:7.2f} seconds")
    print(f"    AMD:    {amd_step:7.2f} seconds")
    print(f"    → AMD is {nvidia_step/amd_step:.2f}x faster per step")
    
    # 2. Memory Usage
    print("\n💾 Memory Usage")
    print("-" * 80)
    
    if nvidia_mem_series and amd_mem_series:
        nvidia_mem_used = sum(nvidia_mem_series) / len(nvidia_mem_series)
        amd_mem_used = sum(amd_mem_series) / len(amd_mem_series)
        nvidia_mem_total = nvidia_gpu.get('total_memory_gb', 80)
        amd_mem_total = amd_gpu.get('total_memory_gb', 192)
        
        nvidia_util = (nvidia_mem_used / nvidia_mem_total) * 100 if nvidia_mem_total > 0 else 0
        amd_util = (amd_mem_used / amd_mem_total) * 100 if amd_mem_total > 0 else 0
        
        print(f"  NVIDIA: {nvidia_mem_used:5.1f} GB / {nvidia_mem_total:5.1f} GB ({nvidia_util:4.1f}% utilized) [{len(nvidia_mem_series)} samples]")
        print(f"  AMD:    {amd_mem_used:5.1f} GB / {amd_mem_total:5.1f} GB ({amd_util:4.1f}% utilized) [{len(amd_mem_series)} samples]")
    else:
        print(f"  Memory time series data not available")
    
    # 3. Hardware Info
    print("\n🖥️  Hardware Configuration")
    print("-" * 80)
    print(f"  NVIDIA: {nvidia_gpu.get('device_name', 'Unknown')} ({nvidia_gpu.get('device_count', 0)} GPUs)")
    print(f"  AMD:    {amd_gpu.get('device_name', 'Unknown')} ({amd_gpu.get('device_count', 0)} GPUs)")
    
    # Summary table
    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY")
    print("="*80)
    
    metrics = [
        ("Total Throughput (tokens/s)", nvidia_tps, amd_tps, "higher"),
        ("Per-GPU Throughput (tokens/s)", nvidia_tps_gpu, amd_tps_gpu, "higher"),
        ("Step Time (seconds)", nvidia_step, amd_step, "lower"),
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
        description='GPU benchmark comparison with enhanced metrics'
    )
    parser.add_argument('--results-dir', default='./output',
                       help='Directory containing benchmark JSON files')
    
    args = parser.parse_args()
    
    print("Loading benchmark results...")
    nvidia_results, amd_results = load_benchmark_results(args.results_dir)
    
    if not nvidia_results:
        print("❌ No NVIDIA benchmark results found!")
        return 1
    
    if not amd_results:
        print("❌ No AMD benchmark results found!")
        return 1
    
    # Use most recent results
    nvidia_data = sorted(nvidia_results, key=lambda x: x['timestamp'])[-1]
    amd_data = sorted(amd_results, key=lambda x: x['timestamp'])[-1]
    
    print(f"\n📊 Comparing:")
    print(f"  NVIDIA: {nvidia_data['gpu_info']['device_name']} ({nvidia_data['timestamp']})")
    print(f"  AMD:    {amd_data['gpu_info']['device_name']} ({amd_data['timestamp']})")
    
    # Generate standard comparison plot
    print("\nGenerating comparison plot...")
    try:
        create_comparison_plot(nvidia_data, amd_data, "comparison.png")
    except Exception as e:
        print(f"⚠️  Could not generate plot: {e}")
    
    # Print comparison
    print_comparison(nvidia_data, amd_data)
    
    return 0


if __name__ == "__main__":
    exit(main())
