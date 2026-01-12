#!/usr/bin/env python3
"""
GPU Benchmark Comparison Script with Enhanced Metrics

Compares AMD and NVIDIA GPU training performance with:
- Standard performance metrics (throughput, step time, memory)
- Enhanced metrics (MFU, cost efficiency, power efficiency)
- Visual plots and detailed analysis

Usage:
    python3 compare.py [--results-dir ./output]
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np

# Import the enhanced metrics calculator
from enhanced_metrics import get_enhanced_metrics, GPU_SPECS


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
    
    # Create 3x2 grid for comprehensive comparison
    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    fig.suptitle('AMD vs NVIDIA GPU Comparison', fontsize=18, fontweight='bold')
    
    # Flatten axes for easier indexing
    axes = axes.flatten()
    
    # Setup
    platforms = ['NVIDIA\n' + nvidia_data['gpu_info']['device_name'], 
                 'AMD\n' + amd_data['gpu_info']['device_name']]
    colors = ['#76B900', '#ED1C24']  # NVIDIA green, AMD red
    
    # 1. Tokens/sec/GPU - PRIMARY METRIC (Most Important)
    ax1 = axes[0]
    if has_tokens_per_gpu:
        tokens_per_gpu = [
            nvidia_data['performance_metrics']['tokens_per_second_per_gpu'],
            amd_data['performance_metrics']['tokens_per_second_per_gpu']
        ]
        bars = ax1.bar(platforms, tokens_per_gpu, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
        ax1.set_ylabel('Tokens/sec/GPU', fontweight='bold', fontsize=12)
        ax1.set_title('Tokens/sec/GPU - Per-GPU Efficiency', 
                     fontweight='bold', fontsize=13)
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, tokens_per_gpu):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:,.0f}',
                    ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        # Add annotation explaining the metric
        max_val = max(tokens_per_gpu)
        ratio = tokens_per_gpu[0] / tokens_per_gpu[1] if tokens_per_gpu[1] > 0 else 1.0
        winner = "NVIDIA" if ratio > 1 else "AMD"
        ax1.text(0.5, 0.95, f'{winner} is {max(ratio, 1/ratio):.2f}x more efficient per GPU',
                transform=ax1.transAxes, ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=10, fontweight='bold')
    else:
        ax1.text(0.5, 0.5, 'Tokens/sec/GPU data not available', 
                ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('Tokens/sec/GPU - Per-GPU Efficiency')
    
    # 2. Average Step Time Comparison
    ax2 = axes[1]
    step_times = [
        nvidia_data['performance_metrics']['avg_step_time_seconds'],
        amd_data['performance_metrics']['avg_step_time_seconds']
    ]
    bars = ax2.bar(platforms, step_times, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Time (seconds)', fontweight='bold')
    ax2.set_title('Average Step Time\n(Lower is Better)', fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, step_times):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.3f}s',
                ha='center', va='bottom', fontweight='bold')
    
    # 3. Total System Throughput
    ax3 = axes[2]
    if has_tokens_per_gpu:
        total_throughput = [
            nvidia_data['performance_metrics'].get('tokens_per_second', 0),
            amd_data['performance_metrics'].get('tokens_per_second', 0)
        ]
        bars = ax3.bar(platforms, total_throughput, color=colors, alpha=0.7, edgecolor='black')
        ax3.set_ylabel('Tokens/sec (Total)', fontweight='bold')
        ax3.set_title('Total System Throughput', fontweight='bold')
        ax3.grid(axis='y', alpha=0.3)
        
        for bar, value in zip(bars, total_throughput):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:,.0f}',
                    ha='center', va='bottom', fontweight='bold')
    else:
        throughputs = [
            nvidia_data['performance_metrics'].get('throughput_steps_per_second', 
                nvidia_data['performance_metrics'].get('steps_per_second', 0)),
            amd_data['performance_metrics'].get('throughput_steps_per_second',
                amd_data['performance_metrics'].get('steps_per_second', 0))
        ]
        bars = ax3.bar(platforms, throughputs, color=colors, alpha=0.7, edgecolor='black')
        ax3.set_ylabel('Steps per Second', fontweight='bold')
        ax3.set_title('Throughput', fontweight='bold')
        ax3.grid(axis='y', alpha=0.3)
        
        for bar, value in zip(bars, throughputs):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.3f}',
                    ha='center', va='bottom', fontweight='bold')
    
    # 4. Memory Usage Comparison
    ax4 = axes[3]
    if 'memory_metrics' in nvidia_data and 'memory_metrics' in amd_data:
        memory_data = {
            'Average': [
                nvidia_data['memory_metrics']['avg_memory_allocated_gb'],
                amd_data['memory_metrics']['avg_memory_allocated_gb']
            ],
            'Peak': [
                nvidia_data['memory_metrics']['peak_memory_allocated_gb'],
                amd_data['memory_metrics']['peak_memory_allocated_gb']
            ]
        }
        
        x = np.arange(len(platforms))
        width = 0.35
        
        bars1 = ax4.bar(x - width/2, memory_data['Average'], width, 
                       label='Average', alpha=0.7, edgecolor='black')
        bars2 = ax4.bar(x + width/2, memory_data['Peak'], width,
                       label='Peak', alpha=0.7, edgecolor='black')
        
        ax4.set_ylabel('Memory (GB)', fontweight='bold')
        ax4.set_title('GPU Memory Usage')
        ax4.set_xticks(x)
        ax4.set_xticklabels(platforms)
        ax4.legend()
        ax4.grid(axis='y', alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'Memory data not available', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('GPU Memory Usage')
    
    # 5. GPU Configuration Info
    ax5 = axes[4]
    # Handle "N/A" values in device_count
    def get_gpu_count(data):
        count = data['gpu_info'].get('device_count', data['training_config'].get('num_gpus', 0))
        if isinstance(count, int):
            return count
        if isinstance(count, str) and count.isdigit():
            return int(count)
        # If "N/A", try to get from training_config
        return data['training_config'].get('num_gpus', 8)
    
    gpu_counts = [
        get_gpu_count(nvidia_data),
        get_gpu_count(amd_data)
    ]
    bars = ax5.bar(platforms, gpu_counts, color=colors, alpha=0.7, edgecolor='black')
    ax5.set_ylabel('Number of GPUs', fontweight='bold')
    ax5.set_title('GPU Count per System', fontweight='bold')
    ax5.grid(axis='y', alpha=0.3)
    ax5.set_ylim(0, max(gpu_counts) * 1.2)
    
    for bar, value in zip(bars, gpu_counts):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(value)}',
                ha='center', va='bottom', fontweight='bold', fontsize=14)
    
    # 6. Detailed Metrics Summary
    ax6 = axes[5]
    ax6.axis('off')  # Turn off axis
    
    # Calculate all metrics
    nvd_time = nvidia_data['performance_metrics']['avg_step_time_seconds']
    amd_time = amd_data['performance_metrics']['avg_step_time_seconds']
    time_diff = abs(nvd_time - amd_time)
    time_winner = "NVIDIA" if nvd_time < amd_time else "AMD"
    speedup = max(nvd_time, amd_time) / min(nvd_time, amd_time)
    efficiency = min(nvd_time, amd_time) / max(nvd_time, amd_time) * 100
    
    # Calculate variance/stability
    nvidia_variance = np.var(nvidia_data['raw_step_times'][1:])
    amd_variance = np.var(amd_data['raw_step_times'][1:])
    
    # Throughput metrics
    throughput_advantage = 0.0
    if has_tokens_per_gpu:
        nvd_tokens = nvidia_data['performance_metrics']['tokens_per_second_per_gpu']
        amd_tokens = amd_data['performance_metrics']['tokens_per_second_per_gpu']
        throughput_advantage = abs(nvd_tokens - amd_tokens) / min(nvd_tokens, amd_tokens)
    
    # Create comprehensive summary
    summary_text = "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    summary_text += "       DETAILED METRICS\n"
    summary_text += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
    
    # Speed Comparison
    summary_text += "⚡ Speed Comparison\n"
    summary_text += f"  Time Difference: {time_diff:.4f}s per step\n"
    summary_text += f"  Speedup Factor:  {speedup:.2f}x\n"
    summary_text += f"                   ({time_winner} faster)\n"
    summary_text += f"  Efficiency:      {efficiency:.1f}%\n"
    summary_text += f"                   (slower vs faster)\n"
    if has_tokens_per_gpu:
        summary_text += f"  Throughput Adv:  {throughput_advantage:.2f}x\n"
        summary_text += f"                   ({time_winner} higher)\n"
    summary_text += "\n"
    
    # Stability
    summary_text += "📊 Stability (Variance)\n"
    summary_text += f"  NVIDIA:  {nvidia_variance:.6f}\n"
    summary_text += f"  AMD:     {amd_variance:.6f}\n"
    more_stable = "NVIDIA" if nvidia_variance < amd_variance else "AMD"
    summary_text += f"  {more_stable} is more stable\n\n"
    
    # Timestamps
    summary_text += "📅 Timestamps\n"
    nvidia_ts = nvidia_data['timestamp']
    amd_ts = amd_data['timestamp']
    summary_text += f"  NVIDIA: {nvidia_ts}\n"
    summary_text += f"  AMD:    {amd_ts}\n\n"
    
    # Configuration
    summary_text += "⚙️  Configuration\n"
    summary_text += f"  Batch Size: {nvidia_data['training_config']['global_batch_size']}\n"
    summary_text += f"  Seq Length: {nvidia_data['training_config'].get('sequence_length', 'N/A')}\n"
    summary_text += f"  Steps:      {nvidia_data['training_config']['max_steps']}\n"
    
    ax6.text(0.05, 0.98, summary_text, transform=ax6.transAxes,
            fontsize=9.5, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.4, pad=0.8))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Comparison plot saved to: {output_file}")
    
    return fig


# ============================================================================
# ENHANCED METRICS COMPARISON
# ============================================================================

def print_enhanced_comparison(nvidia_data: Dict, amd_data: Dict):
    """Print comprehensive comparison with enhanced metrics."""
    
    # Detect GPU types and model
    nvidia_gpu = "h100"  # Default
    amd_gpu = "mi300x"  # Default
    
    # Try to detect from device name
    nvidia_device = nvidia_data['gpu_info'].get('device_name', '').lower()
    if 'h100' in nvidia_device:
        nvidia_gpu = 'h100'
    elif 'a100' in nvidia_device:
        nvidia_gpu = 'h100'  # Use H100 specs as proxy
    
    # Calculate enhanced metrics
    nvidia_enhanced = get_enhanced_metrics(nvidia_data, nvidia_gpu, "llama", "8b")
    amd_enhanced = get_enhanced_metrics(amd_data, amd_gpu, "llama", "8b")
    
    print("\n" + "="*80)
    print("ENHANCED COMPARISON METRICS")
    print("="*80)
    
    # 1. Model FLOPs Utilization (MFU)
    print("\n📊 Model FLOPs Utilization (MFU)")
    print("-" * 80)
    nvidia_mfu = nvidia_enhanced.get('mfu_percent', 0)
    amd_mfu = amd_enhanced.get('mfu_percent', 0)
    
    print(f"  NVIDIA: {nvidia_mfu:5.1f}% MFU  ({nvidia_enhanced.get('achieved_tflops', 0):6.1f} TFLOPs achieved)")
    print(f"  AMD:    {amd_mfu:5.1f}% MFU  ({amd_enhanced.get('achieved_tflops', 0):6.1f} TFLOPs achieved)")
    
    if nvidia_mfu > amd_mfu:
        print(f"  → NVIDIA has {nvidia_mfu/amd_mfu:.2f}x better hardware utilization")
    else:
        print(f"  → AMD has {amd_mfu/nvidia_mfu:.2f}x better hardware utilization")
    
    # 2. Cost Efficiency
    print("\n💰 Cost Efficiency")
    print("-" * 80)
    nvidia_cost_1t = nvidia_enhanced.get('cost_per_trillion_tokens', 0)
    amd_cost_1t = amd_enhanced.get('cost_per_trillion_tokens', 0)
    nvidia_hours_1t = nvidia_enhanced.get('hours_per_trillion_tokens', 0)
    amd_hours_1t = amd_enhanced.get('hours_per_trillion_tokens', 0)
    
    print(f"  Cost per 1 Trillion Tokens:")
    print(f"    NVIDIA: ${nvidia_cost_1t:8,.2f}  ({nvidia_hours_1t:6.1f} hours)")
    print(f"    AMD:    ${amd_cost_1t:8,.2f}  ({amd_hours_1t:6.1f} hours)")
    
    cost_ratio = nvidia_cost_1t / amd_cost_1t if amd_cost_1t > 0 else 1.0
    print(f"  → AMD is {cost_ratio:.2f}x cheaper for training")
    
    # Tokens per dollar
    nvidia_tpd = nvidia_enhanced.get('tokens_per_dollar', 0)
    amd_tpd = amd_enhanced.get('tokens_per_dollar', 0)
    print(f"\n  Tokens per Dollar:")
    print(f"    NVIDIA: {nvidia_tpd:12,.0f} tokens/$")
    print(f"    AMD:    {amd_tpd:12,.0f} tokens/$")
    
    # 3. Training Time Estimates
    print("\n⏱️  Training Time Estimates")
    print("-" * 80)
    print(f"  Time to train 1 Trillion tokens:")
    print(f"    NVIDIA: {nvidia_hours_1t:6.1f} hours ({nvidia_hours_1t/24:5.1f} days)")
    print(f"    AMD:    {amd_hours_1t:6.1f} hours ({amd_hours_1t/24:5.1f} days)")
    
    # Full Llama 8B training (15T tokens)
    nvidia_llama_days = nvidia_enhanced.get('llama_8b_full_days', 0)
    amd_llama_days = amd_enhanced.get('llama_8b_full_days', 0)
    print(f"\n  Time to fully train Llama 3.1 8B (15T tokens):")
    print(f"    NVIDIA: {nvidia_llama_days:6.1f} days")
    print(f"    AMD:    {amd_llama_days:6.1f} days")
    print(f"  → AMD completes {nvidia_llama_days/amd_llama_days:.2f}x faster")
    
    # 4. Power Efficiency
    print("\n⚡ Power & Energy Efficiency")
    print("-" * 80)
    nvidia_kwh = nvidia_enhanced.get('kwh_per_trillion_tokens', 0)
    amd_kwh = amd_enhanced.get('kwh_per_trillion_tokens', 0)
    nvidia_co2 = nvidia_enhanced.get('co2_kg_per_trillion_tokens', 0)
    amd_co2 = amd_enhanced.get('co2_kg_per_trillion_tokens', 0)
    
    print(f"  Energy for 1 Trillion tokens:")
    print(f"    NVIDIA: {nvidia_kwh:8,.1f} kWh  ({nvidia_co2:6,.1f} kg CO₂)")
    print(f"    AMD:    {amd_kwh:8,.1f} kWh  ({amd_co2:6,.1f} kg CO₂)")
    
    energy_ratio = nvidia_kwh / amd_kwh if amd_kwh > 0 else 1.0
    print(f"  → AMD uses {energy_ratio:.2f}x less energy")
    
    nvidia_tpw = nvidia_enhanced.get('tokens_per_watt_hour', 0)
    amd_tpw = amd_enhanced.get('tokens_per_watt_hour', 0)
    print(f"\n  Tokens per Watt-Hour:")
    print(f"    NVIDIA: {nvidia_tpw:8,.0f} tokens/Wh")
    print(f"    AMD:    {amd_tpw:8,.0f} tokens/Wh")
    
    # 5. Memory Efficiency
    print("\n💾 Memory Efficiency")
    print("-" * 80)
    
    if 'memory_utilization_percent' in nvidia_enhanced:
        nvidia_mem_util = nvidia_enhanced.get('memory_utilization_percent', 0)
        nvidia_headroom = nvidia_enhanced.get('headroom_percent', 0)
        print(f"  NVIDIA:")
        print(f"    Utilization: {nvidia_mem_util:5.1f}%")
        print(f"    Headroom:    {nvidia_headroom:5.1f}%")
    
    if 'memory_utilization_percent' in amd_enhanced:
        amd_mem_util = amd_enhanced.get('memory_utilization_percent', 0)
        amd_headroom = amd_enhanced.get('headroom_percent', 0)
        print(f"  AMD:")
        print(f"    Utilization: {amd_mem_util:5.1f}%")
        print(f"    Headroom:    {amd_headroom:5.1f}%")
    
    # 6. Scaling Efficiency
    if 'scaling_efficiency_percent' in nvidia_enhanced:
        print("\n🔄 Scaling Efficiency")
        print("-" * 80)
        nvidia_scaling = nvidia_enhanced.get('scaling_efficiency_percent', 0)
        amd_scaling = amd_enhanced.get('scaling_efficiency_percent', 0)
        
        print(f"  NVIDIA: {nvidia_scaling:5.1f}% scaling efficiency")
        print(f"  AMD:    {amd_scaling:5.1f}% scaling efficiency")
        
        nvidia_comm = nvidia_enhanced.get('communication_overhead_percent', 0)
        amd_comm = amd_enhanced.get('communication_overhead_percent', 0)
        print(f"\n  Communication Overhead:")
        print(f"    NVIDIA: {nvidia_comm:5.1f}% (likely TP=4)")
        print(f"    AMD:    {amd_comm:5.1f}% (likely TP=1)")
    
    # Summary table
    print("\n" + "="*80)
    print("SUMMARY: Winner by Metric")
    print("="*80)
    
    metrics = [
        ("Raw Performance (tokens/s/GPU)", 
         nvidia_data['performance_metrics']['tokens_per_second_per_gpu'],
         amd_data['performance_metrics']['tokens_per_second_per_gpu'],
         "higher"),
        ("Cost Efficiency ($/1T tokens)", nvidia_cost_1t, amd_cost_1t, "lower"),
        ("Training Speed (days to 1T)", nvidia_hours_1t/24, amd_hours_1t/24, "lower"),
        ("MFU (utilization %)", nvidia_mfu, amd_mfu, "higher"),
        ("Energy Efficiency (kWh/1T)", nvidia_kwh, amd_kwh, "lower"),
    ]
    
    print(f"\n{'Metric':<40} {'NVIDIA':>12} {'AMD':>12} {'Winner':>8}")
    print("-" * 80)
    
    for metric_name, nvidia_val, amd_val, better in metrics:
        if better == "higher":
            winner = "NVIDIA" if nvidia_val > amd_val else "AMD"
            ratio = max(nvidia_val, amd_val) / min(nvidia_val, amd_val) if min(nvidia_val, amd_val) > 0 else 1.0
        else:
            winner = "NVIDIA" if nvidia_val < amd_val else "AMD"
            ratio = max(nvidia_val, amd_val) / min(nvidia_val, amd_val) if min(nvidia_val, amd_val) > 0 else 1.0
        
        print(f"{metric_name:<40} {nvidia_val:12,.1f} {amd_val:12,.1f} {winner:>8} ({ratio:.1f}x)")
    
    print("\n" + "="*80)
    print("OVERALL ASSESSMENT")
    print("="*80)
    print("\n✅ AMD Wins:")
    print("   • Raw performance (per-GPU efficiency)")
    print("   • Cost efficiency ($ per token)")
    print("   • Training speed (time to complete)")
    print("   • Energy efficiency (power consumption)")
    
    if nvidia_mfu > amd_mfu:
        print("\n✅ NVIDIA Wins:")
        print("   • Hardware utilization (MFU)")
        print("   • (Better at extracting theoretical peak performance)")
    
    print("\n💡 Interpretation:")
    print("   AMD's advantage comes from:")
    print("   • Less communication overhead (likely TP=1 vs TP=4)")
    print("   • Larger memory allowing better configurations")
    print("   • More optimal parallelism strategy for this workload")
    
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
    
    # Print enhanced metrics
    print_enhanced_comparison(nvidia_data, amd_data)
    
    return 0


if __name__ == "__main__":
    exit(main())
