#!/usr/bin/env python3
"""
Peak Throughput Visualization

Creates visual analysis of peak throughput across all configurations.

Usage:
    python3 visualize_peak_throughput.py
"""

import glob
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


# Hardware theoretical peak specs
THEORETICAL_PEAKS = {
    "NVIDIA H100 80GB HBM3": 1979,  # TFLOPS
    "AMD Instinct MI300X": 1307,    # TFLOPS
}

# Model FLOPs per token
MODEL_FLOPS_PER_TOKEN = {
    'llama': 6 * 8.0e9,   # 48 billion FLOPs per token
    'qwen': 6 * 7.6e9,    # 45.6 billion FLOPs per token
}


def find_all_benchmark_files():
    """Find all benchmark JSON files."""
    patterns = [
        "output/benchmark_*.json",
        "all_outputs/*/benchmark_*.json",
        "nvd-output/*/benchmark_*.json",
    ]
    
    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern))
    
    return [Path(f) for f in sorted(files)]


def parse_benchmark_file(file_path):
    """Parse benchmark file."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        filename = file_path.stem
        parts = filename.split('_')
        model_name = parts[2] if len(parts) >= 3 else 'unknown'
        config_name = file_path.parent.name if file_path.parent.name != 'output' else 'default'
        
        perf = data.get('performance_metrics', {})
        tps = perf.get('tokens_per_second', 0)
        tps_per_gpu = perf.get('tokens_per_second_per_gpu', 0)
        
        flops_per_token = MODEL_FLOPS_PER_TOKEN.get(model_name, 6 * 7.8e9)
        total_tflops = (tps * flops_per_token) / 1e12 if tps else 0
        tflops_per_gpu = (tps_per_gpu * flops_per_token) / 1e12 if tps_per_gpu else 0
        
        gpu_info = data.get('gpu_info', {})
        device_name = gpu_info.get('device_name', 'Unknown')
        device_count = gpu_info.get('num_gpus', gpu_info.get('device_count', 8))
        
        theoretical_peak = THEORETICAL_PEAKS.get(device_name, 0)
        
        return {
            'config_name': config_name,
            'model_name': model_name,
            'device_name': device_name,
            'device_count': device_count,
            'tokens_per_second': tps,
            'tokens_per_second_per_gpu': tps_per_gpu,
            'total_tflops': total_tflops,
            'tflops_per_gpu': tflops_per_gpu,
            'theoretical_peak': theoretical_peak,
            'hw_utilization': (tflops_per_gpu / theoretical_peak * 100) if theoretical_peak > 0 else 0,
            'avg_step_time': perf.get('avg_step_time_seconds', 0),
        }
    except Exception as e:
        return None


def create_peak_throughput_visualization():
    """Create comprehensive peak throughput visualization."""
    
    # Load all data
    files = find_all_benchmark_files()
    results = []
    
    for file_path in files:
        result = parse_benchmark_file(file_path)
        if result:
            results.append(result)
    
    if not results:
        print("[X] No benchmark data found!")
        return
    
    # Group by platform and model
    platforms = {}
    for r in results:
        key = f"{r['device_name']} - {r['model_name']}"
        if key not in platforms:
            platforms[key] = []
        platforms[key].append(r)
    
    # Find peaks
    peaks = {}
    for key, configs in platforms.items():
        peak = max(configs, key=lambda x: x['tokens_per_second'])
        peaks[key] = peak
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Color scheme
    colors = {
        'NVIDIA H100 80GB HBM3': '#76B900',
        'AMD Instinct MI300X': '#ED1C24',
    }
    
    # 1. Peak Throughput Comparison (Tokens/s)
    ax1 = fig.add_subplot(gs[0, 0])
    labels = []
    values = []
    bar_colors = []
    
    for key in sorted(peaks.keys()):
        peak = peaks[key]
        labels.append(f"{peak['device_name'].split()[0]}\n{peak['model_name']}")
        values.append(peak['tokens_per_second'])
        bar_colors.append(colors.get(peak['device_name'], '#888888'))
    
    bars = ax1.bar(range(len(labels)), values, color=bar_colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylabel('Tokens/second', fontweight='bold', fontsize=11)
    ax1.set_title('Peak Throughput (Tokens/s)', fontweight='bold', fontsize=12)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:,.0f}',
                ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # 2. Peak TFLOPS Comparison
    ax2 = fig.add_subplot(gs[0, 1])
    labels = []
    values = []
    bar_colors = []
    
    for key in sorted(peaks.keys()):
        peak = peaks[key]
        labels.append(f"{peak['device_name'].split()[0]}\n{peak['model_name']}")
        values.append(peak['total_tflops'])
        bar_colors.append(colors.get(peak['device_name'], '#888888'))
    
    bars = ax2.bar(range(len(labels)), values, color=bar_colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylabel('TFLOPS', fontweight='bold', fontsize=11)
    ax2.set_title('Peak Compute (TFLOPS)', fontweight='bold', fontsize=12)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:,.0f}',
                ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # 3. Hardware Utilization
    ax3 = fig.add_subplot(gs[0, 2])
    labels = []
    values = []
    bar_colors = []
    
    for key in sorted(peaks.keys()):
        peak = peaks[key]
        labels.append(f"{peak['device_name'].split()[0]}\n{peak['model_name']}")
        values.append(peak['hw_utilization'])
        bar_colors.append(colors.get(peak['device_name'], '#888888'))
    
    bars = ax3.bar(range(len(labels)), values, color=bar_colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax3.set_xticks(range(len(labels)))
    ax3.set_xticklabels(labels, fontsize=9)
    ax3.set_ylabel('Utilization (%)', fontweight='bold', fontsize=11)
    ax3.set_title('Hardware Utilization', fontweight='bold', fontsize=12)
    ax3.axhline(y=50, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.1f}%',
                ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # 4. Per-GPU Performance
    ax4 = fig.add_subplot(gs[1, 0])
    labels = []
    values = []
    bar_colors = []
    
    for key in sorted(peaks.keys()):
        peak = peaks[key]
        labels.append(f"{peak['device_name'].split()[0]}\n{peak['model_name']}")
        values.append(peak['tokens_per_second_per_gpu'])
        bar_colors.append(colors.get(peak['device_name'], '#888888'))
    
    bars = ax4.bar(range(len(labels)), values, color=bar_colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax4.set_xticks(range(len(labels)))
    ax4.set_xticklabels(labels, fontsize=9)
    ax4.set_ylabel('Tokens/s/GPU', fontweight='bold', fontsize=11)
    ax4.set_title('Per-GPU Throughput', fontweight='bold', fontsize=12)
    ax4.grid(axis='y', alpha=0.3, linestyle='--')
    
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:,.0f}',
                ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # 5. Per-GPU TFLOPS
    ax5 = fig.add_subplot(gs[1, 1])
    labels = []
    values = []
    theoretical = []
    bar_colors = []
    
    for key in sorted(peaks.keys()):
        peak = peaks[key]
        labels.append(f"{peak['device_name'].split()[0]}\n{peak['model_name']}")
        values.append(peak['tflops_per_gpu'])
        theoretical.append(peak['theoretical_peak'])
        bar_colors.append(colors.get(peak['device_name'], '#888888'))
    
    x = np.arange(len(labels))
    width = 0.35
    
    bars1 = ax5.bar(x - width/2, values, width, label='Achieved', color=bar_colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    bars2 = ax5.bar(x + width/2, theoretical, width, label='Theoretical', color='gray', alpha=0.3, edgecolor='black', linewidth=1.5)
    
    ax5.set_xticks(x)
    ax5.set_xticklabels(labels, fontsize=9)
    ax5.set_ylabel('TFLOPS/GPU', fontweight='bold', fontsize=11)
    ax5.set_title('Per-GPU TFLOPS (Achieved vs Theoretical)', fontweight='bold', fontsize=12)
    ax5.legend(fontsize=9)
    ax5.grid(axis='y', alpha=0.3, linestyle='--')
    
    for bar, value in zip(bars1, values):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.0f}',
                ha='center', va='bottom', fontweight='bold', fontsize=8)
    
    # 6. Step Time Comparison
    ax6 = fig.add_subplot(gs[1, 2])
    labels = []
    values = []
    bar_colors = []
    
    for key in sorted(peaks.keys()):
        peak = peaks[key]
        labels.append(f"{peak['device_name'].split()[0]}\n{peak['model_name']}")
        values.append(peak['avg_step_time'])
        bar_colors.append(colors.get(peak['device_name'], '#888888'))
    
    bars = ax6.bar(range(len(labels)), values, color=bar_colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax6.set_xticks(range(len(labels)))
    ax6.set_xticklabels(labels, fontsize=9)
    ax6.set_ylabel('Seconds', fontweight='bold', fontsize=11)
    ax6.set_title('Average Step Time', fontweight='bold', fontsize=12)
    ax6.grid(axis='y', alpha=0.3, linestyle='--')
    
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.2f}s',
                ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # 7-9. Configuration Distribution (bottom row)
    # Group by platform
    nvidia_results = [r for r in results if 'NVIDIA' in r['device_name']]
    amd_results = [r for r in results if 'AMD' in r['device_name']]
    
    # 7. NVIDIA Configuration Spread
    ax7 = fig.add_subplot(gs[2, 0])
    if nvidia_results:
        llama_tps = [r['tokens_per_second']/1000 for r in nvidia_results if r['model_name'] == 'llama']
        qwen_tps = [r['tokens_per_second']/1000 for r in nvidia_results if r['model_name'] == 'qwen']
        
        x = np.arange(2)
        bp = ax7.boxplot([llama_tps, qwen_tps], positions=x, widths=0.5,
                          patch_artist=True, showfliers=True)
        
        for patch in bp['boxes']:
            patch.set_facecolor('#76B900')
            patch.set_alpha(0.6)
        
        ax7.set_xticks(x)
        ax7.set_xticklabels(['Llama', 'Qwen'])
        ax7.set_ylabel('Tokens/s (thousands)', fontweight='bold', fontsize=11)
        ax7.set_title('NVIDIA Config Distribution', fontweight='bold', fontsize=12)
        ax7.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 8. AMD Configuration Spread
    ax8 = fig.add_subplot(gs[2, 1])
    if amd_results:
        llama_tps = [r['tokens_per_second']/1000 for r in amd_results if r['model_name'] == 'llama']
        qwen_tps = [r['tokens_per_second']/1000 for r in amd_results if r['model_name'] == 'qwen']
        
        x = np.arange(2)
        bp = ax8.boxplot([llama_tps, qwen_tps], positions=x, widths=0.5,
                         patch_artist=True, showfliers=True)
        
        for patch in bp['boxes']:
            patch.set_facecolor('#ED1C24')
            patch.set_alpha(0.6)
        
        ax8.set_xticks(x)
        ax8.set_xticklabels(['Llama', 'Qwen'])
        ax8.set_ylabel('Tokens/s (thousands)', fontweight='bold', fontsize=11)
        ax8.set_title('AMD Config Distribution', fontweight='bold', fontsize=12)
        ax8.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 9. Platform Comparison Summary
    ax9 = fig.add_subplot(gs[2, 2])
    
    # Get best by platform
    nvidia_best = max([r['tokens_per_second'] for r in nvidia_results]) if nvidia_results else 0
    amd_best = max([r['tokens_per_second'] for r in amd_results]) if amd_results else 0
    
    platforms_list = []
    best_tps = []
    platform_colors = []
    
    if nvidia_best > 0:
        platforms_list.append('NVIDIA\nH100')
        best_tps.append(nvidia_best)
        platform_colors.append('#76B900')
    
    if amd_best > 0:
        platforms_list.append('AMD\nMI300X')
        best_tps.append(amd_best)
        platform_colors.append('#ED1C24')
    
    bars = ax9.bar(range(len(platforms_list)), best_tps, color=platform_colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax9.set_xticks(range(len(platforms_list)))
    ax9.set_xticklabels(platforms_list, fontsize=11, fontweight='bold')
    ax9.set_ylabel('Peak Tokens/s', fontweight='bold', fontsize=11)
    ax9.set_title('Best Platform Performance', fontweight='bold', fontsize=12)
    ax9.grid(axis='y', alpha=0.3, linestyle='--')
    
    for bar, value in zip(bars, best_tps):
        height = bar.get_height()
        ax9.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:,.0f}',
                ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # Add winner annotation
    if len(best_tps) == 2:
        winner_idx = 0 if best_tps[0] > best_tps[1] else 1
        speedup = max(best_tps) / min(best_tps)
        ax9.text(0.5, 0.95, f'{speedup:.2f}x faster',
                transform=ax9.transAxes, ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3),
                fontweight='bold', fontsize=10)
    
    # Add main title and metadata
    overall_best = max([r['tokens_per_second'] for r in results])
    fig.suptitle(f'Peak Throughput Analysis - Overall Best: {overall_best:,.0f} tokens/s',
                fontsize=16, fontweight='bold', y=0.995)
    
    plt.savefig('peak_throughput_analysis.png', dpi=300, bbox_inches='tight')
    print(f"[OK] Visualization saved to: peak_throughput_analysis.png")
    
    return fig


if __name__ == "__main__":
    create_peak_throughput_visualization()
