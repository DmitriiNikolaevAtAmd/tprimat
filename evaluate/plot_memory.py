#!/usr/bin/env python3
"""
Plot per-step memory consumption from training results.

Usage:
    python3 evaluate/plot_memory.py [--results-dir output] [--output memory_plot.png]
"""
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def load_results(results_dir: str) -> dict:
    """Load all training results with memory data."""
    results_path = Path(results_dir)
    results = {}
    
    for json_file in sorted(results_path.glob("train_*.json")):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Extract key info from filename
            filename = json_file.stem  # e.g., "train_amd_prim_llama"
            results[filename] = data
            
        except Exception as e:
            print(f"Warning: Could not load {json_file}: {e}")
    
    return results


def plot_memory_timeline(results: dict, output_file: str):
    """Plot memory usage over training steps."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('GPU Memory Usage Analysis', fontsize=14, fontweight='bold')
    
    # Color scheme
    colors = {
        'amd': '#E74C3C',
        'nvd': '#3498DB',
        'nvidia': '#3498DB',
    }
    
    # Plot 1: Per-step memory timeline
    ax1 = axes[0, 0]
    has_timeline_data = False
    
    for name, data in results.items():
        if 'memory_values' in data and data['memory_values']:
            memory_values = data['memory_values']
            steps = range(len(memory_values))
            
            # Determine color from platform
            platform = data.get('platform', 'unknown')
            color = colors.get(platform, '#95A5A6')
            
            ax1.plot(steps, memory_values, label=name, color=color, linewidth=1.5, alpha=0.8)
            has_timeline_data = True
    
    if has_timeline_data:
        ax1.set_xlabel('Training Step')
        ax1.set_ylabel('Memory (GB)')
        ax1.set_title('Memory Usage Over Time')
        ax1.legend(fontsize=8)
        ax1.grid(alpha=0.3)
    else:
        ax1.text(0.5, 0.5, 'No per-step memory data available\n\nRun training with memory tracking enabled',
                ha='center', va='center', transform=ax1.transAxes, fontsize=10)
        ax1.set_title('Memory Usage Over Time')
    
    # Plot 2: Peak memory comparison (bar chart)
    ax2 = axes[0, 1]
    names = []
    peaks = []
    bar_colors = []
    
    for name, data in results.items():
        mem_metrics = data.get('memory_metrics', {})
        peak = mem_metrics.get('peak_memory_allocated_gb')
        if peak:
            names.append(name.replace('train_', ''))
            peaks.append(peak)
            platform = data.get('platform', 'unknown')
            bar_colors.append(colors.get(platform, '#95A5A6'))
    
    if peaks:
        bars = ax2.bar(names, peaks, color=bar_colors, alpha=0.75, edgecolor='black')
        ax2.set_ylabel('Peak Memory (GB)')
        ax2.set_title('Peak GPU Memory by Configuration')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, val in zip(bars, peaks):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    else:
        ax2.text(0.5, 0.5, 'No peak memory data available',
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Peak GPU Memory by Configuration')
    
    # Plot 3: Memory vs Throughput scatter
    ax3 = axes[1, 0]
    scatter_data = []
    
    for name, data in results.items():
        mem_metrics = data.get('memory_metrics', {})
        perf_metrics = data.get('performance_metrics', {})
        
        peak_mem = mem_metrics.get('peak_memory_allocated_gb')
        throughput = perf_metrics.get('tokens_per_second_per_gpu')
        
        if peak_mem and throughput:
            platform = data.get('platform', 'unknown')
            scatter_data.append({
                'name': name.replace('train_', ''),
                'memory': peak_mem,
                'throughput': throughput,
                'color': colors.get(platform, '#95A5A6'),
            })
    
    if scatter_data:
        for item in scatter_data:
            ax3.scatter(item['memory'], item['throughput'], 
                       c=item['color'], s=150, alpha=0.7, edgecolors='black')
            ax3.annotate(item['name'], (item['memory'], item['throughput']),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax3.set_xlabel('Peak Memory (GB)')
        ax3.set_ylabel('Throughput (tokens/s/GPU)')
        ax3.set_title('Memory vs Throughput Trade-off')
        ax3.grid(alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'Insufficient data for scatter plot',
                ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Memory vs Throughput Trade-off')
    
    # Plot 4: Summary table
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    table_data = []
    for name, data in results.items():
        mem_metrics = data.get('memory_metrics', {})
        perf_metrics = data.get('performance_metrics', {})
        gpu_info = data.get('gpu_info', {})
        
        peak = mem_metrics.get('peak_memory_allocated_gb', 'N/A')
        avg = mem_metrics.get('avg_memory_allocated_gb', 'N/A')
        throughput = perf_metrics.get('tokens_per_second_per_gpu', 'N/A')
        total_gpu_mem = gpu_info.get('total_memory_gb', 'N/A')
        
        # Calculate utilization
        if isinstance(peak, (int, float)) and isinstance(total_gpu_mem, (int, float)):
            util = f"{100 * peak / total_gpu_mem:.1f}%"
        else:
            util = 'N/A'
        
        table_data.append([
            name.replace('train_', ''),
            f"{peak:.1f}" if isinstance(peak, (int, float)) else peak,
            f"{avg:.1f}" if isinstance(avg, (int, float)) else avg,
            util,
            f"{throughput:,.0f}" if isinstance(throughput, (int, float)) else throughput,
        ])
    
    if table_data:
        table = ax4.table(
            cellText=table_data,
            colLabels=['Config', 'Peak (GB)', 'Avg (GB)', 'Util %', 'Tokens/s/GPU'],
            loc='center',
            cellLoc='center',
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        ax4.set_title('Memory Summary', fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Memory plot saved to: {output_file}")
    
    return fig


def main():
    parser = argparse.ArgumentParser(description='Plot memory consumption from training results')
    parser.add_argument('--results-dir', default='output',
                       help='Directory containing training JSON files')
    parser.add_argument('--output', '-o', default='output/memory_plot.png',
                       help='Output image file')
    
    args = parser.parse_args()
    
    print(f"Loading results from: {args.results_dir}")
    results = load_results(args.results_dir)
    
    if not results:
        print("No training results found!")
        return 1
    
    print(f"Found {len(results)} result file(s):")
    for name, data in results.items():
        mem = data.get('memory_metrics', {})
        peak = mem.get('peak_memory_allocated_gb', 'N/A')
        has_timeline = 'memory_values' in data and len(data.get('memory_values', [])) > 0
        print(f"  - {name}: peak={peak} GB, per-step={'Yes' if has_timeline else 'No'}")
    
    plot_memory_timeline(results, args.output)
    
    return 0


if __name__ == "__main__":
    exit(main())
