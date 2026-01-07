#!/usr/bin/env python3
"""
Compare benchmark results from AMD and NVIDIA GPU runs.
Creates visualization and statistical comparison.
"""
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np


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
            hardware_type = data.get('gpu_info', {}).get('hardware_type', '').lower()
            
            # NVIDIA: cuda, nvd, nvidia
            if platform in ['cuda', 'nvd', 'nvidia'] or hardware_type == 'nvidia':
                nvidia_results.append(data)
            # AMD: rocm, amd
            elif platform in ['rocm', 'amd'] or hardware_type == 'amd':
                amd_results.append(data)
        except Exception as e:
            print(f"⚠️  Error loading {json_file}: {e}")
    
    return nvidia_results, amd_results


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
        ax1.set_title('Tokens/sec/GPU - Per-GPU Efficiency\n(Higher is Better)', 
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
        ax3.set_title('Total System Throughput\n(Higher is Better)', fontweight='bold')
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
        ax3.set_title('Throughput\n(Higher is Better)', fontweight='bold')
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
    gpu_counts = [
        nvidia_data['gpu_info'].get('device_count', nvidia_data['training_config'].get('num_gpus', 0)),
        amd_data['gpu_info'].get('device_count', amd_data['training_config'].get('num_gpus', 0))
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
    
    # 6. Efficiency Summary
    ax6 = axes[5]
    ax6.axis('off')  # Turn off axis
    
    # Create summary text box
    summary_text = "Performance Summary\n" + "="*40 + "\n\n"
    
    # Add tokens/sec/GPU comparison
    if has_tokens_per_gpu:
        nvd_tokens = nvidia_data['performance_metrics']['tokens_per_second_per_gpu']
        amd_tokens = amd_data['performance_metrics']['tokens_per_second_per_gpu']
        ratio = nvd_tokens / amd_tokens if amd_tokens > 0 else 1.0
        winner = "NVIDIA" if ratio > 1 else "AMD"
        summary_text += f"Tokens/sec/GPU:\n"
        summary_text += f"  NVIDIA: {nvd_tokens:,.0f}\n"
        summary_text += f"  AMD:    {amd_tokens:,.0f}\n"
        summary_text += f"  {winner} is {max(ratio, 1/ratio):.2f}x more efficient\n\n"
        
        # Calculate total system throughput
        nvd_total = nvd_tokens * gpu_counts[0]
        amd_total = amd_tokens * gpu_counts[1]
        summary_text += f"Total System Throughput:\n"
        summary_text += f"  NVIDIA: {nvd_total:,.0f} tokens/sec\n"
        summary_text += f"  AMD:    {amd_total:,.0f} tokens/sec\n\n"
    
    # Add step time comparison
    nvd_time = nvidia_data['performance_metrics']['avg_step_time_seconds']
    amd_time = amd_data['performance_metrics']['avg_step_time_seconds']
    time_winner = "NVIDIA" if nvd_time < amd_time else "AMD"
    time_speedup = max(nvd_time, amd_time) / min(nvd_time, amd_time)
    summary_text += f"Step Time:\n"
    summary_text += f"  NVIDIA: {nvd_time:.3f}s\n"
    summary_text += f"  AMD:    {amd_time:.3f}s\n"
    summary_text += f"  {time_winner} is {time_speedup:.2f}x faster\n\n"
    
    # Add configuration
    summary_text += f"Configuration:\n"
    summary_text += f"  Batch Size: {nvidia_data['training_config']['global_batch_size']}\n"
    summary_text += f"  Seq Length: {nvidia_data['training_config'].get('sequence_length', 'N/A')}\n"
    
    ax6.text(0.1, 0.95, summary_text, transform=ax6.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Comparison plot saved to: {output_file}")
    
    return fig


def generate_comparison_report(nvidia_data: Dict, amd_data: Dict, output_file: str = "comparison_report.md"):
    """Generate detailed markdown comparison report."""
    
    nvidia_time = nvidia_data['performance_metrics']['avg_step_time_seconds']
    amd_time = amd_data['performance_metrics']['avg_step_time_seconds']
    
    faster_platform = "NVIDIA" if nvidia_time < amd_time else "AMD"
    speedup = max(nvidia_time, amd_time) / min(nvidia_time, amd_time)
    
    nvidia_throughput = nvidia_data['performance_metrics'].get('tokens_per_second_per_gpu',
                        nvidia_data['performance_metrics'].get('throughput_steps_per_second', 0))
    amd_throughput = amd_data['performance_metrics'].get('tokens_per_second_per_gpu',
                     amd_data['performance_metrics'].get('throughput_steps_per_second', 0))
    throughput_ratio = nvidia_throughput / amd_throughput if amd_throughput else 1.0
    
    report = f"""# AMD vs NVIDIA GPU Benchmark Comparison

## Executive Summary

**Winner**: {faster_platform} is **{speedup:.2f}x faster**

- NVIDIA Throughput: {nvidia_throughput:.1f} tokens/s/GPU
- AMD Throughput: {amd_throughput:.1f} tokens/s/GPU
- Throughput Ratio (NVIDIA/AMD): {throughput_ratio:.2f}x

---

## Hardware Configuration

### NVIDIA GPU
- **Device**: {nvidia_data['gpu_info']['device_name']}
- **GPU Cores**: {nvidia_data['gpu_info'].get('gpu_cores', 'N/A'):,}
- **Total Memory**: {nvidia_data['gpu_info']['total_memory_gb']:.2f} GB
- **PyTorch Version**: {nvidia_data['gpu_info'].get('pytorch_version', 'N/A')}

### AMD GPU
- **Device**: {amd_data['gpu_info']['device_name']}
- **GPU Cores**: {amd_data['gpu_info'].get('gpu_cores', 'N/A'):,}
- **Total Memory**: {amd_data['gpu_info']['total_memory_gb']:.2f} GB
- **PyTorch Version**: {amd_data['gpu_info'].get('pytorch_version', 'N/A')}

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Max Steps | {nvidia_data['training_config']['max_steps']} |
| Global Batch Size | {nvidia_data['training_config']['global_batch_size']} |
| Sequence Length | {nvidia_data['training_config'].get('sequence_length', 'N/A')} |

---

## Performance Metrics

### Step Time

| Platform | Avg Time | Min Time | Max Time | Std Dev |
|----------|----------|----------|----------|---------|
| NVIDIA   | {nvidia_time:.4f}s | {nvidia_data['performance_metrics']['min_step_time_seconds']:.4f}s | {nvidia_data['performance_metrics']['max_step_time_seconds']:.4f}s | {np.std(nvidia_data['raw_step_times'][1:]):.4f}s |
| AMD      | {amd_time:.4f}s | {amd_data['performance_metrics']['min_step_time_seconds']:.4f}s | {amd_data['performance_metrics']['max_step_time_seconds']:.4f}s | {np.std(amd_data['raw_step_times'][1:]):.4f}s |

### Throughput

| Platform | Tokens/sec/GPU | Total Throughput |
|----------|----------------|------------------|
| NVIDIA   | {nvidia_throughput:.1f} | {nvidia_data['performance_metrics'].get('tokens_per_second', 0):,.0f} |
| AMD      | {amd_throughput:.1f} | {amd_data['performance_metrics'].get('tokens_per_second', 0):,.0f} |

"""
    
    # Add memory metrics if available
    if 'memory_metrics' in nvidia_data and 'memory_metrics' in amd_data:
        report += f"""
### Memory Usage

| Platform | Avg Memory | Peak Memory |
|----------|------------|-------------|
| NVIDIA   | {nvidia_data['memory_metrics']['avg_memory_allocated_gb']:.2f} GB | {nvidia_data['memory_metrics']['peak_memory_allocated_gb']:.2f} GB |
| AMD      | {amd_data['memory_metrics']['avg_memory_allocated_gb']:.2f} GB | {amd_data['memory_metrics']['peak_memory_allocated_gb']:.2f} GB |
"""
    
    report += f"""
---

## Detailed Analysis

### Speed Comparison
- **Time Difference**: {abs(nvidia_time - amd_time):.4f} seconds per step
- **Speedup Factor**: {speedup:.2f}x ({faster_platform} faster)
- **Efficiency**: {min(nvidia_time, amd_time) / max(nvidia_time, amd_time) * 100:.1f}% (slower platform relative to faster)
- **Throughput Advantage**: {throughput_ratio:.2f}x ({faster_platform} higher tokens/s/GPU)

### Stability
- **NVIDIA Variance**: {np.var(nvidia_data['raw_step_times'][1:]):.6f}
- **AMD Variance**: {np.var(amd_data['raw_step_times'][1:]):.6f}

---

## Timestamps
- **NVIDIA Run**: {nvidia_data['timestamp']}
- **AMD Run**: {amd_data['timestamp']}

---

*Generated by benchmark_utils.py*
"""
    
    with open(output_file, 'w') as f:
        f.write(report)
    
    print(f"✅ Comparison report saved to: {output_file}")
    return report


def main():
    parser = argparse.ArgumentParser(description='Compare AMD and NVIDIA GPU benchmark results')
    parser.add_argument('--results-dir', default='./outs',
                       help='Directory containing benchmark JSON files')
    parser.add_argument('--output-plot', default='comparison_plot.png',
                       help='Output file for comparison plot')
    parser.add_argument('--output-report', default='comparison_report.md',
                       help='Output file for comparison report')
    
    args = parser.parse_args()
    
    print("Loading benchmark results...")
    nvidia_results, amd_results = load_benchmark_results(args.results_dir)
    
    if not nvidia_results:
        print("❌ No NVIDIA benchmark results found!")
        print(f"   Run your training script on an NVIDIA GPU first.")
        return
    
    if not amd_results:
        print("❌ No AMD benchmark results found!")
        print(f"   Run your training script on an AMD GPU first.")
        return
    
    # Use most recent results
    nvidia_data = sorted(nvidia_results, key=lambda x: x['timestamp'])[-1]
    amd_data = sorted(amd_results, key=lambda x: x['timestamp'])[-1]
    
    print(f"\n📊 Comparing:")
    print(f"  NVIDIA: {nvidia_data['gpu_info']['device_name']} ({nvidia_data['timestamp']})")
    print(f"  AMD:    {amd_data['gpu_info']['device_name']} ({amd_data['timestamp']})")
    print()
    
    # Generate comparison plot
    try:
        create_comparison_plot(nvidia_data, amd_data, args.output_plot)
    except ImportError:
        print("⚠️  matplotlib not available, skipping plot generation")
        print("   Install with: pip install matplotlib")
    except Exception as e:
        print(f"⚠️  Error generating plot: {e}")
    
    # Generate report
    generate_comparison_report(nvidia_data, amd_data, args.output_report)
    
    # Print summary
    nvidia_time = nvidia_data['performance_metrics']['avg_step_time_seconds']
    amd_time = amd_data['performance_metrics']['avg_step_time_seconds']
    faster = "NVIDIA" if nvidia_time < amd_time else "AMD"
    speedup = max(nvidia_time, amd_time) / min(nvidia_time, amd_time)
    
    print(f"\n{'='*60}")
    print(f"🏆 RESULT: {faster} is {speedup:.2f}x FASTER")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

