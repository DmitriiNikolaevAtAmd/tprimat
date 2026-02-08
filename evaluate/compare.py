#!/usr/bin/env python3
"""
Universal GPU Benchmark Comparison Script

Automatically discovers and compares any available benchmark results with:
- Dynamic platform/framework/model detection
- Automatic color and style assignment
- Performance metrics (throughput, step time, memory)
- Visual plots and analysis

Usage:
    python3 compare.py [--results-dir ./output]
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict
import matplotlib.pyplot as plt
import numpy as np


# Color palettes for dynamic assignment
PLATFORM_COLORS = {
    'nvidia': ['#27AE60', '#229954', '#1E8449', '#52BE80', '#7DCEA0'],
    'amd': ['#E74C3C', '#C0392B', '#922B21', '#EC7063', '#F1948A'],
    'unknown': ['#9B59B6', '#8E44AD', '#7D3C98', '#AF7AC5', '#D2B4DE'],
}

FRAMEWORK_DISPLAY = {
    'nemo': 'NeMo',
    'mega': 'Megatron',
    'prim': 'Primus',
    'deep': 'DeepSpeed',
    'fsdp': 'FSDP',
    'tran': 'Transformers',
}

MARKERS = ['o', 's', '^', 'D', 'v', 'p', 'h', '*']
LINESTYLES = ['-', '--', '-.', ':']


def load_benchmarks(results_dir: str) -> Dict[str, Dict]:
    """Load all benchmark JSON files from the results directory.
    
    Args:
        results_dir: Directory containing benchmark JSON files
    
    Returns:
        Dict mapping unique keys to benchmark data
    """
    results_path = Path(results_dir)
    benchmarks = {}
    
    for json_file in sorted(results_path.glob("*.json")):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            filename = json_file.stem
            parts = filename.split('_')
            
            # Parse filename: train_{platform}_{framework}_{model}[_{dataset}]
            platform = 'unknown'
            framework = 'unknown'
            model = 'unknown'
            dataset = None
            
            if len(parts) >= 4:
                platform_code = parts[1]
                platform = {'nvd': 'nvidia', 'amd': 'amd'}.get(platform_code, platform_code)
                framework = parts[2]
                model = parts[3]
                if len(parts) >= 5:
                    dataset = parts[4]
            
            # Fallback to JSON content
            if dataset is None:
                dataset = data.get('dataset')
            if platform == 'unknown':
                platform = data.get('platform', 'unknown')
                if platform == 'nvd':
                    platform = 'nvidia'
            
            # Build unique key
            key = f"{platform}-{framework}-{model}"
            if dataset:
                key = f"{key}-{dataset}"
            
            # Store parsed metadata
            data['_platform'] = platform
            data['_framework'] = framework
            data['_model'] = model
            data['_dataset'] = dataset
            data['_key'] = key
            
            benchmarks[key] = data
            
            ds_info = f" ({dataset})" if dataset else ""
            fw_display = FRAMEWORK_DISPLAY.get(framework, framework.upper())
            print(f"[+] Loaded: {platform.upper()} {fw_display} {model.upper()}{ds_info}")
            
        except Exception as e:
            print(f"[!] Error loading {json_file}: {e}")
    
    return benchmarks


def generate_styles(benchmarks: Dict[str, Dict]) -> Dict[str, Dict]:
    """Generate unique styles for each benchmark."""
    styles = {}
    
    # Group by platform to assign colors
    platform_counts = {}
    for key, data in benchmarks.items():
        platform = data.get('_platform', 'unknown')
        platform_counts.setdefault(platform, []).append(key)
    
    for platform, keys in platform_counts.items():
        colors = PLATFORM_COLORS.get(platform, PLATFORM_COLORS['unknown'])
        
        for i, key in enumerate(sorted(keys)):
            data = benchmarks[key]
            framework = data.get('_framework', 'unknown')
            model = data.get('_model', 'unknown')
            fw_display = FRAMEWORK_DISPLAY.get(framework, framework.upper())
            platform_display = platform.upper()
            
            label = f"{platform_display} {fw_display} {model.capitalize()}"
            
            styles[key] = {
                'color': colors[i % len(colors)],
                'marker': MARKERS[i % len(MARKERS)],
                'linestyle': LINESTYLES[i % len(LINESTYLES)],
                'label': label,
            }
    
    return styles


def create_comparison_plot(
    benchmarks: Dict[str, Dict],
    output_file: str,
):
    """Create visual comparison plot for all benchmarks."""
    
    if not benchmarks:
        print("[!] No benchmark data to plot")
        return None
    
    styles = generate_styles(benchmarks)
    ordered_keys = sorted(benchmarks.keys())
    
    # Determine title
    platforms = set(d.get('_platform', 'unknown') for d in benchmarks.values())
    
    if 'nvidia' in platforms and 'amd' in platforms:
        title = 'NVIDIA vs AMD Benchmark Comparison'
    elif 'nvidia' in platforms:
        title = 'NVIDIA Benchmark Results'
    elif 'amd' in platforms:
        title = 'AMD Benchmark Results'
    else:
        title = 'Benchmark Results'
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), facecolor='white')
    fig.suptitle(title, fontsize=18, fontweight='bold', y=0.995, color='#2C3E50')
    axes = axes.flatten()
    
    # Panel 1: Throughput bar chart
    ax1 = axes[0]
    labels, values, colors = [], [], []
    
    for key in ordered_keys:
        perf = benchmarks[key].get('performance_metrics', {})
        tps = perf.get('tokens_per_second_per_gpu')
        if tps:
            labels.append(styles[key]['label'])
            values.append(tps)
            colors.append(styles[key]['color'])
    
    if values:
        bars = ax1.bar(range(len(values)), values, color=colors, alpha=0.75, edgecolor='#333', linewidth=1.2)
        ax1.set_ylabel('Tokens/s/GPU', fontweight='bold', fontsize=11)
        ax1.set_title('Average Per-GPU Throughput', fontweight='bold', fontsize=12)
        ax1.set_xticks(range(len(labels)))
        ax1.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax1.grid(axis='y', alpha=0.2, linestyle='--', linewidth=0.5)
        for bar, val in zip(bars, values):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{val:,.0f}', ha='center', va='bottom', fontweight='bold', fontsize=8)
    else:
        ax1.text(0.5, 0.5, 'Throughput data not available', ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('Average Per-GPU Throughput', fontweight='bold', fontsize=12)
    
    # Panel 2: Memory bar chart — dual bars (allocated vs reserved)
    ax2 = axes[1]
    mem_labels = []
    mem_allocated = []
    mem_reserved = []
    mem_colors = []
    
    for key in ordered_keys:
        mem = benchmarks[key].get('memory_metrics', {})
        alloc = mem.get('avg_memory_allocated_gb')
        resv = mem.get('avg_memory_reserved_gb')
        # Only include if at least one metric exists
        if (alloc and alloc != 'N/A') or (resv and resv != 'N/A'):
            mem_labels.append(styles[key]['label'])
            mem_allocated.append(float(alloc) if alloc and alloc != 'N/A' else 0)
            mem_reserved.append(float(resv) if resv and resv != 'N/A' else 0)
            mem_colors.append(styles[key]['color'])
    
    if mem_labels:
        n = len(mem_labels)
        bar_width = 0.35
        x = np.arange(n)
        
        has_alloc = any(v > 0 for v in mem_allocated)
        has_resv = any(v > 0 for v in mem_reserved)
        
        if has_alloc and has_resv:
            # Dual bars: allocated (solid) + reserved (hatched)
            bars_a = ax2.bar(x - bar_width/2, mem_allocated, bar_width,
                            color=mem_colors, alpha=0.80, edgecolor='#333', linewidth=1.0,
                            label='Allocated')
            bars_r = ax2.bar(x + bar_width/2, mem_reserved, bar_width,
                            color=mem_colors, alpha=0.40, edgecolor='#333', linewidth=1.0,
                            hatch='///', label='Reserved')
            for bar, val in zip(bars_a, mem_allocated):
                if val > 0:
                    ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                            f'{val:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=7)
            for bar, val in zip(bars_r, mem_reserved):
                if val > 0:
                    ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                            f'{val:.1f}', ha='center', va='bottom', fontsize=7, color='#555')
            ax2.legend(fontsize=7, loc='upper right')
        elif has_alloc:
            bars = ax2.bar(x, mem_allocated, bar_width * 2,
                          color=mem_colors, alpha=0.75, edgecolor='#333', linewidth=1.2)
            for bar, val in zip(bars, mem_allocated):
                if val > 0:
                    ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                            f'{val:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=8)
        else:
            bars = ax2.bar(x, mem_reserved, bar_width * 2,
                          color=mem_colors, alpha=0.50, edgecolor='#333', linewidth=1.2, hatch='///')
            for bar, val in zip(bars, mem_reserved):
                if val > 0:
                    ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                            f'{val:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=8)
        
        ax2.set_ylabel('Memory (GB)', fontweight='bold', fontsize=11)
        ax2.set_title('GPU Memory Usage', fontweight='bold', fontsize=12)
        ax2.set_xticks(x)
        ax2.set_xticklabels(mem_labels, rotation=45, ha='right', fontsize=7)
        ax2.grid(axis='y', alpha=0.2, linestyle='--', linewidth=0.5)
    else:
        ax2.text(0.5, 0.5, 'Memory data not available', ha='center', va='center', transform=ax2.transAxes, fontsize=10)
        ax2.set_title('GPU Memory Usage', fontweight='bold', fontsize=12)
    
    # Panel 3: Loss over time
    ax3 = axes[2]
    has_data = False
    for key in ordered_keys:
        loss = benchmarks[key].get('loss_values', [])
        if loss:
            s = styles[key]
            ax3.plot(range(len(loss)), loss, marker=s['marker'], linestyle=s['linestyle'],
                    color=s['color'], label=s['label'], linewidth=1.5, markersize=2, alpha=0.85)
            has_data = True
    
    if has_data:
        ax3.set_xlabel('Step', fontweight='bold', fontsize=10)
        ax3.set_ylabel('Loss', fontweight='bold', fontsize=11)
        ax3.set_title('Training Loss over Time', fontweight='bold', fontsize=12)
        ax3.legend(fontsize=7, loc='best')
        ax3.grid(alpha=0.2, linestyle='--', linewidth=0.5)
    else:
        ax3.text(0.5, 0.5, 'Loss data not available', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Training Loss over Time', fontweight='bold', fontsize=12)
    
    # Panel 4: Learning rate over time
    ax4 = axes[3]
    has_data = False
    for key in ordered_keys:
        lr = benchmarks[key].get('learning_rates', [])
        if lr:
            s = styles[key]
            ax4.plot(range(len(lr)), lr, marker=s['marker'], linestyle=s['linestyle'],
                    color=s['color'], label=s['label'], linewidth=1.5, markersize=2, alpha=0.85)
            has_data = True
    
    if has_data:
        ax4.set_xlabel('Step', fontweight='bold', fontsize=10)
        ax4.set_ylabel('Learning Rate', fontweight='bold', fontsize=11)
        ax4.set_title('Learning Rate over Time', fontweight='bold', fontsize=12)
        ax4.legend(fontsize=7, loc='best')
        ax4.grid(alpha=0.2, linestyle='--', linewidth=0.5)
        ax4.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    else:
        ax4.text(0.5, 0.5, 'Learning rate data not available', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Learning Rate over Time', fontweight='bold', fontsize=12)
    
    # Panel 5: Step time over time
    ax5 = axes[4]
    has_data = False
    for key in ordered_keys:
        times = benchmarks[key].get('step_times', [])
        if times:
            s = styles[key]
            ax5.plot(range(len(times)), times, marker=s['marker'], linestyle=s['linestyle'],
                    color=s['color'], label=s['label'], linewidth=1.5, markersize=2, alpha=0.85)
            avg_time = sum(times) / len(times)
            ax5.axhline(y=avg_time, color=s['color'], linestyle=':', alpha=0.3, linewidth=0.8)
            has_data = True
    
    if has_data:
        ax5.set_xlabel('Step', fontweight='bold', fontsize=10)
        ax5.set_ylabel('Time (secs)', fontweight='bold', fontsize=11)
        ax5.set_title('Step Duration over Time', fontweight='bold', fontsize=12)
        ax5.legend(fontsize=7, loc='best')
        ax5.grid(alpha=0.2, linestyle='--', linewidth=0.5)
    else:
        ax5.text(0.5, 0.5, 'Step time data not available', ha='center', va='center', transform=ax5.transAxes)
        ax5.set_title('Step Duration over Time', fontweight='bold', fontsize=12)
    
    # Panel 6: Configuration table
    ax6 = axes[5]
    ax6.axis('off')
    ax6.set_title('Experiment Configuration', fontweight='bold', fontsize=12)
    
    table_data = []
    for key in ordered_keys:
        data = benchmarks[key]
        config = data.get('training_config', {})
        parallel = data.get('parallelism_config', data.get('parallelism', {}))
        gpu_info = data.get('gpu_info', {})
        
        tp = parallel.get('tensor_model_parallel_size', parallel.get('tensor_parallel_size', config.get('tensor_parallel', 1)))
        pp = parallel.get('pipeline_model_parallel_size', parallel.get('pipeline_parallel_size', config.get('pipeline_parallel', 1)))
        num_gpus = config.get('num_gpus', gpu_info.get('device_count', 8))
        dp = parallel.get('data_parallel_size', num_gpus // max(tp * pp, 1))
        
        gbs = config.get('global_batch_size', 'N/A')
        mbs = config.get('micro_batch_size', 1)
        seq = config.get('sequence_length', 'N/A')
        
        # Truncate label for table
        label = styles[key]['label']
        if len(label) > 22:
            label = label[:19] + '...'
        
        table_data.append([label, str(tp), str(pp), str(dp), str(mbs), str(gbs), str(seq)])
    
    if table_data:
        col_labels = ['Config', 'TP', 'PP', 'DP', 'MBS', 'GBS', 'SL']
        table = ax6.table(cellText=table_data, colLabels=col_labels, loc='center',
                         cellLoc='center', colColours=['#E8E8E8'] * len(col_labels))
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.4, 1.5)
        
        col_widths = [0.30, 0.08, 0.08, 0.08, 0.08, 0.10, 0.10]
        for i, width in enumerate(col_widths):
            for row in range(len(table_data) + 1):
                table[(row, i)].set_width(width)
        for j in range(len(col_labels)):
            table[(0, j)].set_text_props(fontweight='bold')
    else:
        ax6.text(0.5, 0.5, 'No configuration data', ha='center', va='center', transform=ax6.transAxes)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  + Plot saved to: {output_file}")
    
    return fig


def print_summary(benchmarks: Dict[str, Dict]):
    """Print performance summary for all benchmarks."""
    
    print("\n" + "=" * 100)
    print("PERFORMANCE SUMMARY")
    print("=" * 100)
    
    print(f"\n{'Configuration':<40} {'Tokens/s/GPU':>14} {'Step Time':>12} {'Final Loss':>12}")
    print("-" * 100)
    
    for key in sorted(benchmarks.keys()):
        data = benchmarks[key]
        perf = data.get('performance_metrics', {})
        tps = perf.get('tokens_per_second_per_gpu', 0)
        step_time = perf.get('avg_step_time_seconds', 0)
        loss = data.get('loss_values', [])
        final_loss = loss[-1] if loss else 0
        
        platform = data.get('_platform', 'unknown').upper()
        framework = FRAMEWORK_DISPLAY.get(data.get('_framework', ''), data.get('_framework', 'unknown'))
        model = data.get('_model', 'unknown').capitalize()
        
        label = f"{platform} {framework} {model}"
        print(f"{label:<40} {tps:>14,.0f} {step_time:>12.3f}s {final_loss:>12.4f}")
    
    # Best performers
    print("\n" + "-" * 100)
    print("BEST PERFORMERS")
    print("-" * 100)
    
    # Best throughput
    best_tps = max(benchmarks.items(),
                   key=lambda x: x[1].get('performance_metrics', {}).get('tokens_per_second_per_gpu', 0))
    tps_val = best_tps[1].get('performance_metrics', {}).get('tokens_per_second_per_gpu', 0)
    print(f"\n  Highest Throughput: {best_tps[0]}")
    print(f"    {tps_val:,.0f} tokens/s/GPU")
    
    # Fastest step
    best_step = min(benchmarks.items(),
                    key=lambda x: x[1].get('performance_metrics', {}).get('avg_step_time_seconds', float('inf')))
    step_val = best_step[1].get('performance_metrics', {}).get('avg_step_time_seconds', 0)
    print(f"\n  Fastest Step Time: {best_step[0]}")
    print(f"    {step_val:.3f}s per step")
    
    # Cross-platform comparison if both exist
    platforms = set(d.get('_platform') for d in benchmarks.values())
    if 'nvidia' in platforms and 'amd' in platforms:
        print("\n" + "-" * 100)
        print("CROSS-PLATFORM COMPARISON")
        print("-" * 100)
        
        nvidia_best = max(
            (d for d in benchmarks.values() if d.get('_platform') == 'nvidia'),
            key=lambda x: x.get('performance_metrics', {}).get('tokens_per_second_per_gpu', 0),
            default=None
        )
        amd_best = max(
            (d for d in benchmarks.values() if d.get('_platform') == 'amd'),
            key=lambda x: x.get('performance_metrics', {}).get('tokens_per_second_per_gpu', 0),
            default=None
        )
        
        if nvidia_best and amd_best:
            nvidia_tps = nvidia_best.get('performance_metrics', {}).get('tokens_per_second_per_gpu', 0)
            amd_tps = amd_best.get('performance_metrics', {}).get('tokens_per_second_per_gpu', 0)
            
            if nvidia_tps > 0 and amd_tps > 0:
                ratio = amd_tps / nvidia_tps
                winner = "AMD" if ratio > 1 else "NVIDIA"
                ratio_display = ratio if ratio > 1 else 1 / ratio
                
                print(f"\n  NVIDIA best: {nvidia_tps:,.0f} tokens/s/GPU")
                print(f"  AMD best:    {amd_tps:,.0f} tokens/s/GPU")
                print(f"  -> {winner} is {ratio_display:.2f}x faster")
    
    print("\n" + "=" * 100 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Universal GPU benchmark comparison - auto-discovers all results'
    )
    default_dir = os.environ.get('OUTPUT_DIR', './output')
    parser.add_argument('--results-dir', default=default_dir,
                       help='Directory containing benchmark JSON files (default: OUTPUT_DIR or ./output)')
    parser.add_argument('--output', default='compare.png',
                       help='Output filename for the plot (default: compare.png)')
    
    args = parser.parse_args()
    
    print("=" * 100)
    print("UNIVERSAL BENCHMARK COMPARISON")
    print("=" * 100)
    print(f"\nScanning: {args.results_dir}")
    
    results_path = Path(args.results_dir)
    
    benchmarks = load_benchmarks(args.results_dir)
    
    if not benchmarks:
        print("\n  x No benchmarks found")
        print(f"\nExpected files in {args.results_dir}/:")
        print("  Format: train_{platform}_{framework}_{model}[_{dataset}].json")
        print("  Examples:")
        print("    train_nvd_nemo_llama_bc.json")
        print("    train_amd_prim_qwen_c4.json")
        return 1
    
    print(f"\n  Found {len(benchmarks)} benchmark(s)")
    
    output_file = str(results_path / os.path.basename(args.output))
    
    print(f"\nGenerating plot: {output_file}")
    try:
        fig = create_comparison_plot(benchmarks, output_file)
        if fig:
            plt.close(fig)
    except Exception as e:
        print(f"[!] Plot generation failed: {e}")
        import traceback
        traceback.print_exc()
    
    print_summary(benchmarks)
    
    print("\n" + "=" * 100)
    print("GENERATED PLOT")
    print("=" * 100)
    print(f"  + {output_file}")
    print()
    
    return 0


if __name__ == "__main__":
    exit(main())
