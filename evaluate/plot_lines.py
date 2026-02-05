#!/usr/bin/env python3
"""
Generate separate line chart PNG files for training metrics.

Creates individual chart files:
  - loss.png: Training loss over steps
  - learning_rate.png: Learning rate schedule
  - step_time.png: Step duration over time
  - memory.png: GPU memory usage over time (if available)

Usage:
    python3 plot_lines.py [--input-dir output/] [--output-dir output/]
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib


def load_benchmarks(input_dir: Path) -> dict:
    """Load all benchmark JSON files from directory."""
    benchmarks = {}
    
    patterns = [
        "train_amd_prim_*.json",
        "train_nvd_nemo_*.json",
        "train_amd_*.json",
        "train_nvd_*.json",
    ]
    
    for pattern in patterns:
        for json_file in input_dir.glob(pattern):
            try:
                with open(json_file) as f:
                    data = json.load(f)
                    key = json_file.stem
                    benchmarks[key] = data
                    print(f"  Loaded: {json_file.name}")
            except (json.JSONDecodeError, IOError) as e:
                print(f"  Warning: Could not load {json_file}: {e}")
    
    return benchmarks


def get_style_map():
    """Return consistent style mapping for all charts."""
    return {
        'nvidia-llama': {'color': '#2E86AB', 'marker': 'o', 'label': 'NVIDIA Llama', 'linestyle': '-'},
        'nvidia-qwen':  {'color': '#5DA9E9', 'marker': 's', 'label': 'NVIDIA Qwen', 'linestyle': '--'},
        'amd-llama':    {'color': '#E94F37', 'marker': 'o', 'label': 'AMD Llama', 'linestyle': '-'},
        'amd-qwen':     {'color': '#F28C82', 'marker': 's', 'label': 'AMD Qwen', 'linestyle': '--'},
    }


def find_key(benchmarks: dict, platform: str, model: str) -> str:
    """Find the benchmark key for a given platform and model."""
    for key in benchmarks:
        key_lower = key.lower()
        platform_match = platform in key_lower or (platform == "nvidia" and "nvd" in key_lower)
        model_match = model in key_lower
        if platform_match and model_match:
            return key
    return None


def plot_loss(benchmarks: dict, output_file: Path):
    """Generate loss over steps chart."""
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
    style_map = get_style_map()
    has_data = False
    
    for platform in ("nvidia", "amd"):
        for model_name in ("llama", "qwen"):
            suffix = f"{platform}-{model_name}"
            key = find_key(benchmarks, platform, model_name)
            if key and 'loss_values' in benchmarks[key]:
                loss_values = benchmarks[key]['loss_values']
                if loss_values:
                    steps = range(1, len(loss_values) + 1)
                    style = style_map[suffix]
                    ax.plot(steps, loss_values, 
                            marker=style['marker'], 
                            linestyle=style['linestyle'],
                            color=style['color'], 
                            label=style['label'], 
                            linewidth=1.5, 
                            markersize=3, 
                            alpha=0.85)
                    has_data = True
    
    if has_data:
        ax.set_xlabel('Step', fontweight='bold', fontsize=11)
        ax.set_ylabel('Loss', fontweight='bold', fontsize=11)
        ax.set_title('Training Loss', fontweight='bold', fontsize=14)
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(alpha=0.3, linestyle='--', linewidth=0.5)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"  + Saved: {output_file}")
    else:
        print(f"  - Skipped loss.png (no data)")
    
    plt.close(fig)
    return has_data


def plot_learning_rate(benchmarks: dict, output_file: Path):
    """Generate learning rate schedule chart."""
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
    style_map = get_style_map()
    has_data = False
    
    for platform in ("nvidia", "amd"):
        for model_name in ("llama", "qwen"):
            suffix = f"{platform}-{model_name}"
            key = find_key(benchmarks, platform, model_name)
            if key and 'learning_rates' in benchmarks[key]:
                lr_values = benchmarks[key]['learning_rates']
                if lr_values:
                    steps = range(1, len(lr_values) + 1)
                    style = style_map[suffix]
                    ax.plot(steps, lr_values, 
                            marker=style['marker'], 
                            linestyle=style['linestyle'],
                            color=style['color'], 
                            label=style['label'], 
                            linewidth=1.5, 
                            markersize=3, 
                            alpha=0.85)
                    has_data = True
    
    if has_data:
        ax.set_xlabel('Step', fontweight='bold', fontsize=11)
        ax.set_ylabel('Learning Rate', fontweight='bold', fontsize=11)
        ax.set_title('Learning Rate Schedule', fontweight='bold', fontsize=14)
        ax.legend(fontsize=9, loc='best')
        ax.grid(alpha=0.3, linestyle='--', linewidth=0.5)
        ax.ticklabel_format(axis='y', style='scientific', scilimits=(0,0))
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"  + Saved: {output_file}")
    else:
        print(f"  - Skipped learning_rate.png (no data)")
    
    plt.close(fig)
    return has_data


def plot_step_time(benchmarks: dict, output_file: Path):
    """Generate step duration chart."""
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
    style_map = get_style_map()
    has_data = False
    
    for platform in ("nvidia", "amd"):
        for model_name in ("llama", "qwen"):
            suffix = f"{platform}-{model_name}"
            key = find_key(benchmarks, platform, model_name)
            if key and 'step_times' in benchmarks[key]:
                step_times = benchmarks[key]['step_times']
                if step_times:
                    steps = range(1, len(step_times) + 1)
                    style = style_map[suffix]
                    ax.plot(steps, step_times, 
                            marker=style['marker'], 
                            linestyle=style['linestyle'],
                            color=style['color'], 
                            label=style['label'], 
                            linewidth=1.5, 
                            markersize=3, 
                            alpha=0.85)
                    has_data = True
    
    if has_data:
        ax.set_xlabel('Step', fontweight='bold', fontsize=11)
        ax.set_ylabel('Time (seconds)', fontweight='bold', fontsize=11)
        ax.set_title('Step Duration', fontweight='bold', fontsize=14)
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(alpha=0.3, linestyle='--', linewidth=0.5)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"  + Saved: {output_file}")
    else:
        print(f"  - Skipped step_time.png (no data)")
    
    plt.close(fig)
    return has_data


def plot_memory(benchmarks: dict, output_file: Path):
    """Generate memory usage over time chart."""
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
    style_map = get_style_map()
    has_data = False
    
    for platform in ("nvidia", "amd"):
        for model_name in ("llama", "qwen"):
            suffix = f"{platform}-{model_name}"
            key = find_key(benchmarks, platform, model_name)
            if key and 'memory_values' in benchmarks[key]:
                memory_values = benchmarks[key]['memory_values']
                if memory_values:
                    steps = range(1, len(memory_values) + 1)
                    style = style_map[suffix]
                    ax.plot(steps, memory_values, 
                            marker=style['marker'], 
                            linestyle=style['linestyle'],
                            color=style['color'], 
                            label=style['label'], 
                            linewidth=1.5, 
                            markersize=3, 
                            alpha=0.85)
                    has_data = True
    
    if has_data:
        ax.set_xlabel('Step', fontweight='bold', fontsize=11)
        ax.set_ylabel('Memory (GB)', fontweight='bold', fontsize=11)
        ax.set_title('GPU Memory Usage', fontweight='bold', fontsize=14)
        ax.legend(fontsize=9, loc='best')
        ax.grid(alpha=0.3, linestyle='--', linewidth=0.5)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"  + Saved: {output_file}")
    else:
        print(f"  - Skipped memory.png (no memory_values data)")
    
    plt.close(fig)
    return has_data


def plot_throughput(benchmarks: dict, output_file: Path):
    """Generate throughput bar chart."""
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
    style_map = get_style_map()
    
    labels = []
    values = []
    colors = []
    
    for platform in ("nvidia", "amd"):
        for model_name in ("llama", "qwen"):
            suffix = f"{platform}-{model_name}"
            key = find_key(benchmarks, platform, model_name)
            if key:
                perf = benchmarks[key].get('performance_metrics', {})
                throughput = perf.get('tokens_per_second_per_gpu', 0)
                if throughput:
                    style = style_map[suffix]
                    labels.append(style['label'])
                    values.append(throughput)
                    colors.append(style['color'])
    
    if values:
        bars = ax.bar(labels, values, color=colors, edgecolor='white', linewidth=1.5)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                   f'{val:,.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_ylabel('Tokens/sec/GPU', fontweight='bold', fontsize=11)
        ax.set_title('Throughput Comparison', fontweight='bold', fontsize=14)
        ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"  + Saved: {output_file}")
    else:
        print(f"  - Skipped throughput.png (no data)")
    
    plt.close(fig)
    return bool(values)


def plot_peak_memory(benchmarks: dict, output_file: Path):
    """Generate peak memory bar chart."""
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
    style_map = get_style_map()
    
    labels = []
    values = []
    colors = []
    
    for platform in ("nvidia", "amd"):
        for model_name in ("llama", "qwen"):
            suffix = f"{platform}-{model_name}"
            key = find_key(benchmarks, platform, model_name)
            if key:
                mem = benchmarks[key].get('memory_metrics', {})
                peak = mem.get('peak_memory_allocated_gb', 0)
                if peak:
                    style = style_map[suffix]
                    labels.append(style['label'])
                    values.append(peak)
                    colors.append(style['color'])
    
    if values:
        bars = ax.bar(labels, values, color=colors, edgecolor='white', linewidth=1.5)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                   f'{val:.1f} GB', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_ylabel('Peak Memory (GB)', fontweight='bold', fontsize=11)
        ax.set_title('Peak GPU Memory Usage', fontweight='bold', fontsize=14)
        ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"  + Saved: {output_file}")
    else:
        print(f"  - Skipped peak_memory.png (no data)")
    
    plt.close(fig)
    return bool(values)


def main():
    parser = argparse.ArgumentParser(
        description='Generate separate line chart PNG files for training metrics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--input-dir', '-i', default='output/',
                       help='Directory containing benchmark JSON files (default: output/)')
    parser.add_argument('--output-dir', '-o', default=None,
                       help='Output directory for charts (default: same as input)')
    parser.add_argument('--prefix', '-p', default='',
                       help='Prefix for output filenames (e.g., "run1_")')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else input_dir
    
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        return 1
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading benchmarks from: {input_dir}")
    benchmarks = load_benchmarks(input_dir)
    
    if not benchmarks:
        print("Error: No benchmark files found")
        return 1
    
    print(f"\nGenerating charts in: {output_dir}")
    prefix = args.prefix
    
    # Generate all charts
    charts = [
        ('loss', plot_loss),
        ('learning_rate', plot_learning_rate),
        ('step_time', plot_step_time),
        ('memory', plot_memory),
        ('throughput', plot_throughput),
        ('peak_memory', plot_peak_memory),
    ]
    
    generated = 0
    for name, plot_func in charts:
        output_file = output_dir / f"{prefix}{name}.png"
        if plot_func(benchmarks, output_file):
            generated += 1
    
    print(f"\nGenerated {generated}/{len(charts)} chart files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
