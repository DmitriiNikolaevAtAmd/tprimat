#!/usr/bin/env python3
"""
GPU Benchmark Comparison Script

Compares AMD and NVIDIA GPU training performance with:
- Performance metrics (throughput, step time, memory)
- Visual plots and analysis

Usage:
    python3 compare.py [--results-dir ./output] [--framework nemo]
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np


def load_all_benchmark_results(results_dir: str) -> Dict[str, Dict]:
    """Load all benchmark results with model and platform information."""
    results_path = Path(results_dir)
    
    benchmarks = {}
    
    for json_file in sorted(results_path.glob("train_*.json")):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            filename = json_file.stem
            parts = filename.split('_')
            platform = "unknown"
            framework = "unknown"
            model_name = "unknown"
            if len(parts) >= 4 and parts[1] in ("nvd", "amd"):
                platform = "nvidia" if parts[1] == "nvd" else "amd"
                framework = parts[2]
                model_name = parts[3]
            elif len(parts) >= 3:
                framework = parts[1]
                model_name = parts[2]
                if framework in ['mega', 'tran', 'deep', 'nemo', 'fsdp']:
                    platform = "nvidia"
                elif framework in ['prim', 'amd']:
                    platform = "amd"
                else:
                    platform = data.get('platform', 'unknown')
            
            if model_name == "unknown" or platform == "unknown":
                print(f"[!] Skipping unrecognized file name: {json_file.name}")
                continue
            
            key = f"{platform}-{framework}-{model_name}"
            data['model_name'] = model_name
            data['platform_key'] = platform
            data['framework'] = framework
            benchmarks[key] = data
            
            print(f"[+] Loaded: {key} from {json_file.name}")
        except Exception as e:
            print(f"[!] Error loading {json_file}: {e}")
    
    return benchmarks


def pick_default_framework(benchmarks: Dict[str, Dict]) -> str | None:
    """Pick a framework with the best cross-platform coverage."""
    nvidia_frameworks = set()
    amd_frameworks = set()
    
    for key, data in benchmarks.items():
        framework = data.get("framework", "unknown")
        platform = data.get("platform_key", "unknown")
        if framework == "unknown" or platform == "unknown":
            continue
        if platform == "nvidia":
            nvidia_frameworks.add(framework)
        elif platform == "amd":
            amd_frameworks.add(framework)
    
    if nvidia_frameworks and amd_frameworks and not nvidia_frameworks.intersection(amd_frameworks):
        return None
    
    frameworks = {}
    for key, data in benchmarks.items():
        framework = data.get("framework", "unknown")
        platform = data.get("platform_key", "unknown")
        model_name = data.get("model_name", "unknown")
        if framework == "unknown" or platform == "unknown" or model_name == "unknown":
            continue
        frameworks.setdefault(framework, set()).add((platform, model_name))
    
    best_framework = None
    best_score = 0
    for framework, combos in frameworks.items():
        has_cross_platform = any(
            ("nvidia", model) in combos and ("amd", model) in combos
            for model in ("llama", "qwen")
        )
        score = len(combos) + (100 if has_cross_platform else 0)
        if score > best_score:
            best_score = score
            best_framework = framework
    
    return best_framework


def create_comparison_plot(
    benchmarks: Dict[str, Dict],
    output_file: str = "compare.png",
    framework_filter: str | None = None,
):
    """Create visual comparison of all platform-model combinations."""
    
    def find_key(platform: str, model_name: str) -> str | None:
        """Find benchmark key for given platform and model, with framework preference."""
        if framework_filter:
            candidate = f"{platform}-{framework_filter}-{model_name}"
            return candidate if candidate in benchmarks else None
        
        matches = [
            key for key in benchmarks.keys()
            if key.startswith(f"{platform}-") and key.endswith(f"-{model_name}")
        ]
        
        if not matches:
            return None
        
        preferred = {"nvidia": "nemo", "amd": "prim"}
        for match in matches:
            if preferred.get(platform, "") in match:
                return match
        
        return sorted(matches)[0]
    
    has_nvidia = any(key.startswith('nvidia-') for key in benchmarks.keys())
    has_amd = any(key.startswith('amd-') for key in benchmarks.keys())
    
    if has_nvidia and has_amd:
        title = 'NVIDIA H100 vs AMD MI300X'
    elif has_nvidia:
        title = 'NVIDIA H100 Benchmark Results'
    elif has_amd:
        title = 'AMD MI300X Benchmark Results'
    else:
        title = 'GPU Benchmark Results'
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), facecolor='white')
    fig.suptitle(title, fontsize=18, fontweight='bold', y=0.995, color='#2C3E50')
    axes = axes.flatten()
    
    style_map = {
        'nvidia-llama': {'color': '#7FB3D5', 'marker': 'o', 'label': 'NVIDIA Llama', 'linestyle': '-'},
        'nvidia-qwen':  {'color': '#85C1E9', 'marker': 's', 'label': 'NVIDIA Qwen', 'linestyle': '--'},
        'amd-llama':    {'color': '#F1948A', 'marker': 'o', 'label': 'AMD Llama', 'linestyle': '-'},
        'amd-qwen':     {'color': '#F5B7B1', 'marker': 's', 'label': 'AMD Qwen', 'linestyle': '--'},
    }
    
    first_data = next(iter(benchmarks.values()))
    config = first_data.get('training_config', {})
    global_batch_size = config.get('global_batch_size', 128)
    seq_length = config.get('sequence_length', 2048)
    num_gpus = config.get('num_gpus', 8)
    
    ax1 = axes[0]
    labels = []
    values = []
    colors_list = []
    
    for platform in ("nvidia", "amd"):
        for model_name in ("llama", "qwen"):
            suffix = f"{platform}-{model_name}"
            key = find_key(platform, model_name)
            if key:
                perf = benchmarks[key]['performance_metrics']
                tps_gpu = perf.get('tokens_per_second_per_gpu')
                if tps_gpu:
                    labels.append(style_map[suffix]['label'])
                    values.append(tps_gpu)
                    colors_list.append(style_map[suffix]['color'])
    
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
    
    ax2 = axes[1]
    labels = []
    values = []
    colors_list = []
    
    for platform in ("nvidia", "amd"):
        for model_name in ("llama", "qwen"):
            suffix = f"{platform}-{model_name}"
            key = find_key(platform, model_name)
            if key:
                data = benchmarks[key]
                mem_metrics = data.get('memory_metrics', {})
                # Use peak memory (most relevant for comparison)
                peak_mem = mem_metrics.get('peak_memory_allocated_gb')
                if peak_mem is None:
                    peak_mem = mem_metrics.get('avg_memory_allocated_gb')
                # NOTE: Don't estimate from total GPU memory - that's misleading
                
                if peak_mem and peak_mem != 'N/A':
                    labels.append(style_map[suffix]['label'])
                    values.append(float(peak_mem))
                    colors_list.append(style_map[suffix]['color'])
    
    if values:
        bars = ax2.bar(labels, values, color=colors_list, alpha=0.75, edgecolor='#333333', linewidth=1.2)
        ax2.set_ylabel('Memory (GB)', fontweight='bold', fontsize=11)
        ax2.set_title('Peak GPU Memory Usage', fontweight='bold', fontsize=12)
        ax2.grid(axis='y', alpha=0.2, linestyle='--', linewidth=0.5)
        
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.1f}',
                    ha='center', va='bottom', fontweight='bold', fontsize=9)
    else:
        ax2.text(0.5, 0.5, 'Memory data not available\n(run with memory tracking enabled)', 
                ha='center', va='center', transform=ax2.transAxes, fontsize=10)
        ax2.set_title('Peak GPU Memory Usage', fontweight='bold', fontsize=12)
    
    ax3 = axes[2]
    has_data = False
    
    for platform in ("nvidia", "amd"):
        for model_name in ("llama", "qwen"):
            suffix = f"{platform}-{model_name}"
            key = find_key(platform, model_name)
            if key and 'loss_values' in benchmarks[key]:
                loss_values = benchmarks[key]['loss_values']
                if loss_values:
                    steps = range(len(loss_values))
                    style = style_map[suffix]
                    ax3.plot(steps, loss_values, 
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
        ax3.set_ylabel('Loss', fontweight='bold', fontsize=11)
        ax3.set_title('Training Loss over Time', fontweight='bold', fontsize=12)
        ax3.legend(fontsize=8, loc='best')
        ax3.grid(alpha=0.2, linestyle='--', linewidth=0.5)
    else:
        ax3.text(0.5, 0.5, 'Loss data not available', 
                ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Training Loss over Time', fontweight='bold', fontsize=12)
    
    ax4 = axes[3]
    has_data = False
    
    for platform in ("nvidia", "amd"):
        for model_name in ("llama", "qwen"):
            suffix = f"{platform}-{model_name}"
            key = find_key(platform, model_name)
            if key and 'learning_rates' in benchmarks[key]:
                lr_values = benchmarks[key]['learning_rates']
                if lr_values:
                    steps = range(len(lr_values))
                    style = style_map[suffix]
                    ax4.plot(steps, lr_values, 
                            marker=style['marker'], 
                            linestyle=style['linestyle'],
                            color=style['color'], 
                            label=style['label'], 
                            linewidth=1.5, 
                            markersize=2, 
                            alpha=0.85)
                    has_data = True
    
    if has_data:
        ax4.set_xlabel('Step', fontweight='bold', fontsize=10)
        ax4.set_ylabel('Learning Rate', fontweight='bold', fontsize=11)
        ax4.set_title('Learning Rate over Time', fontweight='bold', fontsize=12)
        ax4.legend(fontsize=8, loc='best')
        ax4.grid(alpha=0.2, linestyle='--', linewidth=0.5)
        ax4.ticklabel_format(axis='y', style='scientific', scilimits=(0,0))
    else:
        ax4.text(0.5, 0.5, 'Learning rate data not available', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Learning Rate over Time', fontweight='bold', fontsize=12)
    
    ax5 = axes[4]
    has_data = False
    
    for platform in ("nvidia", "amd"):
        for model_name in ("llama", "qwen"):
            suffix = f"{platform}-{model_name}"
            key = find_key(platform, model_name)
            if key and 'step_times' in benchmarks[key]:
                step_times = benchmarks[key]['step_times']
                if step_times:
                    steps = range(len(step_times))
                    style = style_map[suffix]
                    ax5.plot(steps, step_times, 
                            marker=style['marker'], 
                            linestyle=style['linestyle'],
                            color=style['color'], 
                            label=style['label'], 
                            linewidth=1.5, 
                            markersize=2, 
                            alpha=0.85)
                    
                    avg_time = sum(step_times) / len(step_times)
                    ax5.axhline(y=avg_time, color=style['color'], linestyle=':', alpha=0.3, linewidth=0.8)
                    has_data = True
    
    if has_data:
        ax5.set_xlabel('Step', fontweight='bold', fontsize=10)
        ax5.set_ylabel('Time (secs)', fontweight='bold', fontsize=11)
        ax5.set_title('Step Duration over Time', fontweight='bold', fontsize=12)
        ax5.legend(fontsize=8, loc='best')
        ax5.grid(alpha=0.2, linestyle='--', linewidth=0.5)
    else:
        ax5.text(0.5, 0.5, 'Step time data not available', 
                ha='center', va='center', transform=ax5.transAxes)
        ax5.set_title('Step Duration over Time', fontweight='bold', fontsize=12)
    
    ax6 = axes[5]
    ax6.axis('off')
    ax6.set_title('Experiment Configuration', fontweight='bold', fontsize=12)
    
    table_data = []
    
    for platform in ("nvidia", "amd"):
        for model_name in ("llama", "qwen"):
            key = find_key(platform, model_name)
            if key:
                data = benchmarks[key]
                config = data.get('training_config', {})
                parallel = data.get('parallelism_config', data.get('parallelism', {}))
                gpu_info = data.get('gpu_info', {})
                
                platform_label = "NVIDIA" if platform == "nvidia" else "AMD"
                model_label = model_name.capitalize()
                
                tp = parallel.get('tensor_model_parallel_size', parallel.get('tensor_parallel_size', 1))
                pp = parallel.get('pipeline_model_parallel_size', parallel.get('pipeline_parallel_size', 1))
                num_gpus = config.get('num_gpus', gpu_info.get('device_count', 8))
                dp = parallel.get('data_parallel_size', num_gpus // (tp * pp) if tp and pp else 'N/A')
                
                gbs = config.get('global_batch_size', 'N/A')
                mbs = config.get('micro_batch_size', 1)
                seq = config.get('sequence_length', 'N/A')
                gpus = num_gpus
                
                table_data.append([
                    f"{platform_label} {model_label}",
                    f"TP={tp}, PP={pp}, DP={dp}",
                    f"{gbs}",
                    f"{mbs}",
                    f"{seq}",
                    f"{gpus}"
                ])
    
    if table_data:
        col_labels = ['Config', 'Parallelism', 'GBS', 'MBS', 'SeqLen', 'GPUs']
        
        table = ax6.table(
            cellText=table_data,
            colLabels=col_labels,
            loc='center',
            cellLoc='center',
            colColours=['#E8E8E8'] * len(col_labels),
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.4, 1.8)
        
        col_widths = [0.18, 0.25, 0.12, 0.12, 0.13, 0.10]
        for i, width in enumerate(col_widths):
            for row in range(len(table_data) + 1):
                table[(row, i)].set_width(width)
        
        for j in range(len(col_labels)):
            table[(0, j)].set_text_props(fontweight='bold')
    else:
        ax6.text(0.5, 0.5, 'No experiment data available', 
                ha='center', va='center', transform=ax6.transAxes)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  + Comparison plot saved to: {output_file}")
    
    return fig


def print_comparison(nvidia_data: Dict, amd_data: Dict):
    """Print comprehensive comparison of benchmark metrics."""
    
    print("\n" + "="*80)
    print("BENCHMARK COMPARISON")
    print("="*80)
    
    nvidia_perf = nvidia_data['performance_metrics']
    amd_perf = amd_data['performance_metrics']
    nvidia_gpu = nvidia_data['gpu_info']
    amd_gpu = amd_data['gpu_info']
    
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
    
    print("\n  Hardware Configuration")
    print("-" * 80)
    print(f"  NVIDIA: {nvidia_gpu.get('device_name', 'Unknown')} ({nvidia_gpu.get('device_count', 0)} GPUs)")
    print(f"  AMD:    {amd_gpu.get('device_name', 'Unknown')} ({amd_gpu.get('device_count', 0)} GPUs)")
    
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


def main():
    parser = argparse.ArgumentParser(
        description='GPU benchmark comparison - all models and platforms'
    )
    default_dir = os.environ.get('OUTPUT_DIR', './output')
    parser.add_argument('--results-dir', default=default_dir,
                       help='Directory containing benchmark JSON files (default: OUTPUT_DIR env var or ./output)')
    parser.add_argument('--output', default='compare.png',
                       help='Output filename for the comparison plot (default: compare.png)')
    parser.add_argument('--framework', default=None,
                       help='Framework filter (e.g., nemo, deep, fsdp). Auto-detects if omitted.')
    
    args = parser.parse_args()
    
    print("Loading benchmark results...")
    benchmarks = load_all_benchmark_results(args.results_dir)
    
    if not benchmarks:
        print("  x No benchmark results found!")
        print(f"Expected files in {args.results_dir}/:")
        print("  NVIDIA frameworks:")
        print("    - train_nvd_mega_llama.json, train_nvd_mega_qwen.json (Megatron)")
        print("    - train_nvd_tran_llama.json, train_nvd_tran_qwen.json (Transformers)")
        print("    - train_nvd_deep_llama.json, train_nvd_deep_qwen.json (DeepSpeed)")
        print("    - train_nvd_nemo_llama.json, train_nvd_nemo_qwen.json (NeMo)")
        print("  AMD frameworks:")
        print("    - train_amd_nemo_llama.json, train_amd_nemo_qwen.json (NeMo)")
        print("    - train_amd_prim_llama.json, train_amd_prim_qwen.json (Primus)")
        return 1
    
    # Detect which platforms are available
    has_nvidia = any(key.startswith('nvidia-') for key in benchmarks.keys())
    has_amd = any(key.startswith('amd-') for key in benchmarks.keys())
    framework_filter = args.framework or pick_default_framework(benchmarks)
    
    print(f"\n  * Found {len(benchmarks)} benchmark(s):")
    for key in sorted(benchmarks.keys()):
        data = benchmarks[key]
        print(f"  {key}: {data['gpu_info']['device_name']} ({data['timestamp']})")
    
    print(f"\n  * Platform availability:")
    print(f"  NVIDIA: {'+ Available' if has_nvidia else 'x Not available'}")
    print(f"  AMD:    {'+ Available' if has_amd else 'x Not available'}")
    
    if not has_nvidia and not has_amd:
        print("[!] Warning: No recognized platform data found")
    elif not has_nvidia:
        print("  Note: Generating AMD-only comparison (no NVIDIA data)")
    elif not has_amd:
        print("  Note: Generating NVIDIA-only comparison (no AMD data)")
    
    if framework_filter:
        print(f"\n  * Using framework filter: {framework_filter}")
    else:
        print("\n  * No framework filter selected; using first matching per model/platform")
    
    print(f"\nGenerating comparison plot: {args.output}")
    try:
        create_comparison_plot(benchmarks, args.output, framework_filter=framework_filter)
    except Exception as e:
        print(f"[!] Could not generate plot: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*100)
    print("PERFORMANCE SUMMARY")
    print("="*100)
    print(f"\n{'Configuration':<25} {'Tokens/sec/GPU':>15} {'Avg Step Time':>15} {'Avg Loss':>12}")
    print("-" * 100)
    
    def find_key_summary(platform: str, model_name: str) -> str | None:
        """Find benchmark key for given platform and model."""
        if framework_filter:
            candidate = f"{platform}-{framework_filter}-{model_name}"
            return candidate if candidate in benchmarks else None
        
        matches = [
            key for key in benchmarks.keys()
            if key.startswith(f"{platform}-") and key.endswith(f"-{model_name}")
        ]
        
        if not matches:
            return None
        
        preferred = {"nvidia": "nemo", "amd": "prim"}
        for match in matches:
            if preferred.get(platform, "") in match:
                return match
        
        return sorted(matches)[0]
    
    for platform in ["nvidia", "amd"]:
        for model_name in ["llama", "qwen"]:
            key = find_key_summary(platform, model_name)
            if not key:
                continue
            data = benchmarks[key]
            perf = data['performance_metrics']
            tps_gpu = perf.get('tokens_per_second_per_gpu', 0)
            step_time = perf.get('avg_step_time_seconds', 0)
            loss_values = data.get('loss_values', [])
            avg_loss = sum(loss_values) / len(loss_values) if loss_values else 0
            
            print(f"{key:<25} {tps_gpu:>15,.1f} {step_time:>15.2f} {avg_loss:>12.2f}")
    
    if has_nvidia and has_amd:
        print("\n" + "-"*100)
        print("SPEEDUP COMPARISON (AMD vs NVIDIA)")
        print("-"*100)
        
        for model_name in ["llama", "qwen"]:
            nvidia_key = find_key_summary("nvidia", model_name)
            amd_key = find_key_summary("amd", model_name)
            
            if nvidia_key and amd_key:
                nvidia_tps = benchmarks[nvidia_key]['performance_metrics'].get('tokens_per_second_per_gpu', 0)
                amd_tps = benchmarks[amd_key]['performance_metrics'].get('tokens_per_second_per_gpu', 0)
                nvidia_step = benchmarks[nvidia_key]['performance_metrics'].get('avg_step_time_seconds', 0)
                amd_step = benchmarks[amd_key]['performance_metrics'].get('avg_step_time_seconds', 0)
                
                if nvidia_tps > 0 and amd_tps > 0:
                    tps_ratio = amd_tps / nvidia_tps
                    step_ratio = nvidia_step / amd_step if amd_step > 0 else 0
                    
                    winner = "AMD" if tps_ratio > 1 else "NVIDIA"
                    ratio = tps_ratio if tps_ratio > 1 else 1/tps_ratio
                    
                    print(f"\n  {model_name.upper()}:")
                    print(f"    Throughput: AMD {amd_tps:,.0f} vs NVIDIA {nvidia_tps:,.0f} tokens/s/GPU")
                    print(f"    Step Time:  AMD {amd_step:.3f}s vs NVIDIA {nvidia_step:.3f}s")
                    print(f"    -> {winner} is {ratio:.2f}x faster")
    
    print("\n" + "="*100 + "\n")
    
    return 0


if __name__ == "__main__":
    exit(main())
