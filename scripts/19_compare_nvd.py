#!/usr/bin/env python3
"""
NVIDIA GPU Benchmark Comparison Script

Compares different frameworks (Megatron, Transformers, DeepSpeed, NeMo) on NVIDIA GPUs with:
- Performance metrics (throughput, step time, memory)
- Visual plots and analysis

Usage:
    python3 nvd_compare.py [--results-dir ./]
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np


def load_nvidia_benchmarks(results_dir: str) -> Dict[str, Dict]:
    """Load all NVIDIA benchmark results with framework and model information."""
    results_path = Path(results_dir)
    
    benchmarks = {}
    
    for json_file in sorted(results_path.glob("train_*.json")):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            platform = data.get('platform', '')
            if platform != 'nvd':
                continue
            
            filename = json_file.stem
            parts = filename.split('_')
            
            framework = None
            model_name = None
            
            if len(parts) >= 4 and parts[1] == 'nvd':
                framework = parts[2]
                model_name = parts[3]
            elif len(parts) >= 3:
                framework = parts[1]
                model_name = parts[2]
            
            if framework and model_name:
                framework_display = {
                    'mega': 'Megatron',
                    'tran': 'Transformers',
                    'deep': 'DeepSpeed',
                    'nemo': 'NeMo',
                    'fsdp': 'FSDP',
                    'nvd': 'NVIDIA'
                }.get(framework, framework.upper())
                
                key = f"{framework}-{model_name}"
                data['framework'] = framework
                data['framework_display'] = framework_display
                data['model_name'] = model_name
                benchmarks[key] = data
                
                print(f"[+] Loaded: {framework_display} {model_name.upper()} from {json_file.name}")
        except Exception as e:
            print(f"[!] Error loading {json_file}: {e}")
    
    return benchmarks


def create_nvidia_comparison_plot(benchmarks: Dict[str, Dict], output_file: str = "nvd_compare.png"):
    """Create visual comparison of all NVIDIA framework-model combinations."""
    
    if not benchmarks:
        print("[!] No NVIDIA benchmark data to plot")
        return None
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), facecolor='white')
    fig.suptitle('NVIDIA H100 - Framework Comparison', fontsize=20, fontweight='bold', y=0.995, color='#1E4D2B')
    axes = axes.flatten()
    
    style_map = {
        'mega-llama': {'color': '#2E86AB', 'marker': 'o', 'label': 'Megatron Llama', 'linestyle': '-'},
        'mega-qwen':  {'color': '#5FA8D3', 'marker': 's', 'label': 'Megatron Qwen', 'linestyle': '--'},
        'tran-llama': {'color': '#06A77D', 'marker': 'o', 'label': 'Transformers Llama', 'linestyle': '-'},
        'tran-qwen':  {'color': '#4DB896', 'marker': 's', 'label': 'Transformers Qwen', 'linestyle': '--'},
        'deep-llama': {'color': '#D97B2C', 'marker': 'o', 'label': 'DeepSpeed Llama', 'linestyle': '-'},
        'deep-qwen':  {'color': '#F4A261', 'marker': 's', 'label': 'DeepSpeed Qwen', 'linestyle': '--'},
        'nemo-llama': {'color': '#9B59B6', 'marker': 'o', 'label': 'NeMo Llama', 'linestyle': '-'},
        'nemo-qwen':  {'color': '#BB8FCE', 'marker': 's', 'label': 'NeMo Qwen', 'linestyle': '--'},
        'fsdp-llama': {'color': '#C41E3A', 'marker': 'o', 'label': 'FSDP Llama', 'linestyle': '-'},
        'fsdp-qwen':  {'color': '#E74C3C', 'marker': 's', 'label': 'FSDP Qwen', 'linestyle': '--'},
    }
    
    first_data = next(iter(benchmarks.values()))
    config = first_data.get('training_config', {})
    
    ax1 = axes[0]
    labels = []
    values = []
    colors_list = []
    ordered_keys = sorted(benchmarks.keys())
    
    for key in ordered_keys:
        if key in benchmarks and key in style_map:
            perf = benchmarks[key]['performance_metrics']
            tps_gpu = perf.get('tokens_per_second_per_gpu')
            if tps_gpu:
                labels.append(style_map[key]['label'])
                values.append(tps_gpu)
                colors_list.append(style_map[key]['color'])
    
    if values:
        x_pos = np.arange(len(labels))
        bars = ax1.bar(x_pos, values, color=colors_list, alpha=0.8, edgecolor='#333333', linewidth=1.5)
        ax1.set_ylabel('Tokens/s/GPU', fontweight='bold', fontsize=12)
        ax1.set_title('Average Per-GPU Throughput', fontweight='bold', fontsize=14, pad=10)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        ax1.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
        
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:,.0f}',
                    ha='center', va='bottom', fontweight='bold', fontsize=9)
    else:
        ax1.text(0.5, 0.5, 'Throughput data not available', 
                ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('Average Per-GPU Throughput', fontweight='bold', fontsize=14)
    
    ax2 = axes[1]
    has_data = False
    
    for key in ordered_keys:
        if key in benchmarks and 'loss_values' in benchmarks[key] and key in style_map:
            loss_values = benchmarks[key]['loss_values']
            if loss_values:
                steps = range(len(loss_values))
                style = style_map[key]
                ax2.plot(steps, loss_values, 
                        marker=style['marker'], 
                        linestyle=style['linestyle'],
                        color=style['color'], 
                        label=style['label'], 
                        linewidth=2, 
                        markersize=2, 
                        markevery=max(1, len(loss_values)//50),
                        alpha=0.85)
                has_data = True
    
    if has_data:
        ax2.set_xlabel('Step', fontweight='bold', fontsize=11)
        ax2.set_ylabel('Loss', fontweight='bold', fontsize=12)
        ax2.set_title('Training Loss over Time', fontweight='bold', fontsize=14, pad=10)
        ax2.legend(fontsize=9, loc='best', framealpha=0.9)
        ax2.grid(alpha=0.3, linestyle='--', linewidth=0.5)
    else:
        ax2.text(0.5, 0.5, 'Loss data not available', 
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Training Loss over Time', fontweight='bold', fontsize=14)
    
    ax3 = axes[2]
    has_data = False
    
    for key in ordered_keys:
        if key in benchmarks and 'step_times' in benchmarks[key] and key in style_map:
            step_times = benchmarks[key]['step_times']
            if step_times:
                steps = range(len(step_times))
                style = style_map[key]
                ax3.plot(steps, step_times, 
                        marker=style['marker'], 
                        linestyle=style['linestyle'],
                        color=style['color'], 
                        label=style['label'], 
                        linewidth=2, 
                        markersize=2, 
                        markevery=max(1, len(step_times)//50),
                        alpha=0.85)
                
                avg_time = sum(step_times[1:]) / (len(step_times) - 1) if len(step_times) > 1 else sum(step_times) / len(step_times)
                ax3.axhline(y=avg_time, color=style['color'], linestyle=':', alpha=0.3, linewidth=1.2)
                has_data = True
    
    if has_data:
        ax3.set_xlabel('Step', fontweight='bold', fontsize=11)
        ax3.set_ylabel('Time (seconds)', fontweight='bold', fontsize=12)
        ax3.set_title('Step Duration over Time', fontweight='bold', fontsize=14, pad=10)
        ax3.legend(fontsize=9, loc='best', framealpha=0.9)
        ax3.grid(alpha=0.3, linestyle='--', linewidth=0.5)
    else:
        ax3.text(0.5, 0.5, 'Step time data not available', 
                ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Step Duration over Time', fontweight='bold', fontsize=14)
    
    ax4 = axes[3]
    has_data = False
    
    for key in ordered_keys:
        if key in benchmarks and 'learning_rates' in benchmarks[key] and key in style_map:
            learning_rates = benchmarks[key]['learning_rates']
            if learning_rates:
                steps = range(len(learning_rates))
                style = style_map[key]
                ax4.plot(steps, learning_rates, 
                        marker=style['marker'], 
                        linestyle=style['linestyle'],
                        color=style['color'], 
                        label=style['label'], 
                        linewidth=2, 
                        markersize=2, 
                        markevery=max(1, len(learning_rates)//50),
                        alpha=0.85)
                has_data = True
    
    if has_data:
        ax4.set_xlabel('Step', fontweight='bold', fontsize=11)
        ax4.set_ylabel('Learning Rate', fontweight='bold', fontsize=12)
        ax4.set_title('Learning Rate over Time', fontweight='bold', fontsize=14, pad=10)
        ax4.legend(fontsize=9, loc='best', framealpha=0.9)
        ax4.grid(alpha=0.3, linestyle='--', linewidth=0.5)
        ax4.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    else:
        ax4.text(0.5, 0.5, 'Learning rate data not available', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Learning Rate over Time', fontweight='bold', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  + Comparison plot saved to: {output_file}")
    
    return fig


def print_nvidia_comparison(benchmarks: Dict[str, Dict]):
    """Print comprehensive comparison of NVIDIA benchmark metrics."""
    
    print("\n" + "="*100)
    print("NVIDIA H100 - FRAMEWORK PERFORMANCE COMPARISON")
    print("="*100)
    
    first_data = next(iter(benchmarks.values()))
    gpu_info = first_data['gpu_info']
    config = first_data.get('training_config', {})
    
    print(f"\n  Hardware: {gpu_info.get('device_name', 'Unknown')}")
    print(f"  GPUs: {config.get('num_gpus', 'N/A')}")
    print(f"  PyTorch: {gpu_info.get('pytorch_version', 'N/A')}")
    print(f"  CUDA: {gpu_info.get('software_version', 'N/A')}")
    
    llama_benchmarks = {k: v for k, v in benchmarks.items() if 'llama' in k}
    qwen_benchmarks = {k: v for k, v in benchmarks.items() if 'qwen' in k}
    
    if llama_benchmarks:
        print("\n" + "-"*100)
        print("LLAMA 3.1 8B COMPARISON")
        print("-"*100)
        print(f"\n{'Framework':<20} {'Tokens/s/GPU':>15} {'Avg Step Time':>15} {'Final Loss':>12} {'Total Time':>12}")
        print("-" * 100)
        
        for key in sorted(llama_benchmarks.keys()):
            data = llama_benchmarks[key]
            framework = data['framework_display']
            perf = data['performance_metrics']
            tps_gpu = perf.get('tokens_per_second_per_gpu', 0)
            step_time = perf.get('avg_step_time_seconds', 0)
            total_time = perf.get('total_time_seconds', 0)
            loss_values = data.get('loss_values', [])
            final_loss = loss_values[-1] if loss_values else 0
            
            print(f"{framework:<20} {tps_gpu:>15,.1f} {step_time:>15.3f}s {final_loss:>12.4f} {total_time:>12.1f}s")
    
    if qwen_benchmarks:
        print("\n" + "-"*100)
        print("QWEN 2.5 7B COMPARISON")
        print("-"*100)
        print(f"\n{'Framework':<20} {'Tokens/s/GPU':>15} {'Avg Step Time':>15} {'Final Loss':>12} {'Total Time':>12}")
        print("-" * 100)
        
        for key in sorted(qwen_benchmarks.keys()):
            data = qwen_benchmarks[key]
            framework = data['framework_display']
            perf = data['performance_metrics']
            tps_gpu = perf.get('tokens_per_second_per_gpu', 0)
            step_time = perf.get('avg_step_time_seconds', 0)
            total_time = perf.get('total_time_seconds', 0)
            loss_values = data.get('loss_values', [])
            final_loss = loss_values[-1] if loss_values else 0
            
            print(f"{framework:<20} {tps_gpu:>15,.1f} {step_time:>15.3f}s {final_loss:>12.4f} {total_time:>12.1f}s")
    
    print("\n" + "="*100)
    print("PERFORMANCE SUMMARY")
    print("="*100)
    
    best_throughput = max(benchmarks.items(), 
                         key=lambda x: x[1]['performance_metrics'].get('tokens_per_second_per_gpu', 0))
    best_tps = best_throughput[1]['performance_metrics']['tokens_per_second_per_gpu']
    best_framework = best_throughput[1]['framework_display']
    best_model = best_throughput[1]['model_name'].upper()
    
    print(f"\n  Highest Throughput: {best_framework} {best_model}")
    print(f"    {best_tps:,.1f} tokens/s/GPU")
    
    best_step_time = min(benchmarks.items(), 
                        key=lambda x: x[1]['performance_metrics'].get('avg_step_time_seconds', float('inf')))
    best_step = best_step_time[1]['performance_metrics']['avg_step_time_seconds']
    best_step_framework = best_step_time[1]['framework_display']
    best_step_model = best_step_time[1]['model_name'].upper()
    
    print(f"\n  Fastest Step Time: {best_step_framework} {best_step_model}")
    print(f"    {best_step:.3f} seconds/step")
    
    print(f"\n  Throughput Range:")
    min_tps = min(v['performance_metrics'].get('tokens_per_second_per_gpu', float('inf')) 
                  for v in benchmarks.values())
    print(f"    Best: {best_tps:,.1f} tokens/s/GPU")
    print(f"    Worst: {min_tps:,.1f} tokens/s/GPU")
    print(f"    Difference: {((best_tps - min_tps) / min_tps * 100):.1f}% faster")
    
    print("\n" + "="*100 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='NVIDIA GPU benchmark comparison - all frameworks and models'
    )
    default_dir = os.environ.get('OUTPUT_DIR', './output')
    parser.add_argument('--results-dir', default=default_dir,
                       help='Directory containing benchmark JSON files (default: ./output or OUTPUT_DIR env var)')
    parser.add_argument('--output', default='nvd_compare.png',
                       help='Output filename for the comparison plot (default: nvd_compare.png)')
    
    args = parser.parse_args()
    
    print("="*100)
    print("NVIDIA GPU BENCHMARK ANALYSIS")
    print("="*100)
    print(f"\nLoading benchmark results from: {args.results_dir}")
    
    benchmarks = load_nvidia_benchmarks(args.results_dir)
    
    if not benchmarks:
        print("\n  x No NVIDIA benchmark results found!")
        print(f"\nExpected files in {args.results_dir}/:")
        print("  - train_nvd_mega_llama.json, train_nvd_mega_qwen.json (Megatron)")
        print("  - train_nvd_tran_llama.json, train_nvd_tran_qwen.json (Transformers)")
        print("  - train_nvd_deep_llama.json, train_nvd_deep_qwen.json (DeepSpeed)")
        print("  - train_nvd_fsdp_llama.json, train_nvd_fsdp_qwen.json (FSDP)")
        print("  - train_nvd_nemo_llama.json, train_nvd_nemo_qwen.json (NeMo)")
        print("  Note: NeMo scripts now auto-detect platform and output with appropriate prefix")
        print("\n  Note: Only files with platform='nvd' will be included")
        return 1
    
    print(f"\n  Found {len(benchmarks)} NVIDIA benchmark(s)")
    
    output_path = os.path.join(args.results_dir, args.output)
    print(f"\nGenerating comparison plot: {output_path}")
    try:
        create_nvidia_comparison_plot(benchmarks, output_path)
    except Exception as e:
        print(f"[!] Could not generate plot: {e}")
        import traceback
        traceback.print_exc()
    
    print_nvidia_comparison(benchmarks)
    
    return 0


if __name__ == "__main__":
    exit(main())
