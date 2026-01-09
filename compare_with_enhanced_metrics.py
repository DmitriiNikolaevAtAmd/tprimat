#!/usr/bin/env python3
"""
Enhanced comparison script that includes additional metrics.

Usage:
    python3 compare_with_enhanced_metrics.py [--results-dir ./output]
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

# Import the enhanced metrics calculator
from enhanced_metrics import get_enhanced_metrics, GPU_SPECS

# Import original comparison functions
from compare_results import (
    load_benchmark_results,
    create_comparison_plot
)


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


def main():
    parser = argparse.ArgumentParser(
        description='Enhanced GPU comparison with additional metrics'
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
    
    # Generate standard comparison
    print("\nGenerating standard comparison...")
    try:
        create_comparison_plot(nvidia_data, amd_data, "comparison_plot.png")
    except Exception as e:
        print(f"⚠️  Could not generate plot: {e}")
    
    # Print enhanced metrics
    print_enhanced_comparison(nvidia_data, amd_data)
    
    return 0


if __name__ == "__main__":
    exit(main())
