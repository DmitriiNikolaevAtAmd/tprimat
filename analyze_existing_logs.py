#!/usr/bin/env python3
"""
Analyze existing profiling logs from AMD and NVIDIA runs.
This helps compare with the new benchmark results.
"""
import os
import json
from pathlib import Path
from datetime import datetime


def analyze_tensorboard_events(log_dir: str):
    """
    Analyze TensorBoard event files from NeMo training.
    
    Args:
        log_dir: Directory containing TensorBoard event files
    """
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except ImportError:
        print("[!] TensorBoard not installed. Install with: pip install tensorboard")
        return None
    
    log_path = Path(log_dir)
    event_files = list(log_path.rglob("events.out.tfevents.*"))
    
    if not event_files:
        print(f"[!] No TensorBoard event files found in {log_dir}")
        return None
    
    print(f"[#] Analyzing {len(event_files)} event file(s)...")
    
    results = {}
    for event_file in event_files:
        print(f"  - {event_file.name}")
        
        ea = event_accumulator.EventAccumulator(str(event_file))
        ea.Reload()
        
        # Get available tags
        scalars = ea.Tags().get('scalars', [])
        
        # Extract timing metrics
        timing_metrics = [tag for tag in scalars if 'time' in tag.lower() or 'step' in tag.lower()]
        
        for tag in timing_metrics:
            try:
                events = ea.Scalars(tag)
                values = [e.value for e in events]
                if values:
                    results[tag] = {
                        'mean': sum(values) / len(values),
                        'min': min(values),
                        'max': max(values),
                        'count': len(values)
                    }
            except Exception as e:
                print(f"    [!] Error reading {tag}: {e}")
    
    return results


def analyze_amd_profiling_logs(amd_logs_dir: str = "../amd-logs"):
    """
    Analyze AMD profiling logs (Excel files).
    """
    amd_path = Path(amd_logs_dir)
    
    if not amd_path.exists():
        print(f"[!] AMD logs directory not found: {amd_logs_dir}")
        return
    
    print(f"\n{'='*60}")
    print("AMD PROFILING LOGS ANALYSIS")
    print(f"{'='*60}\n")
    
    # Find all model directories
    for model_dir in amd_path.iterdir():
        if not model_dir.is_dir():
            continue
        
        print(f"[:] {model_dir.name.upper()}")
        print("-" * 60)
        
        excel_files = list(model_dir.glob("*.xlsx"))
        
        if not excel_files:
            print("  No profiling data found")
            continue
        
        print(f"  Found {len(excel_files)} rank profiling reports")
        print(f"  Files:")
        
        for excel_file in sorted(excel_files):
            # Extract rank number from filename
            if "rank[" in excel_file.name:
                rank = excel_file.name.split("rank[")[1].split("]")[0]
                print(f"    - Rank {rank}: {excel_file.name}")
        
        print()
    
    print("[i] Note: AMD profiling reports are in Excel format (.xlsx)")
    print("   You can open them in Excel/LibreOffice to view detailed metrics")
    print()


def analyze_nvidia_logs(nvi_logs_dir: str = "../nvi-logs"):
    """
    Analyze NVIDIA TensorBoard logs.
    """
    nvi_path = Path(nvi_logs_dir)
    
    if not nvi_path.exists():
        print(f"[!] NVIDIA logs directory not found: {nvi_logs_dir}")
        return
    
    print(f"\n{'='*60}")
    print("NVIDIA TENSORBOARD LOGS ANALYSIS")
    print(f"{'='*60}\n")
    
    # Find all experiment directories
    for exp_dir in nvi_path.iterdir():
        if not exp_dir.is_dir():
            continue
        
        print(f"[:] {exp_dir.name.upper()}")
        print("-" * 60)
        
        # Analyze TensorBoard events
        results = analyze_tensorboard_events(str(exp_dir))
        
        if results:
            print("\n  Metrics found:")
            for metric, stats in results.items():
                print(f"    {metric}:")
                print(f"      Mean: {stats['mean']:.4f}")
                print(f"      Min:  {stats['min']:.4f}")
                print(f"      Max:  {stats['max']:.4f}")
                print(f"      Samples: {stats['count']}")
        
        # Check for hparams
        hparams_file = exp_dir / "hparams.yaml"
        if hparams_file.exists():
            print(f"\n  [OK] Hyperparameters: {hparams_file}")
        
        print()


def compare_with_new_benchmarks(benchmark_dir: str = "./output"):
    """
    Compare old profiling logs with new benchmark results.
    """
    benchmark_path = Path(benchmark_dir)
    
    if not benchmark_path.exists() or not list(benchmark_path.glob("*.json")):
        print("[i] No new benchmark results found yet.")
        print("   Run your training scripts to generate new benchmarks.")
        return
    
    print(f"\n{'='*60}")
    print("NEW BENCHMARK RESULTS")
    print(f"{'='*60}\n")
    
    json_files = sorted(benchmark_path.glob("benchmark_*.json"), 
                       key=os.path.getmtime, reverse=True)
    
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        platform = data['platform'].upper()
        device = data['gpu_info']['device_name']
        timestamp = data['timestamp']
        avg_time = data['performance_metrics']['avg_step_time_seconds']
        throughput = data['performance_metrics']['throughput_steps_per_second']
        
        print(f"[#] {platform} - {device}")
        print(f"   Timestamp: {timestamp}")
        print(f"   Avg Step Time: {avg_time:.4f}s")
        print(f"   Throughput: {throughput:.3f} steps/s")
        
        if 'memory_metrics' in data:
            peak_mem = data['memory_metrics']['peak_memory_allocated_gb']
            print(f"   Peak Memory: {peak_mem:.2f} GB")
        
        print()


def main():
    print("="*60)
    print("PROFILING LOGS ANALYZER")
    print("="*60)
    
    # Analyze existing AMD logs
    analyze_amd_profiling_logs()
    
    # Analyze existing NVIDIA logs
    analyze_nvidia_logs()
    
    # Show new benchmark results
    compare_with_new_benchmarks()
    
    print("="*60)
    print("SUMMARY")
    print("="*60)
    print()
    print("[:] Existing Logs:")
    print("   - AMD:    ../amd-logs/    (Excel profiling reports)")
    print("   - NVIDIA: ../nvi-logs/    (TensorBoard event files)")
    print()
    print("[+] New Benchmarks:")
    print("   - Location: ./output/")
    print("   - Run training scripts to generate new benchmarks")
    print("   - Use compare_results.py to create visual comparisons")
    print()
    print("[i] The new benchmark system provides:")
    print("   [OK] Unified metrics across AMD and NVIDIA")
    print("   [OK] Automated comparison reports")
    print("   [OK] Visual comparison charts")
    print("   [OK] Statistical analysis")
    print()


if __name__ == "__main__":
    main()

