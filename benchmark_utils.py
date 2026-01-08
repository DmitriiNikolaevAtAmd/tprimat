"""
Unified benchmarking utilities for AMD vs NVIDIA GPU comparison.
Works on both ROCm and CUDA platforms.
"""
import time
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import torch
from lightning.pytorch.callbacks import Callback


def round_floats(obj: Any, precision: int = 3) -> Any:
    """
    Recursively round all float values in a nested structure to specified precision.
    
    Args:
        obj: Dictionary, list, or value to process
        precision: Number of decimal places (default: 3)
    
    Returns:
        Object with all floats rounded
    """
    if isinstance(obj, float):
        return round(obj, precision)
    elif isinstance(obj, dict):
        return {key: round_floats(value, precision) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [round_floats(item, precision) for item in obj]
    else:
        return obj


class BenchmarkCallback(Callback):
    """Callback to collect platform-agnostic performance metrics."""
    
    def __init__(self, output_dir: str = "./output", platform: str = "auto", model_name: str = None):
        """
        Args:
            output_dir: Directory to save benchmark results
            platform: 'cuda', 'rocm', or 'auto' for auto-detection
            model_name: Name of the model (e.g., 'llama', 'mistral', 'qwen')
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Detect platform
        if platform == "auto":
            if torch.cuda.is_available():
                # Check if ROCm (has torch.version.hip) or CUDA
                is_rocm = hasattr(torch.version, 'hip') and torch.version.hip is not None
                self.platform = "amd" if is_rocm else "nvd"
            else:
                self.platform = "cpu"
        else:
            self.platform = platform
        
        # Metrics storage
        self.step_times = []
        self.memory_allocated = []
        self.memory_reserved = []
        self.step_start_time = None
        self.train_start_time = None
        self.gpu_info = {}
        self.global_batch_size = None
        self.sequence_length = None
        self.num_gpus = None
        self.model_name = model_name
    
    def _get_gpu_core_count(self, device_name: str, device_props) -> int:
        """
        Get approximate GPU core count based on device name.
        
        NVIDIA GPUs use CUDA cores, AMD GPUs use Stream Processors.
        Note: These are approximate values for common models.
        """
        device_name_lower = device_name.lower()
        
        # NVIDIA GPUs (CUDA cores)
        nvidia_cores = {
            # H100 series
            "h100": 16896,  # H100 SXM5 (80GB/94GB)
            "h100 sxm5": 16896,
            "h100 pcie": 14592,
            
            # A100 series
            "a100": 6912,   # A100 (40GB/80GB)
            "a100-sxm4": 6912,
            "a100-pcie": 6912,
            
            # V100 series
            "v100": 5120,
            "v100-sxm2": 5120,
            "v100-pcie": 5120,
            
            # A40/A30 series
            "a40": 10752,
            "a30": 10752,
            "a10": 9216,
            
            # Consumer/Workstation
            "rtx 4090": 16384,
            "rtx 3090": 10496,
            "rtx 3080": 8704,
        }
        
        # AMD GPUs (Stream Processors per GCD)
        amd_cores = {
            # MI300 series
            "mi300x": 19456,  # Per GCD
            "mi300a": 19456,
            
            # MI250 series (has 2 GCDs)
            "mi250x": 14080 * 2,  # 2 GCDs with 14,080 SPs each = 28,160 total
            "mi250": 13312 * 2,   # 2 GCDs with 13,312 SPs each = 26,624 total
            
            # MI210 series
            "mi210": 13312,
            
            # MI100 series
            "mi100": 7680,
            
            # Radeon Instinct
            "instinct mi250x": 14080 * 2,
            "instinct mi250": 13312 * 2,
            "instinct mi210": 13312,
            "instinct mi100": 7680,
        }
        
        # Try to match device name
        for gpu_name, cores in nvidia_cores.items():
            if gpu_name in device_name_lower:
                return cores
        
        for gpu_name, cores in amd_cores.items():
            if gpu_name in device_name_lower:
                return cores
        
        # Try to get from device properties (may not always be accurate)
        if hasattr(device_props, 'multi_processor_count'):
            # NVIDIA: multiply by cores per SM (assuming 128 for modern GPUs)
            return device_props.multi_processor_count * 128
        
        # Return 0 if unknown (will be handled in metrics calculation)
        return 0
        
    def on_train_start(self, trainer, pl_module):
        """Collect GPU information at training start."""
        self.train_start_time = time.time()
        
        # Get training configuration
        self.num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        
        # Try to get batch size and sequence length from datamodule
        if hasattr(trainer, 'datamodule'):
            self.global_batch_size = getattr(trainer.datamodule, 'global_batch_size', None)
            self.sequence_length = getattr(trainer.datamodule, 'seq_length', 
                                          getattr(trainer.datamodule, 'sequence_length', 2048))  # Default 2048
        
        if torch.cuda.is_available():
            device_props = torch.cuda.get_device_properties(0)
            device_name = torch.cuda.get_device_name(0)
            
            # Get GPU core count (approximate based on known models)
            gpu_cores = self._get_gpu_core_count(device_name, device_props)
            
            # Detect software stack and version
            is_rocm = hasattr(torch.version, 'hip') and torch.version.hip is not None
            software_stack = "rocm" if is_rocm else "cuda"
            software_version = torch.version.hip if is_rocm else torch.version.cuda
            
            self.gpu_info = {
                "device_count": self.num_gpus,
                "device_name": device_name,
                "total_memory_gb": device_props.total_memory / 1e9,
                "gpu_cores": gpu_cores,
                "pytorch_version": torch.__version__,
                "software_stack": software_stack,
                "software_version": software_version,
            }
        
        # Only print on rank 0 to avoid duplicate output in distributed training
        if trainer.is_global_zero:
            software_stack = self.gpu_info.get("software_stack", "unknown")
            print(f"\n{'='*60}")
            print(f"BENCHMARK START - Platform: {self.platform.upper()} ({software_stack.upper()})")
            print(f"{'='*60}")
            for key, value in self.gpu_info.items():
                print(f"{key}: {value}")
            print(f"{'='*60}\n")
    
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        """Mark start of training step."""
        self.step_start_time = time.time()
        
        # Clear cache for consistent measurements
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Collect metrics after each training step."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        step_time = time.time() - self.step_start_time
        self.step_times.append(step_time)
        
        # Collect memory stats
        if torch.cuda.is_available():
            mem_allocated = torch.cuda.memory_allocated() / 1e9  # GB
            mem_reserved = torch.cuda.memory_reserved() / 1e9    # GB
            self.memory_allocated.append(mem_allocated)
            self.memory_reserved.append(mem_reserved)
        
        # Log every 10 steps (only on rank 0)
        if trainer.is_global_zero and batch_idx > 0 and batch_idx % 10 == 0:
            recent_times = self.step_times[-10:]
            avg_time = sum(recent_times) / len(recent_times)
            
            if torch.cuda.is_available():
                avg_mem = sum(self.memory_allocated[-10:]) / len(self.memory_allocated[-10:])
                print(f"[{self.platform.upper()}] Step {batch_idx:3d} | "
                      f"Time: {step_time:.3f}s | Avg: {avg_time:.3f}s | "
                      f"Memory: {avg_mem:.2f}GB")
            else:
                print(f"[{self.platform.upper()}] Step {batch_idx:3d} | "
                      f"Time: {step_time:.3f}s | Avg: {avg_time:.3f}s")
    
    def on_train_end(self, trainer, pl_module):
        """Save benchmark results."""
        # Only save results on rank 0 to avoid duplicate files in distributed training
        if not trainer.is_global_zero:
            return
            
        total_time = time.time() - self.train_start_time
        
        # Calculate statistics
        if len(self.step_times) > 1:
            # Skip first step (warmup)
            step_times_no_warmup = self.step_times[1:]
            
            avg_step_time = sum(step_times_no_warmup) / len(step_times_no_warmup)
            steps_per_second = len(step_times_no_warmup) / sum(step_times_no_warmup)
            
            # Calculate token-based throughput
            # Throughput = tokens processed per second (total system throughput)
            # Tokens/sec/GPU = throughput per GPU (efficiency metric independent of cluster size)
            tokens_per_second = None
            tokens_per_second_per_gpu = None
            
            if self.global_batch_size and self.sequence_length:
                # Tokens per step = global_batch_size × sequence_length
                tokens_per_step = self.global_batch_size * self.sequence_length
                # Total system throughput in tokens/sec
                tokens_per_second = tokens_per_step / avg_step_time
                # Per-GPU throughput (efficiency metric)
                tokens_per_second_per_gpu = tokens_per_second / self.num_gpus if self.num_gpus else None
            
            results = {
                "platform": self.platform,
                "gpu_info": self.gpu_info,
                "timestamp": datetime.now().isoformat(),
                "training_config": {
                    "max_steps": trainer.max_steps,
                    "global_batch_size": self.global_batch_size or 'N/A',
                    "micro_batch_size": getattr(trainer.datamodule, 'micro_batch_size', 'N/A'),
                    "sequence_length": self.sequence_length or 'N/A',
                    "num_gpus": self.num_gpus,
                },
                "performance_metrics": {
                    "total_steps": len(self.step_times),
                    "total_time_seconds": total_time,
                    "avg_step_time_seconds": avg_step_time,
                    "min_step_time_seconds": min(step_times_no_warmup),
                    "max_step_time_seconds": max(step_times_no_warmup),
                    "steps_per_second": steps_per_second,
                    # Primary throughput metrics (token-based)
                    "tokens_per_second": tokens_per_second,  # Total system throughput
                    "tokens_per_second_per_gpu": tokens_per_second_per_gpu,  # Per-GPU efficiency
                    # Secondary metrics (for reference)
                    "throughput_per_gpu_core": steps_per_second / self.gpu_info["gpu_cores"] if self.gpu_info.get("gpu_cores", 0) > 0 else 0,
                },
                "raw_step_times": self.step_times,
            }
            
            if self.memory_allocated:
                mem_no_warmup = self.memory_allocated[1:]
                results["memory_metrics"] = {
                    "avg_memory_allocated_gb": sum(mem_no_warmup) / len(mem_no_warmup),
                    "peak_memory_allocated_gb": max(mem_no_warmup),
                    "avg_memory_reserved_gb": sum(self.memory_reserved[1:]) / len(self.memory_reserved[1:]),
                    "peak_memory_reserved_gb": max(self.memory_reserved[1:]),
                }
            
            # Save results (round all floats to 3 decimal places)
            # Use software stack (cuda/rocm) for filename
            software_stack = self.gpu_info.get("software_stack", self.platform)
            
            # Create filename with model name if provided
            if self.model_name:
                filename = f"benchmark_{software_stack}_{self.model_name}.json"
            else:
                # Fallback to timestamp if no model name
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"benchmark_{software_stack}_{timestamp}.json"
            
            filepath = self.output_dir / filename
            
            # Round all float values to 3 decimal places
            results_rounded = round_floats(results, precision=3)
            
            with open(filepath, 'w') as f:
                json.dump(results_rounded, f, indent=2)
            
            software_stack = self.gpu_info.get("software_stack", self.platform)
            print(f"\n{'='*60}")
            print(f"BENCHMARK COMPLETE - Platform: {self.platform.upper()} ({software_stack.upper()})")
            print(f"{'='*60}")
            print(f"GPUs: {self.num_gpus}")
            print(f"Total Steps: {results['performance_metrics']['total_steps']}")
            print(f"Total Time: {total_time:.2f}s")
            print(f"Avg Step Time: {results['performance_metrics']['avg_step_time_seconds']:.3f}s")
            
            # Primary throughput metrics (token-based)
            if results['performance_metrics']['tokens_per_second']:
                print(f"\nThroughput Metrics:")
                print(f"  Total Throughput: {results['performance_metrics']['tokens_per_second']:,.0f} tokens/sec")
                print(f"  Per-GPU Throughput: {results['performance_metrics']['tokens_per_second_per_gpu']:,.0f} tokens/sec/GPU")
                print(f"  (Global batch size: {self.global_batch_size}, Sequence length: {self.sequence_length})")
            else:
                print(f"Throughput: {results['performance_metrics']['steps_per_second']:.3f} steps/s")
                print(f"  (Token-based metrics unavailable - need batch size & sequence length)")
            
            if 'memory_metrics' in results:
                print(f"\nMemory Usage:")
                print(f"  Avg Memory: {results['memory_metrics']['avg_memory_allocated_gb']:.2f}GB")
                print(f"  Peak Memory: {results['memory_metrics']['peak_memory_allocated_gb']:.2f}GB")
            
            print(f"\nResults saved to: {filepath}")
            print(f"{'='*60}\n")


def compare_benchmarks(results_dir: str = "./output") -> Dict:
    """
    Compare benchmark results from AMD and NVIDIA runs.
    
    Args:
        results_dir: Directory containing benchmark JSON files
        
    Returns:
        Dictionary with comparison results
    """
    results_path = Path(results_dir)
    
    # Find latest results for each platform
    nvidia_results = []
    amd_results = []
    
    for json_file in results_path.glob("*.json"):
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Support both old and new platform naming
        platform = data.get('platform', '').lower()
        software_stack = data.get('gpu_info', {}).get('software_stack', '').lower()
        
        # NVIDIA: cuda, nvd, nvidia
        if platform in ['cuda', 'nvd', 'nvidia'] or software_stack == 'cuda':
            nvidia_results.append(data)
        # AMD: rocm, amd
        elif platform in ['rocm', 'amd'] or software_stack == 'rocm':
            amd_results.append(data)
    
    if not nvidia_results or not amd_results:
        print("⚠️  Need results from both NVIDIA and AMD platforms for comparison")
        return {}
    
    # Use most recent results
    nvidia = sorted(nvidia_results, key=lambda x: x['timestamp'])[-1]
    amd = sorted(amd_results, key=lambda x: x['timestamp'])[-1]
    
    comparison = {
        "nvidia": {
            "device": nvidia['gpu_info']['device_name'],
            "num_gpus": nvidia['gpu_info'].get('device_count', nvidia['training_config'].get('num_gpus', 'N/A')),
            "avg_step_time": nvidia['performance_metrics']['avg_step_time_seconds'],
            "tokens_per_second": nvidia['performance_metrics'].get('tokens_per_second'),
            "tokens_per_second_per_gpu": nvidia['performance_metrics'].get('tokens_per_second_per_gpu'),
            "steps_per_second": nvidia['performance_metrics'].get('steps_per_second'),
            "peak_memory": nvidia.get('memory_metrics', {}).get('peak_memory_allocated_gb', 'N/A'),
        },
        "amd": {
            "device": amd['gpu_info']['device_name'],
            "num_gpus": amd['gpu_info'].get('device_count', amd['training_config'].get('num_gpus', 'N/A')),
            "avg_step_time": amd['performance_metrics']['avg_step_time_seconds'],
            "tokens_per_second": amd['performance_metrics'].get('tokens_per_second'),
            "tokens_per_second_per_gpu": amd['performance_metrics'].get('tokens_per_second_per_gpu'),
            "steps_per_second": amd['performance_metrics'].get('steps_per_second'),
            "peak_memory": amd.get('memory_metrics', {}).get('peak_memory_allocated_gb', 'N/A'),
        }
    }
    
    # Calculate speedup based on time (lower is better)
    nvidia_time = nvidia['performance_metrics']['avg_step_time_seconds']
    amd_time = amd['performance_metrics']['avg_step_time_seconds']
    
    comparison["speedup"] = {
        "faster_platform": "NVIDIA" if nvidia_time < amd_time else "AMD",
        "speedup_factor": max(nvidia_time, amd_time) / min(nvidia_time, amd_time),
        "time_difference_seconds": abs(nvidia_time - amd_time),
    }
    
    # Add throughput ratios (tokens/sec/GPU - higher is better)
    if comparison["nvidia"]["tokens_per_second_per_gpu"] and comparison["amd"]["tokens_per_second_per_gpu"]:
        comparison["speedup"]["tokens_per_gpu_ratio"] = (
            comparison["nvidia"]["tokens_per_second_per_gpu"] / comparison["amd"]["tokens_per_second_per_gpu"]
        )
    elif comparison["nvidia"]["steps_per_second"] and comparison["amd"]["steps_per_second"]:
        comparison["speedup"]["tokens_per_gpu_ratio"] = (
            comparison["nvidia"]["steps_per_second"] / comparison["amd"]["steps_per_second"]
        )
    
    # Print comparison
    print(f"\n{'='*80}")
    print("AMD vs NVIDIA GPU COMPARISON")
    print(f"{'='*80}")
    print(f"\nNVIDIA ({comparison['nvidia']['device']}):")
    print(f"  GPUs:            {comparison['nvidia']['num_gpus']}")
    print(f"  Avg Step Time:   {comparison['nvidia']['avg_step_time']:.4f}s")
    
    if comparison['nvidia']['tokens_per_second_per_gpu']:
        print(f"  Throughput:      {comparison['nvidia']['tokens_per_second']:,.0f} tokens/sec (total)")
        print(f"  Tokens/sec/GPU:  {comparison['nvidia']['tokens_per_second_per_gpu']:,.0f}")
    else:
        print(f"  Throughput:      {comparison['nvidia']['steps_per_second']:.3f} steps/s")
    
    print(f"  Peak Memory:     {comparison['nvidia']['peak_memory']}GB")
    
    print(f"\nAMD ({comparison['amd']['device']}):")
    print(f"  GPUs:            {comparison['amd']['num_gpus']}")
    print(f"  Avg Step Time:   {comparison['amd']['avg_step_time']:.4f}s")
    
    if comparison['amd']['tokens_per_second_per_gpu']:
        print(f"  Throughput:      {comparison['amd']['tokens_per_second']:,.0f} tokens/sec (total)")
        print(f"  Tokens/sec/GPU:  {comparison['amd']['tokens_per_second_per_gpu']:,.0f}")
    else:
        print(f"  Throughput:      {comparison['amd']['steps_per_second']:.3f} steps/s")
    
    print(f"  Peak Memory:     {comparison['amd']['peak_memory']}GB")
    
    print(f"\nResult:")
    print(f"  {comparison['speedup']['faster_platform']} is {comparison['speedup']['speedup_factor']:.2f}x faster (by time)")
    if 'tokens_per_gpu_ratio' in comparison['speedup']:
        print(f"  Tokens/sec/GPU ratio (NVIDIA/AMD): {comparison['speedup']['tokens_per_gpu_ratio']:.2f}x")
    
    print(f"{'='*80}\n")
    
    return comparison


if __name__ == "__main__":
    # Run comparison if executed directly
    compare_benchmarks()

