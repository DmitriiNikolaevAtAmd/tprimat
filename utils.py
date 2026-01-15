"""
Unified benchmarking utilities for AMD vs NVIDIA GPU comparison.
Works on both ROCm and CUDA platforms.

Note: torch.cuda.* APIs work for both CUDA and ROCm thanks to HIP compatibility layer.
ROCm provides CUDA API compatibility, so code written for CUDA works on ROCm GPUs.
"""
import time
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import torch
from lightning.pytorch.callbacks import Callback


def round_floats(obj: Any, precision: int = 5) -> Any:
    """
    Recursively round all float values in a nested structure to specified precision.
    
    Args:
        obj: Dictionary, list, or value to process
        precision: Number of decimal places (default: 5)
    
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
    
    def __init__(self, output_dir: str = "./output", platform: str = "auto", model_name: str = None, 
                 parallel_strategy: str = "unknown", profiler_config: Optional[Dict] = None):
        """
        Args:
            output_dir: Directory to save benchmark results
            platform: 'cuda', 'rocm', or 'auto' for auto-detection
            model_name: Name of the model (e.g., 'llama', 'qwen')
            parallel_strategy: Parallelism strategy name (e.g., 'minimal_communication', 'balanced')
            profiler_config: Dictionary with profiling configuration (from config.yaml)
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
        self.loss_values = []
        self.learning_rates = []
        self.step_start_time = None
        self.train_start_time = None
        self.gpu_info = {}
        self.global_batch_size = None
        self.sequence_length = None
        self.num_gpus = None
        self.model_name = model_name
        self.parallel_strategy = parallel_strategy
        
        # Profiling configuration
        self.profiler_config = profiler_config or {}
        self.profiler = None
        self.profiler_enabled = self.profiler_config.get('enabled', False)
    
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
        """
        Collect GPU information at training start.
        
        Note: torch.cuda APIs work for both NVIDIA (CUDA) and AMD (ROCm) GPUs.
        """
        self.train_start_time = time.time()
        
        # Get training configuration (works for both CUDA and ROCm)
        self.num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        
        # Try to get batch size and sequence length from datamodule
        if hasattr(trainer, 'datamodule'):
            self.global_batch_size = getattr(trainer.datamodule, 'global_batch_size', None)
            self.sequence_length = getattr(trainer.datamodule, 'seq_length', 
                                          getattr(trainer.datamodule, 'sequence_length', 2048))  # Default 2048
        
        if torch.cuda.is_available():  # Works for both CUDA and ROCm
            device_props = torch.cuda.get_device_properties(0)
            device_name = torch.cuda.get_device_name(0)
            
            # Get GPU core count (approximate based on known models)
            gpu_cores = self._get_gpu_core_count(device_name, device_props)
            
            # Detect software stack and version
            # ROCm sets torch.version.hip, CUDA sets torch.version.cuda
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
        
        # Initialize profiler if enabled (only on rank 0)
        if self.profiler_enabled and trainer.is_global_zero and torch.cuda.is_available():
            import torch.profiler as profiler
            
            # Get profiler configuration
            schedule_config = self.profiler_config.get('schedule', {})
            wait_steps = schedule_config.get('wait', 1)
            warmup_steps = schedule_config.get('warmup', 1)
            active_steps = schedule_config.get('active', 5)
            repeat_cycles = schedule_config.get('repeat', 1)
            
            # Setup profiler activities
            activities = [profiler.ProfilerActivity.CPU]
            if torch.cuda.is_available():
                activities.append(profiler.ProfilerActivity.CUDA)
            
            # Create profiler output directory
            profiler_dir = self.output_dir / "profiler"
            profiler_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate filename with model name and timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            software_stack = self.gpu_info.get("software_stack", "unknown")
            if self.model_name:
                trace_prefix = f"profile_{software_stack}_{self.model_name}_{self.parallel_strategy}_{timestamp}"
            else:
                trace_prefix = f"profile_{software_stack}_{timestamp}"
            
            self.profiler = profiler.profile(
                activities=activities,
                schedule=profiler.schedule(
                    wait=wait_steps,
                    warmup=warmup_steps,
                    active=active_steps,
                    repeat=repeat_cycles
                ),
                on_trace_ready=profiler.tensorboard_trace_handler(
                    str(profiler_dir),
                    worker_name=trace_prefix
                ),
                record_shapes=self.profiler_config.get('record_shapes', True),
                profile_memory=self.profiler_config.get('profile_memory', True),
                with_stack=self.profiler_config.get('with_stack', True),
                with_flops=self.profiler_config.get('with_flops', True),
            )
            self.profiler.__enter__()
            
            print(f"✓ Kineto profiler enabled (output: {profiler_dir})")
            print(f"  Schedule: wait={wait_steps}, warmup={warmup_steps}, active={active_steps}, repeat={repeat_cycles}")
        
        # Only print on rank 0 to avoid duplicate output in distributed training
        if trainer.is_global_zero:
            software_stack = self.gpu_info.get("software_stack", "unknown")
            print(f"\n{'='*60}")
            print(f"BENCHMARK START - Platform: {self.platform.upper()} ({software_stack.upper()})")
            print(f"{'='*60}")
            for key, value in self.gpu_info.items():
                print(f"{key}: {value}")
            if self.profiler_enabled:
                print(f"profiling: enabled (Kineto)")
            print(f"{'='*60}\n")
    
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        """Mark start of training step."""
        self.step_start_time = time.time()
        
        # Clear cache for consistent measurements (works for both CUDA and ROCm)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Collect metrics after each training step."""
        if torch.cuda.is_available():  # Works for both CUDA and ROCm
            torch.cuda.synchronize()
        
        # Step profiler if enabled
        if self.profiler is not None:
            self.profiler.step()
        
        step_time = time.time() - self.step_start_time
        self.step_times.append(step_time)
        
        # Collect loss value if available
        if outputs is not None:
            # Try to extract loss from outputs (NeMo format)
            if isinstance(outputs, dict):
                loss = outputs.get('loss', None)
            elif hasattr(outputs, 'loss'):
                loss = outputs.loss
            else:
                loss = None
                
            if loss is not None:
                # Convert to float if it's a tensor
                if torch.is_tensor(loss):
                    loss = loss.item()
                self.loss_values.append(float(loss))
        
        # Collect memory stats (works for both CUDA and ROCm)
        if torch.cuda.is_available():
            mem_allocated = torch.cuda.memory_allocated() / 1e9  # GB
            mem_reserved = torch.cuda.memory_reserved() / 1e9    # GB
            self.memory_allocated.append(mem_allocated)
            self.memory_reserved.append(mem_reserved)
        
        # Collect learning rate from optimizer/scheduler
        try:
            # Try to get learning rate from optimizer (works with NeMo)
            lr = trainer.optimizers[0].param_groups[0]['lr'] if trainer.optimizers else None
            if lr is not None:
                self.learning_rates.append(float(lr))
        except (IndexError, KeyError, AttributeError):
            # If we can't get LR, skip (will extract from logs if needed)
            pass
        
        # Log every 10 steps (only on rank 0)
        if trainer.is_global_zero and batch_idx > 0 and batch_idx % 10 == 0:
            recent_times = self.step_times[-10:]
            avg_time = sum(recent_times) / len(recent_times)
            
            loss_str = ""
            if self.loss_values:
                recent_loss = self.loss_values[-1] if self.loss_values else 0
                loss_str = f" | Loss: {recent_loss:.4f}"
            
            if torch.cuda.is_available():
                avg_mem = sum(self.memory_allocated[-10:]) / len(self.memory_allocated[-10:])
                print(f"[{self.platform.upper()}] Step {batch_idx:3d} | "
                      f"Time: {step_time:.3f}s | Avg: {avg_time:.3f}s | "
                      f"Memory: {avg_mem:.2f}GB{loss_str}")
            else:
                print(f"[{self.platform.upper()}] Step {batch_idx:3d} | "
                      f"Time: {step_time:.3f}s | Avg: {avg_time:.3f}s{loss_str}")
    
    def on_train_end(self, trainer, pl_module):
        """Save benchmark results."""
        # Finalize profiler if enabled
        if self.profiler is not None:
            try:
                self.profiler.__exit__(None, None, None)
                
                # Export additional Chrome trace if configured
                if self.profiler_config.get('export_chrome_trace', True):
                    software_stack = self.gpu_info.get("software_stack", "unknown")
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    if self.model_name:
                        chrome_trace_path = self.output_dir / f"trace_{software_stack}_{self.model_name}_{self.parallel_strategy}_{timestamp}.json"
                    else:
                        chrome_trace_path = self.output_dir / f"trace_{software_stack}_{timestamp}.json"
                    
                    # Note: TensorBoard trace handler already saves traces
                    # This exports a standalone Chrome trace for convenience
                    print(f"✓ Profiler traces saved to: {self.output_dir / 'profiler'}")
                    print(f"  View in TensorBoard: tensorboard --logdir={self.output_dir / 'profiler'}")
                    print(f"  Or Chrome: chrome://tracing (load .json.gz files)")
            except Exception as e:
                print(f"⚠️  Warning: Failed to finalize profiler: {e}")
        
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
            
            # Extract parallelism configuration from trainer strategy
            import os
            parallelism_info = {}
            
            # Get strategy name from environment variable
            parallel_strategy = os.environ.get('TPRIMAT_PARALLEL', 'unknown')
            parallelism_info["strategy_name"] = parallel_strategy
            
            try:
                if hasattr(trainer, 'strategy') and hasattr(trainer.strategy, 'tensor_model_parallel_size'):
                    parallelism_info.update({
                        "tensor_model_parallel_size": trainer.strategy.tensor_model_parallel_size,
                        "pipeline_model_parallel_size": trainer.strategy.pipeline_model_parallel_size,
                        "data_parallel_size": self.num_gpus // (
                            trainer.strategy.tensor_model_parallel_size * 
                            trainer.strategy.pipeline_model_parallel_size
                        ),
                    })
                    # Calculate gradient accumulation steps
                    if self.global_batch_size and hasattr(trainer.datamodule, 'micro_batch_size'):
                        parallelism_info["gradient_accumulation_steps"] = self.global_batch_size // (
                            trainer.datamodule.micro_batch_size * parallelism_info["data_parallel_size"]
                        )
            except Exception:
                # If extraction fails, we still have strategy_name
                pass
            
            results = {
                "platform": self.platform,
                "gpu_info": self.gpu_info,
                "timestamp": datetime.now().isoformat(),
                "parallelism_config": parallelism_info,
                "training_config": {
                    "max_steps": trainer.max_steps,
                    "global_batch_size": self.global_batch_size or 'N/A',
                    "micro_batch_size": getattr(trainer.datamodule, 'micro_batch_size', 'N/A'),
                    "sequence_length": self.sequence_length or 'N/A',
                    "num_gpus": self.num_gpus,
                    "parallel_strategy": self.parallel_strategy,
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
                "raw_loss_values": self.loss_values if self.loss_values else [],
                "raw_memory_values": self.memory_allocated if self.memory_allocated else [],
                "raw_learning_rates": self.learning_rates if self.learning_rates else [],
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
            
            # Round all float values to 5 decimal places
            results_rounded = round_floats(results, precision=5)
            
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
            
            if 'raw_memory_values' in results and results['raw_memory_values']:
                memory_values = results['raw_memory_values']
                print(f"\nMemory Usage (time series):")
                print(f"  Memory samples: {len(memory_values)}")
                print(f"  Avg Memory: {sum(memory_values)/len(memory_values):.2f}GB")
                print(f"  Peak Memory: {max(memory_values):.2f}GB")
            
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
        
        # Prioritize software_stack over platform for accurate detection
        # NVIDIA: cuda
        if software_stack == 'cuda':
            nvidia_results.append(data)
        # AMD: rocm
        elif software_stack == 'rocm':
            amd_results.append(data)
        # Fallback to platform field (for older files)
        elif platform in ['cuda', 'nvd', 'nvidia']:
            nvidia_results.append(data)
        elif platform in ['rocm', 'amd']:
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
            "peak_memory": max(nvidia.get('raw_memory_values', [0])) if nvidia.get('raw_memory_values') else 'N/A',
        },
        "amd": {
            "device": amd['gpu_info']['device_name'],
            "num_gpus": amd['gpu_info'].get('device_count', amd['training_config'].get('num_gpus', 'N/A')),
            "avg_step_time": amd['performance_metrics']['avg_step_time_seconds'],
            "tokens_per_second": amd['performance_metrics'].get('tokens_per_second'),
            "tokens_per_second_per_gpu": amd['performance_metrics'].get('tokens_per_second_per_gpu'),
            "steps_per_second": amd['performance_metrics'].get('steps_per_second'),
            "peak_memory": max(amd.get('raw_memory_values', [0])) if amd.get('raw_memory_values') else 'N/A',
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

