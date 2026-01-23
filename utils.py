"""
Unified benchmarking utilities for AMD vs NVIDIA GPU comparison.
Works on both ROCm and CUDA platforms.

Note: torch.cuda.* APIs work for both CUDA and ROCm thanks to HIP compatibility layer.
ROCm provides CUDA API compatibility, so code written for CUDA works on ROCm GPUs.
"""
import time
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import torch
from lightning.pytorch.callbacks import Callback

# Try to import Transformers trainer callback
try:
    from transformers import TrainerCallback, TrainerState, TrainerControl, TrainingArguments
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    TrainerCallback = object  # Dummy base class

# Try to import config_loader
try:
    from config_loader import load_config
    CONFIG_LOADER_AVAILABLE = True
except ImportError:
    CONFIG_LOADER_AVAILABLE = False


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
        # Use higher precision for very small numbers (like learning rates)
        # to avoid rounding 5e-6 and 1e-5 to the same value
        if abs(obj) < 0.001 and obj != 0:
            # For learning rates and other small values, preserve more precision
            return round(obj, 10)  # 10 decimal places for small values
        return round(obj, precision)
    elif isinstance(obj, dict):
        return {key: round_floats(value, precision) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [round_floats(item, precision) for item in obj]
    else:
        return obj


def get_gpu_core_count(device_name: str, device_props) -> int:
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
        "instinct mi300x": 19456,
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


def detect_gpu_info() -> Dict[str, Any]:
    """
    Auto-detect GPU information from PyTorch.
    
    Works for both NVIDIA (CUDA) and AMD (ROCm) GPUs.
    torch.cuda.* APIs are compatible with both platforms.
    
    If no GPU is available, returns placeholder info for log analysis.
    """
    gpu_info = {}
    
    try:
        import torch
        TORCH_AVAILABLE = True
    except ImportError:
        TORCH_AVAILABLE = False
        print("ℹ️  PyTorch not available - using log data only")
    
    if not TORCH_AVAILABLE:
        # PyTorch not available - use placeholder info for log analysis
        gpu_info = {
            "device_count": "N/A",
            "device_name": "AMD GPU (from log)",
            "total_memory_gb": 192,  # MI300X default
            "gpu_cores": 19456,  # MI300X default
            "pytorch_version": "N/A",
            "software_stack": "rocm",
            "software_version": "N/A",
        }
        return gpu_info
    
    if torch.cuda.is_available():  # Works for both CUDA and ROCm
        device_props = torch.cuda.get_device_properties(0)
        device_name = torch.cuda.get_device_name(0)
        
        # Detect GPU cores (approximate based on known models)
        gpu_cores = get_gpu_core_count(device_name, device_props)
        
        # Detect software stack and version
        # ROCm sets torch.version.hip, CUDA sets torch.version.cuda
        is_rocm = hasattr(torch.version, 'hip') and torch.version.hip is not None
        software_stack = "rocm" if is_rocm else "cuda"
        software_version = torch.version.hip if is_rocm else torch.version.cuda
        
        gpu_info = {
            "device_count": torch.cuda.device_count(),
            "device_name": device_name,
            "total_memory_gb": device_props.total_memory / 1e9,
            "gpu_cores": gpu_cores,
            "pytorch_version": torch.__version__,
            "software_stack": software_stack,
            "software_version": software_version,
        }
    else:
        # No GPU available - use placeholder info for log analysis
        print("ℹ️  No GPU detected - using log data only")
        gpu_info = {
            "device_count": "N/A",
            "device_name": "Unknown (from logs)",
            "total_memory_gb": "N/A",
            "gpu_cores": 0,
            "pytorch_version": torch.__version__ if TORCH_AVAILABLE else "N/A",
            "software_stack": "rocm",  # Default to rocm for log analysis
            "software_version": "N/A",
        }
    
    return gpu_info


def extract_step_times_from_log(log_file: str) -> Tuple[List[float], List[float], List[float], List[float]]:
    """
    Extract step timing and loss from Prim/Mega logs.
    
    Prim format:
    elapsed time per iteration (ms): 9836.3/21761.7
    lm loss: 1.189761E+01
    learning rate: 5.000000E-06
    
    Returns:
        Tuple of (step_times, tokens_per_gpu_values, loss_values, learning_rates)
    """
    step_times = []
    tokens_per_gpu_values = []
    loss_values = []
    learning_rates = []
    
    with open(log_file, 'r') as f:
        for line in f:
            # Prim format: elapsed time per iteration (ms): 9836.3/21761.7
            # First value is current iteration, second is average
            match = re.search(r'elapsed time per iteration \(ms\):\s*([0-9.]+)/([0-9.]+)', line)
            if match:
                try:
                    # Get current iteration time in ms, convert to seconds
                    step_time_ms = float(match.group(1))
                    step_time_s = step_time_ms / 1000.0
                    
                    if 0.001 < step_time_s < 1000:  # Sanity check
                        step_times.append(step_time_s)
                except (ValueError, IndexError):
                    continue
            
            # Also extract tokens per GPU if available
            # tokens per GPU (tokens/s/GPU): 13325.3/8608.1
            tokens_match = re.search(r'tokens per GPU \(tokens/s/GPU\):\s*([0-9.]+)/([0-9.]+)', line)
            if tokens_match:
                try:
                    tokens_per_gpu = float(tokens_match.group(1))
                    if 0 < tokens_per_gpu < 1000000:  # Sanity check
                        tokens_per_gpu_values.append(tokens_per_gpu)
                except (ValueError, IndexError):
                    continue
            
            # Extract loss values
            # lm loss: 1.189761E+01 or lm loss: 11.89761
            loss_match = re.search(r'lm loss:\s*([0-9.Ee+-]+)', line)
            if loss_match:
                try:
                    loss = float(loss_match.group(1))
                    if 0 < loss < 10000:  # Sanity check
                        loss_values.append(loss)
                except (ValueError, IndexError):
                    continue
            
            # learning rate: 5.000000E-06 or learning rate: 0.000005
            lr_match = re.search(r'learning rate:\s*([0-9.Ee+-]+)', line)
            if lr_match:
                try:
                    lr = float(lr_match.group(1))
                    if 0 < lr < 1:  # Sanity check (learning rates are typically 1e-7 to 1e-3)
                        learning_rates.append(lr)
                except (ValueError, IndexError):
                    continue
    
    return step_times, tokens_per_gpu_values, loss_values, learning_rates


def extract_memory_from_log(log_file: str) -> List[float]:
    """
    Extract GPU memory usage from log.
    
    Supports multiple formats:
    - Prim/Mega: hip mem usage/free/total/usage_ratio: 117.99GB/74.00GB/191.98GB/61.46%
    - Mega-LM: allocated: 60.2GB, max allocated: 60.3GB, reserved: 62.1GB
    - Generic: memory usage: 60.5 GB
    
    Note: "hip" refers to AMD's HIP (Heterogeneous Interface for Portability),
    which provides CUDA compatibility on ROCm.
    """
    memory_values = []
    
    with open(log_file, 'r') as f:
        for line in f:
            # Format 1: Prim/ROCm - hip mem usage/free/total/usage_ratio: 117.99GB/...
            match = re.search(r'hip mem (?:usage|allocated)[^:]*:\s*([0-9.]+)\s*GB', line, re.IGNORECASE)
            if match:
                try:
                    memory_gb = float(match.group(1))
                    if 0 < memory_gb < 1000:  # Sanity check
                        memory_values.append(memory_gb)
                        continue
                except (ValueError, IndexError):
                    pass
            
            # Format 2: Mega-LM - allocated: 60.2GB or max allocated: 60.3GB
            match = re.search(r'(?:max )?allocated:\s*([0-9.]+)\s*GB', line, re.IGNORECASE)
            if match:
                try:
                    memory_gb = float(match.group(1))
                    if 0 < memory_gb < 1000:  # Sanity check
                        memory_values.append(memory_gb)
                        continue
                except (ValueError, IndexError):
                    pass
            
            # Format 3: Generic - memory usage: 60.5 GB or memory: 60.5GB
            match = re.search(r'memory\s+(?:usage)?:?\s*([0-9.]+)\s*GB', line, re.IGNORECASE)
            if match:
                try:
                    memory_gb = float(match.group(1))
                    if 0 < memory_gb < 1000:  # Sanity check
                        memory_values.append(memory_gb)
                        continue
                except (ValueError, IndexError):
                    pass
    
    return memory_values


def get_parallelism_config(strategy: str, model: str, platform: str) -> Dict[str, Any]:
    """
    Get parallelism configuration from config.yaml.
    
    Args:
        strategy: Parallelism strategy name
        model: Model name (llama, qwen)
        platform: Platform (amd, nvidia/nvd)
        
    Returns:
        Dictionary with parallelism parameters or minimal info if unavailable
    """
    if not CONFIG_LOADER_AVAILABLE or not strategy or strategy == "unknown":
        return {"strategy": strategy or "unknown"}
    
    try:
        config = load_config()
        # Normalize platform name (nvd -> nvidia)
        platform_normalized = "nvidia" if platform in ["nvd", "nvidia"] else "amd"
        
        # Get parallelism config for this strategy/model/platform
        params = config.get_parallelism(model, platform_normalized, methodology=strategy)
        
        return {
            "strategy": strategy,
            "tensor_parallel_size": params.get("tensor_model_parallel_size", 1),
            "pipeline_parallel_size": params.get("pipeline_model_parallel_size", 1),
            "data_parallel_size": params.get("data_parallel_size", 1),
            "gradient_accumulation_steps": params.get("gradient_accumulation_steps", 1),
        }
    except Exception as e:
        print(f"[!] Could not load parallelism config: {e}")
        return {"strategy": strategy or "unknown"}


def analyze_tensorboard_events(log_dir: str) -> Optional[Dict[str, Any]]:
    """
    Analyze TensorBoard event files from NeMo training.
    
    Args:
        log_dir: Directory containing TensorBoard event files
        
    Returns:
        Dictionary with metrics or None if failed
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
    
    print(f"  * Analyzing {len(event_files)} event file(s)...")
    
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


def print_summary(results: Dict[str, Any]) -> None:
    """Print benchmark summary."""
    print(f"\n{'='*60}")
    print(f"BENCHMARK SUMMARY - Platform: {results['platform'].upper()}")
    print(f"{'='*60}")
    print(f"Device: {results['gpu_info'].get('device_name', 'Unknown')}")
    print(f"GPUs: {results['training_config']['num_gpus']}")
    print(f"Total Steps: {results['performance_metrics']['total_steps']}")
    print(f"Avg Step Time: {results['performance_metrics']['avg_step_time_seconds']:.3f}s")
    
    if results['performance_metrics'].get('tokens_per_second'):
        print(f"\nThroughput Metrics:")
        print(f"  Total Throughput: {results['performance_metrics']['tokens_per_second']:,.0f} tokens/sec")
        print(f"  Per-GPU Throughput: {results['performance_metrics']['tokens_per_second_per_gpu']:,.0f} tokens/sec/GPU")
    
    print(f"{'='*60}\n")


class BenchmarkCallback(Callback):
    """Callback to collect platform-agnostic performance metrics."""
    
    def __init__(self, output_dir: str = "./output", platform: str = "auto", model_name: str = None, 
                 parallel_strategy: str = "unknown", profiler_config: Optional[Dict] = None, framework: str = None):
        """
        Args:
            output_dir: Directory to save benchmark results
            platform: 'cuda', 'rocm', or 'auto' for auto-detection
            model_name: Name of the model (e.g., 'llama', 'qwen')
            parallel_strategy: Parallelism strategy name (e.g., 'minimal_communication', 'balanced')
            profiler_config: Dictionary with profiling configuration (from config.yaml)
            framework: Framework name for output filename (e.g., 'tran', 'nemo')
        """
        self.output_dir = Path(output_dir).resolve()
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
        self.memory_allocated_per_gpu = []  # Added for per-core monitoring
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
        self.framework = framework
        
        # Profiling configuration
        self.profiler_config = profiler_config or {}
        self.profiler = None
        self.profiler_enabled = self.profiler_config.get('enabled', False)
        
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
            gpu_cores = get_gpu_core_count(device_name, device_props)
            
            # Detect software stack and version
            # ROCm sets torch.version.hip, CUDA sets torch.version.cuda
            is_rocm = hasattr(torch.version, 'hip') and torch.version.hip is not None
            software_stack = "prim" if is_rocm else "nemo"
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
        
        # Initialize NVIDIA Nsight Systems profiler if enabled (only on rank 0)
        if self.profiler_enabled and trainer.is_global_zero and torch.cuda.is_available():
            # NVIDIA Nsight Systems profiling - handled via nsys wrapper in pretrain scripts
            print(f"[+] NVIDIA Nsight Systems profiling mode")
            print(f"  Output directory: {self.output_dir}")
            # Start CUDA profiler range for Nsight capture
            try:
                torch.cuda.cudart().cudaProfilerStart()
                self._nsight_profiler_started = True
                print(f"  CUDA profiler range started")
            except Exception:
                self._nsight_profiler_started = False
        
        # Only print on rank 0 to avoid duplicate output in distributed training
        if trainer.is_global_zero:
            software_stack = self.gpu_info.get("software_stack", "unknown")
            print(f"\n{'='*60}")
            print(f"BENCHMARK START - Platform: {self.platform.upper()} ({software_stack.upper()})")
            print(f"{'='*60}")
            for key, value in self.gpu_info.items():
                print(f"{key}: {value}")
            if self.profiler_enabled:
                print(f"profiling: enabled (NSIGHT)")
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
            
            # Collect memory from all ranks if in distributed mode
            if torch.distributed.is_initialized():
                try:
                    # Gather allocated memory from all ranks
                    curr_mem = torch.tensor([mem_allocated], device=f"cuda:{torch.cuda.current_device()}")
                    world_size = torch.distributed.get_world_size()
                    all_mems = [torch.zeros(1, device=f"cuda:{torch.cuda.current_device()}") for _ in range(world_size)]
                    torch.distributed.all_gather(all_mems, curr_mem)
                    
                    # Convert tensors to list of floats
                    per_gpu_mems = [m.item() for m in all_mems]
                    self.memory_allocated_per_gpu.append(per_gpu_mems)
                except Exception as e:
                    # Fallback if gathering fails
                    self.memory_allocated_per_gpu.append([mem_allocated])
            else:
                # Non-distributed mode: just one GPU
                self.memory_allocated_per_gpu.append([mem_allocated])
        
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
        # Finalize NVIDIA Nsight Systems profiler if enabled
        if self.profiler_enabled and hasattr(self, '_nsight_profiler_started') and self._nsight_profiler_started:
            try:
                torch.cuda.cudart().cudaProfilerStop()
                print(f"[+] CUDA profiler range stopped")
                print(f"[+] Nsight profile saved by nsys wrapper")
                print(f"  View with: nsys-ui <profile>.nsys-rep")
                print(f"  Or export: nsys export --type=json <profile>.nsys-rep")
            except Exception as e:
                print(f"[!] Warning: Failed to stop CUDA profiler: {e}")
        
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
            parallel_strategy = os.environ.get('PARALLEL', 'unknown')
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
                "step_times": self.step_times,
                "loss_values": self.loss_values if self.loss_values else [],
                "learning_rates": self.learning_rates if self.learning_rates else [],
            }
            
            # Save results (round all floats to 3 decimal places)
            # Use framework name if provided, otherwise fall back to software stack
            if self.framework and self.model_name:
                # Format: train_<framework>_<model>.json (e.g., train_hf_llama.json)
                filename = f"train_{self.framework}_{self.model_name}.json"
            elif self.model_name:
                # Fallback: use software stack if no framework specified
                software_stack = self.gpu_info.get("software_stack", self.platform)
                filename = f"train_{software_stack}_{self.model_name}.json"
            else:
                # Fallback to timestamp if no model name
                software_stack = self.gpu_info.get("software_stack", self.platform)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"train_{software_stack}_{timestamp}.json"
            
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
            
            print(f"\nResults saved to: {filepath}")
            print(f"{'='*60}\n")


class BenchmarkCallbackTran(TrainerCallback):
    """Transformers Trainer-compatible callback for collecting performance metrics."""
    
    def __init__(self, output_dir: str = "./output", platform: str = "auto", model_name: str = None, 
                 parallel_strategy: str = "unknown", profiler_config: Optional[Dict] = None, framework: str = None):
        """
        Args:
            output_dir: Directory to save benchmark results
            platform: 'cuda', 'rocm', or 'auto' for auto-detection
            model_name: Name of the model (e.g., 'llama', 'qwen')
            parallel_strategy: Parallelism strategy name (e.g., 'ddp')
            profiler_config: Dictionary with profiling configuration (from config.yaml)
            framework: Framework name for output filename (e.g., 'tran', 'deep')
        """
        self.output_dir = Path(output_dir).resolve()
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
        self.framework = framework
        
        # Profiling configuration
        self.profiler_config = profiler_config or {}
        self.profiler_enabled = self.profiler_config.get('enabled', False)
    
    def on_train_begin(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        """Called at the beginning of training."""
        self.train_start_time = time.time()
        
        # Get training configuration
        self.num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        
        # Extract batch size and sequence length from args
        self.global_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps * self.num_gpus
        self.sequence_length = 2048  # Default, will try to detect from data
        
        if torch.cuda.is_available():
            device_props = torch.cuda.get_device_properties(0)
            device_name = torch.cuda.get_device_name(0)
            
            gpu_cores = get_gpu_core_count(device_name, device_props)
            
            is_rocm = hasattr(torch.version, 'hip') and torch.version.hip is not None
            software_stack = "prim" if is_rocm else "nemo"
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
        
        # Only print on rank 0
        is_main_process = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
        if is_main_process:
            software_stack = self.gpu_info.get("software_stack", "unknown")
            print(f"\n{'='*60}")
            print(f"BENCHMARK START - Platform: {self.platform.upper()} ({software_stack.upper()})")
            print(f"{'='*60}")
            for key, value in self.gpu_info.items():
                print(f"{key}: {value}")
            print(f"{'='*60}\n")
    
    def on_step_begin(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        """Called at the beginning of a training step."""
        self.step_start_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    
    def on_step_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        """Called at the end of a training step."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        step_time = time.time() - self.step_start_time
        self.step_times.append(step_time)
        
        # Collect loss value
        if len(state.log_history) > 0:
            latest_log = state.log_history[-1]
            if 'loss' in latest_log:
                self.loss_values.append(float(latest_log['loss']))
            if 'learning_rate' in latest_log:
                self.learning_rates.append(float(latest_log['learning_rate']))
        
        # Collect memory stats
        if torch.cuda.is_available():
            mem_allocated = torch.cuda.memory_allocated() / 1e9
            mem_reserved = torch.cuda.memory_reserved() / 1e9
            self.memory_allocated.append(mem_allocated)
            self.memory_reserved.append(mem_reserved)
        
        # Log every 10 steps (only on rank 0)
        is_main_process = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
        if is_main_process and state.global_step > 0 and state.global_step % 10 == 0:
            recent_times = self.step_times[-10:]
            avg_time = sum(recent_times) / len(recent_times)
            
            loss_str = ""
            if self.loss_values:
                recent_loss = self.loss_values[-1]
                loss_str = f" | Loss: {recent_loss:.4f}"
            
            if torch.cuda.is_available():
                avg_mem = sum(self.memory_allocated[-10:]) / len(self.memory_allocated[-10:])
                print(f"[{self.platform.upper()}] Step {state.global_step:3d} | "
                      f"Time: {step_time:.3f}s | Avg: {avg_time:.3f}s | "
                      f"Memory: {avg_mem:.2f}GB{loss_str}")
    
    def on_train_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        """Called at the end of training."""
        # Only save results on rank 0
        is_main_process = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
        if not is_main_process:
            return
            
        total_time = time.time() - self.train_start_time
        
        # Calculate statistics
        if len(self.step_times) > 1:
            step_times_no_warmup = self.step_times[1:]
            
            avg_step_time = sum(step_times_no_warmup) / len(step_times_no_warmup)
            steps_per_second = len(step_times_no_warmup) / sum(step_times_no_warmup)
            
            # Calculate token-based throughput
            tokens_per_second = None
            tokens_per_second_per_gpu = None
            
            if self.global_batch_size and self.sequence_length:
                tokens_per_step = self.global_batch_size * self.sequence_length
                tokens_per_second = tokens_per_step / avg_step_time
                tokens_per_second_per_gpu = tokens_per_second / self.num_gpus if self.num_gpus else None
            
            results = {
                "platform": self.platform,
                "gpu_info": self.gpu_info,
                "timestamp": datetime.now().isoformat(),
                "training_config": {
                    "max_steps": args.max_steps,
                    "global_batch_size": self.global_batch_size,
                    "per_device_train_batch_size": args.per_device_train_batch_size,
                    "gradient_accumulation_steps": args.gradient_accumulation_steps,
                    "sequence_length": self.sequence_length,
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
                    "tokens_per_second": tokens_per_second,
                    "tokens_per_second_per_gpu": tokens_per_second_per_gpu,
                    "throughput_per_gpu_core": steps_per_second / self.gpu_info["gpu_cores"] if self.gpu_info.get("gpu_cores", 0) > 0 else 0,
                },
                "step_times": self.step_times,
                "loss_values": self.loss_values if self.loss_values else [],
                "learning_rates": self.learning_rates if self.learning_rates else [],
            }
            
            # Save results
            if self.framework and self.model_name:
                filename = f"train_{self.framework}_{self.model_name}.json"
            elif self.model_name:
                software_stack = self.gpu_info.get("software_stack", self.platform)
                filename = f"train_{software_stack}_{self.model_name}.json"
            else:
                software_stack = self.gpu_info.get("software_stack", self.platform)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"train_{software_stack}_{timestamp}.json"
            
            filepath = self.output_dir / filename
            
            # Round all float values
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
            
            if results['performance_metrics']['tokens_per_second']:
                print(f"\nThroughput Metrics:")
                print(f"  Total Throughput: {results['performance_metrics']['tokens_per_second']:,.0f} tokens/sec")
                print(f"  Per-GPU Throughput: {results['performance_metrics']['tokens_per_second_per_gpu']:,.0f} tokens/sec/GPU")
                print(f"  (Global batch size: {self.global_batch_size}, Sequence length: {self.sequence_length})")
            else:
                print(f"Throughput: {results['performance_metrics']['steps_per_second']:.3f} steps/s")
            
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
        # NVIDIA: nemo (new) or cuda (old)
        if software_stack in ['nemo', 'cuda']:
            nvidia_results.append(data)
        # AMD: prim (new) or rocm (old)
        elif software_stack in ['prim', 'rocm']:
            amd_results.append(data)
        # Fallback to platform field (for older files)
        elif platform in ['cuda', 'nvd', 'nvidia']:
            nvidia_results.append(data)
        elif platform in ['rocm', 'amd']:
            amd_results.append(data)
    
    if not nvidia_results or not amd_results:
        print("[!] Need results from both NVIDIA and AMD platforms for comparison")
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
        },
        "amd": {
            "device": amd['gpu_info']['device_name'],
            "num_gpus": amd['gpu_info'].get('device_count', amd['training_config'].get('num_gpus', 'N/A')),
            "avg_step_time": amd['performance_metrics']['avg_step_time_seconds'],
            "tokens_per_second": amd['performance_metrics'].get('tokens_per_second'),
            "tokens_per_second_per_gpu": amd['performance_metrics'].get('tokens_per_second_per_gpu'),
            "steps_per_second": amd['performance_metrics'].get('steps_per_second'),
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

