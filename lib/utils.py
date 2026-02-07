import time
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import torch
from lightning.pytorch.callbacks import Callback

try:
    from transformers import TrainerCallback, TrainerState, TrainerControl, TrainingArguments
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    TrainerCallback = object


def round_floats(obj: Any, precision: int = 5) -> Any:
    if isinstance(obj, float):
        if abs(obj) < 0.001 and obj != 0:
            return round(obj, 10)
        return round(obj, precision)
    elif isinstance(obj, dict):
        return {key: round_floats(value, precision) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [round_floats(item, precision) for item in obj]
    else:
        return obj


def get_gpu_core_count(device_name: str, device_props) -> int:
    device_name_lower = device_name.lower()
    
    nvidia_cores = {
        "h100": 16896,
        "h100 sxm5": 16896,
        "h100 pcie": 14592,
        "a100": 6912,
        "a100-sxm4": 6912,
        "a100-pcie": 6912,
        "v100": 5120,
        "v100-sxm2": 5120,
        "v100-pcie": 5120,
        "a40": 10752,
        "a30": 10752,
        "a10": 9216,
        "rtx 4090": 16384,
        "rtx 3090": 10496,
        "rtx 3080": 8704,
    }
    
    amd_cores = {
        "mi300x": 19456,
        "mi300a": 19456,
        "mi250x": 14080 * 2,
        "mi250": 13312 * 2,
        "mi210": 13312,
        "mi100": 7680,
        "instinct mi300x": 19456,
        "instinct mi250x": 14080 * 2,
        "instinct mi250": 13312 * 2,
        "instinct mi210": 13312,
        "instinct mi100": 7680,
    }
    
    for gpu_name, cores in nvidia_cores.items():
        if gpu_name in device_name_lower:
            return cores
    
    for gpu_name, cores in amd_cores.items():
        if gpu_name in device_name_lower:
            return cores
    
    if hasattr(device_props, 'multi_processor_count'):
        return device_props.multi_processor_count * 128
    
    return 0


def detect_gpu_info() -> Dict[str, Any]:
    gpu_info = {}
    
    try:
        import torch
        TORCH_AVAILABLE = True
    except ImportError:
        TORCH_AVAILABLE = False
    
    if not TORCH_AVAILABLE:
        gpu_info = {
            "device_count": "N/A",
            "device_name": "AMD GPU (from log)",
            "total_memory_gb": 192,
            "gpu_cores": 19456,
            "pytorch_version": "N/A",
            "software_stack": "rocm",
            "software_version": "N/A",
        }
        return gpu_info
    
    if torch.cuda.is_available():
        device_props = torch.cuda.get_device_properties(0)
        device_name = torch.cuda.get_device_name(0)
        gpu_cores = get_gpu_core_count(device_name, device_props)
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
        gpu_info = {
            "device_count": "N/A",
            "device_name": "Unknown (from logs)",
            "total_memory_gb": "N/A",
            "gpu_cores": 0,
            "pytorch_version": torch.__version__ if TORCH_AVAILABLE else "N/A",
            "software_stack": "rocm",
            "software_version": "N/A",
        }
    
    return gpu_info


def extract_param_count_from_log(log_file: str) -> Optional[int]:
    """Extract model parameter count from Megatron/Primus training log.
    
    Megatron logs lines like:
        'number of parameters on (tensor, pipeline) model parallel rank (0, 0): 8030261248'
        'number of parameters: 8030261248'
    """
    patterns = [
        r'number of parameters on.*?rank\s*\(0,\s*0\)[:\s]+(\d+)',
        r'number of parameters[:\s]+(\d+)',
        r'total parameters[:\s]+([0-9,]+)',
        r'\[DIAG\]\s*Total parameters[:\s]+([0-9,]+)',
    ]
    with open(log_file, 'r') as f:
        for line in f:
            for pattern in patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    try:
                        return int(match.group(1).replace(',', ''))
                    except (ValueError, IndexError):
                        continue
    return None


def extract_step_times_from_log(log_file: str) -> Tuple[List[float], List[float], List[float], List[float]]:
    step_times = []
    tokens_per_gpu_values = []
    loss_values = []
    learning_rates = []
    
    with open(log_file, 'r') as f:
        for line in f:
            match = re.search(r'elapsed time per iteration \(ms\):\s*([0-9.]+)/([0-9.]+)', line)
            if match:
                try:
                    step_time_ms = float(match.group(1))
                    step_time_s = step_time_ms / 1000.0
                    if 0.001 < step_time_s < 1000:
                        step_times.append(step_time_s)
                except (ValueError, IndexError):
                    continue
            
            tokens_match = re.search(r'tokens per GPU \(tokens/s/GPU\):\s*([0-9.]+)/([0-9.]+)', line)
            if tokens_match:
                try:
                    tokens_per_gpu = float(tokens_match.group(1))
                    if 0 < tokens_per_gpu < 1000000:
                        tokens_per_gpu_values.append(tokens_per_gpu)
                except (ValueError, IndexError):
                    continue
            
            loss_match = re.search(r'lm loss:\s*([0-9.Ee+-]+)', line)
            if loss_match:
                try:
                    loss = float(loss_match.group(1))
                    if 0 < loss < 10000:
                        loss_values.append(loss)
                except (ValueError, IndexError):
                    continue
            
            lr_match = re.search(r'learning rate:\s*([0-9.Ee+-]+)', line)
            if lr_match:
                try:
                    lr = float(lr_match.group(1))
                    if 0 < lr < 1:
                        learning_rates.append(lr)
                except (ValueError, IndexError):
                    continue
    
    return step_times, tokens_per_gpu_values, loss_values, learning_rates


def extract_memory_from_log(log_file: str) -> List[float]:
    """Extract per-iteration memory values (in GB) from training log.
    
    Supports multiple formats:
    - Megatron: "mem-alloc-GB: 72.5", "memory (GB) | allocated: 72.5"
    - Primus/ROCm: "hip mem usage/free/total/usage_ratio: 72.50GiB/..."
    - Primus debug: "memory (MB) | allocated: 54638.4"
    - Generic: "memory: 72.5 GB", "allocated: 72.5 GB"
    """
    memory_values = []
    
    # Patterns to match various memory log formats (order matters - more specific first)
    # Each tuple: (pattern, unit) where unit is 'gb', 'gib', or 'mb'
    patterns = [
        # Megatron format: "mem-alloc-GB: 72.5" or "mem-reserved-GB: 72.5"
        (r'mem-alloc-GB[:\s]+([0-9.]+)', 'gb'),
        # Megatron format: "memory (GB) | allocated: 72.5"
        (r'memory\s*\(GB\)\s*\|\s*allocated[:\s]+([0-9.]+)', 'gb'),
        # Megatron format: "gpu_memory_allocated: 72.5"
        (r'gpu_memory_allocated[:\s]+([0-9.]+)', 'gb'),
        # Primus/ROCm: "hip mem usage/free/total/usage_ratio: 72.50GiB/..."
        (r'hip mem usage[^:]*:\s*([0-9.]+)\s*GiB', 'gib'),
        # HIP/ROCm format: "hip mem allocated: 72.5 GB" or "hip mem usage: 72.5 GB"
        (r'hip mem (?:usage|allocated)[^:]*:\s*([0-9.]+)\s*GB', 'gb'),
        # Primus debug: "memory (MB) | allocated: 54638.4"
        (r'memory\s*\(MB\)\s*\|\s*allocated[:\s]+([0-9.]+)', 'mb'),
        # Generic: "allocated: 72.5 GB" or "max allocated: 72.5 GB"
        (r'(?:max\s+)?allocated[:\s]+([0-9.]+)\s*GB', 'gb'),
        # Generic: "memory: 72.5 GB" or "memory usage: 72.5 GB"
        (r'memory\s*(?:usage)?[:\s]+([0-9.]+)\s*GB', 'gb'),
        # Megatron throughput line with memory: "| mem-alloc-GB: 72.5 |"
        (r'\|\s*mem-alloc-GB[:\s]+([0-9.]+)\s*\|', 'gb'),
    ]
    
    with open(log_file, 'r') as f:
        for line in f:
            for pattern, unit in patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    try:
                        raw_val = float(match.group(1))
                        # Convert to GB
                        if unit == 'gib':
                            memory_gb = raw_val * 1.073741824  # GiB -> GB
                        elif unit == 'mb':
                            memory_gb = raw_val / 1000.0
                        else:
                            memory_gb = raw_val
                        if 0 < memory_gb < 500:  # Reasonable GPU memory range
                            memory_values.append(round(memory_gb, 2))
                            break  # Found a match, move to next line
                    except (ValueError, IndexError):
                        pass
    
    return memory_values


def parse_memory_log(log_file: str, num_steps: int = None) -> Optional[Dict[str, Any]]:
    """Parse rocm-smi or nvidia-smi memory log to extract memory values.
    
    Converts all values to decimal GB (1 GB = 1e9 bytes) for consistency
    with torch.cuda.memory_allocated() / 1e9 used elsewhere.
    
    Args:
        log_file: Path to memory log file
        num_steps: If provided, interpolate memory values to match step count
    
    Returns:
        Dict with memory_values array and summary stats, or None if no data found.
    """
    if not os.path.exists(log_file):
        return None
    
    raw_values = []
    
    with open(log_file, 'r') as f:
        for line in f:
            # rocm-smi format: "VRAM Total Used Memory (B): 73014444032"
            match = re.search(r'Used.*\(B\)[:\s]+(\d+)', line)
            if match:
                bytes_val = int(match.group(1))
                raw_values.append(bytes_val / 1e9)  # bytes -> GB
                continue
            
            # rocm-smi format: "Used: 69632 MB" (GPU tools report MiB)
            match = re.search(r'Used[:\s]+(\d+)\s*MB', line, re.IGNORECASE)
            if match:
                mib_val = int(match.group(1))
                raw_values.append(mib_val * 1048576 / 1e9)  # MiB -> bytes -> GB
                continue
            
            # nvidia-smi CSV format: "0, 65432" (index, memory_mib)
            match = re.search(r'^\d+,\s*(\d+)', line)
            if match:
                mib_val = int(match.group(1))
                raw_values.append(mib_val * 1048576 / 1e9)  # MiB -> bytes -> GB
                continue
    
    if not raw_values:
        return None
    
    # Interpolate to match step count if requested
    if num_steps and num_steps > 0 and len(raw_values) != num_steps:
        import numpy as np
        raw_indices = np.linspace(0, len(raw_values) - 1, len(raw_values))
        step_indices = np.linspace(0, len(raw_values) - 1, num_steps)
        memory_values = list(np.interp(step_indices, raw_indices, raw_values))
    else:
        memory_values = raw_values
    
    memory_values = [round(v, 2) for v in memory_values]
    
    return {
        "memory_values": memory_values,
        "peak_memory_gb": round(max(memory_values), 2),
        "avg_memory_gb": round(sum(memory_values) / len(memory_values), 2),
        "min_memory_gb": round(min(memory_values), 2),
        "raw_samples": len(raw_values),
    }


def get_parallelism_config(strategy: str, model: str, platform: str) -> Dict[str, Any]:
    return {
        "strategy": strategy or "unknown",
        "tensor_parallel_size": int(os.environ.get("TP", 1)),
        "pipeline_parallel_size": int(os.environ.get("PP", 1)),
        "data_parallel_size": int(os.environ.get("DP", 4)),
        "gradient_accumulation_steps": int(os.environ.get("GA", 32)),
    }


def print_summary(results: Dict[str, Any]) -> None:
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
    def __init__(self, output_dir: str = "./output", platform: str = "auto", model_name: str = None, 
                 parallel_strategy: str = "unknown", framework: str = None, dataset: str = None,
                 warmup_steps: int = 3):
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if platform == "auto":
            if torch.cuda.is_available():
                is_rocm = hasattr(torch.version, 'hip') and torch.version.hip is not None
                self.platform = "amd" if is_rocm else "nvd"
            else:
                self.platform = "cpu"
        else:
            self.platform = platform
        
        self.warmup_steps = warmup_steps
        self._batch_count = 0
        self.step_times = []
        self.memory_allocated = []
        self.memory_reserved = []
        self.memory_allocated_per_gpu = []
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
        self.dataset = dataset or os.environ.get("DATASET", "bc")
        
    def on_train_start(self, trainer, pl_module):
        self.train_start_time = time.time()
        self.num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        
        if hasattr(trainer, 'datamodule'):
            self.global_batch_size = getattr(trainer.datamodule, 'global_batch_size', None)
            self.sequence_length = getattr(trainer.datamodule, 'seq_length', 
                                          getattr(trainer.datamodule, 'sequence_length', 2048))
        
        # Count model parameters
        self.total_params = sum(p.numel() for p in pl_module.parameters())
        self.trainable_params = sum(p.numel() for p in pl_module.parameters() if p.requires_grad)
        
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
        if trainer.is_global_zero:
            software_stack = self.gpu_info.get("software_stack", "unknown")
            print(f"\n{'='*60}")
            print(f"BENCHMARK START - Platform: {self.platform.upper()} ({software_stack.upper()})")
            print(f"{'='*60}")
            for key, value in self.gpu_info.items():
                print(f"{key}: {value}")
            print(f"[DIAG] Total parameters: {self.total_params:,}")
            print(f"[DIAG] Trainable parameters: {self.trainable_params:,}")
            print(f"{'='*60}\n")
    
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self._batch_count += 1
        
        # Log token IDs from the first 3 micro-batches for data pipeline verification
        if self._batch_count <= 3 and trainer.is_global_zero and batch is not None:
            try:
                tokens = None
                if isinstance(batch, dict):
                    tokens = batch.get('tokens', batch.get('input_ids', batch.get('text', None)))
                elif isinstance(batch, (list, tuple)) and len(batch) > 0:
                    tokens = batch[0]
                elif torch.is_tensor(batch):
                    tokens = batch
                
                if tokens is not None and torch.is_tensor(tokens):
                    first_seq = tokens[0] if tokens.dim() > 1 else tokens
                    ids = first_seq[:32].tolist()
                    print(f"[DIAG] Batch {self._batch_count} tokens (first 32): {ids}")
                    print(f"[DIAG] Batch {self._batch_count} shape: {list(tokens.shape)}, "
                          f"dtype: {tokens.dtype}, min: {tokens.min().item()}, max: {tokens.max().item()}")
                else:
                    print(f"[DIAG] Batch {self._batch_count}: type={type(batch).__name__}, "
                          f"could not extract token IDs")
            except Exception as e:
                print(f"[DIAG] Batch {self._batch_count}: error extracting tokens: {e}")
        
        # During warmup steps, skip timing (CUDA kernels are still compiling)
        if self._batch_count <= self.warmup_steps:
            self.step_start_time = None
            self._step_lr = None
            return
        self.step_start_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        # Save LR at the START of the step (before scheduler advances)
        # so the recorded value matches the LR actually used for this update.
        try:
            self._step_lr = trainer.optimizers[0].param_groups[0]['lr']
        except (IndexError, KeyError, AttributeError):
            self._step_lr = None
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Skip recording during warmup steps
        if self.step_start_time is None:
            if trainer.is_global_zero and self._batch_count <= self.warmup_steps:
                print(f"[{self.platform.upper()}] Warmup step {self._batch_count}/{self.warmup_steps} (un-timed)")
            return

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        step_time = time.time() - self.step_start_time
        self.step_times.append(step_time)
        
        if outputs is not None:
            if isinstance(outputs, dict):
                loss = outputs.get('loss', None)
            elif hasattr(outputs, 'loss'):
                loss = outputs.loss
            else:
                loss = None
                
            if loss is not None:
                if torch.is_tensor(loss):
                    loss = loss.item()
                self.loss_values.append(float(loss))
        
        if torch.cuda.is_available():
            mem_allocated = torch.cuda.memory_allocated() / 1e9
            # Total VRAM used (matches nvidia-smi/rocm-smi) for cross-platform consistency
            free, total = torch.cuda.mem_get_info()
            mem_reserved = (total - free) / 1e9
            self.memory_allocated.append(mem_allocated)
            self.memory_reserved.append(mem_reserved)
            
            if torch.distributed.is_initialized():
                try:
                    curr_mem = torch.tensor([mem_allocated], device=f"cuda:{torch.cuda.current_device()}")
                    world_size = torch.distributed.get_world_size()
                    all_mems = [torch.zeros(1, device=f"cuda:{torch.cuda.current_device()}") for _ in range(world_size)]
                    torch.distributed.all_gather(all_mems, curr_mem)
                    per_gpu_mems = [m.item() for m in all_mems]
                    self.memory_allocated_per_gpu.append(per_gpu_mems)
                except Exception:
                    self.memory_allocated_per_gpu.append([mem_allocated])
            else:
                self.memory_allocated_per_gpu.append([mem_allocated])
        
        # Use LR saved at batch start (before scheduler advanced)
        if self._step_lr is not None:
            self.learning_rates.append(float(self._step_lr))
        
        measured_step = len(self.step_times)
        if trainer.is_global_zero and measured_step > 0 and measured_step % 10 == 0:
            recent_times = self.step_times[-10:]
            avg_time = sum(recent_times) / len(recent_times)
            
            loss_str = ""
            if self.loss_values:
                recent_loss = self.loss_values[-1] if self.loss_values else 0
                loss_str = f" | Loss: {recent_loss:.4f}"
            
            if torch.cuda.is_available():
                avg_mem = sum(self.memory_allocated[-10:]) / len(self.memory_allocated[-10:])
                print(f"[{self.platform.upper()}] Step {measured_step:3d} | "
                      f"Time: {step_time:.3f}s | Avg: {avg_time:.3f}s | "
                      f"Memory: {avg_mem:.2f}GB{loss_str}")
            else:
                print(f"[{self.platform.upper()}] Step {measured_step:3d} | "
                      f"Time: {step_time:.3f}s | Avg: {avg_time:.3f}s{loss_str}")
    
    def on_train_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
            
        total_time = time.time() - self.train_start_time
        
        if len(self.step_times) > 1:
            # Skip first step (JIT/compilation warmup), matching extract.py
            step_times_no_warmup = self.step_times[1:]
            avg_step_time = sum(step_times_no_warmup) / len(step_times_no_warmup)
            steps_per_second = len(step_times_no_warmup) / sum(step_times_no_warmup)
            
            tokens_per_second = None
            tokens_per_second_per_gpu = None
            
            if self.global_batch_size and self.sequence_length:
                tokens_per_step = self.global_batch_size * self.sequence_length
                tokens_per_second = tokens_per_step / avg_step_time
                tokens_per_second_per_gpu = tokens_per_second / self.num_gpus if self.num_gpus else None
            
            parallelism_info = {}
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
                    if self.global_batch_size and hasattr(trainer.datamodule, 'micro_batch_size'):
                        parallelism_info["gradient_accumulation_steps"] = self.global_batch_size // (
                            trainer.datamodule.micro_batch_size * parallelism_info["data_parallel_size"]
                        )
            except Exception:
                pass
            
            results = {
                "platform": self.platform,
                "dataset": self.dataset,
                "gpu_info": self.gpu_info,
                "timestamp": datetime.now().isoformat(),
                "model_info": {
                    "total_params": self.total_params,
                    "trainable_params": self.trainable_params,
                },
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
                    "tokens_per_second": tokens_per_second,
                    "tokens_per_second_per_gpu": tokens_per_second_per_gpu,
                    "throughput_per_gpu_core": steps_per_second / self.gpu_info["gpu_cores"] if self.gpu_info.get("gpu_cores", 0) > 0 else 0,
                },
                "step_times": self.step_times,
                "loss_values": self.loss_values if self.loss_values else [],
                "learning_rates": self.learning_rates if self.learning_rates else [],
            }
            
            if self.memory_allocated:
                results["memory_metrics"] = {
                    "peak_memory_allocated_gb": max(self.memory_allocated),
                    "avg_memory_allocated_gb": sum(self.memory_allocated) / len(self.memory_allocated),
                    "min_memory_allocated_gb": min(self.memory_allocated),
                    "measurement_method": "torch.cuda.memory_allocated",
                }
            if self.memory_reserved:
                results.setdefault("memory_metrics", {}).update({
                    "peak_memory_reserved_gb": max(self.memory_reserved),
                    "avg_memory_reserved_gb": sum(self.memory_reserved) / len(self.memory_reserved),
                    "min_memory_reserved_gb": min(self.memory_reserved),
                })
            
            dataset_suffix = f"_{self.dataset}" if self.dataset else ""
            if self.framework and self.model_name:
                filename = f"train_{self.framework}_{self.model_name}{dataset_suffix}.json"
            elif self.model_name:
                software_stack = self.gpu_info.get("software_stack", self.platform)
                filename = f"train_{software_stack}_{self.model_name}{dataset_suffix}.json"
            else:
                software_stack = self.gpu_info.get("software_stack", self.platform)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"train_{software_stack}_{timestamp}{dataset_suffix}.json"
            
            filepath = self.output_dir / filename
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
            
            mem_metrics = results.get('memory_metrics', {})
            if mem_metrics:
                alloc = mem_metrics.get('avg_memory_allocated_gb')
                resv = mem_metrics.get('avg_memory_reserved_gb')
                print(f"\nMemory (avg per GPU):")
                if alloc:
                    print(f"  Allocated: {alloc:.2f} GB")
                if resv:
                    print(f"  Reserved (VRAM used): {resv:.2f} GB")
            
            print(f"\nResults saved to: {filepath}")
            print(f"{'='*60}\n")


class BenchmarkCallbackTran(TrainerCallback):
    def __init__(self, output_dir: str = "./output", platform: str = "auto", model_name: str = None, 
                 parallel_strategy: str = "unknown", framework: str = None, dataset: str = None):
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if platform == "auto":
            if torch.cuda.is_available():
                is_rocm = hasattr(torch.version, 'hip') and torch.version.hip is not None
                self.platform = "amd" if is_rocm else "nvd"
            else:
                self.platform = "cpu"
        else:
            self.platform = platform
        
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
        self.dataset = dataset or os.environ.get("DATASET", "bc")
    
    def on_train_begin(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        self.train_start_time = time.time()
        self.num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        self.global_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps * self.num_gpus
        self.sequence_length = 2048
        
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
        self.step_start_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    
    def on_step_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        step_time = time.time() - self.step_start_time
        self.step_times.append(step_time)
        
        if len(state.log_history) > 0:
            latest_log = state.log_history[-1]
            if 'loss' in latest_log:
                self.loss_values.append(float(latest_log['loss']))
            if 'learning_rate' in latest_log:
                self.learning_rates.append(float(latest_log['learning_rate']))
        
        if torch.cuda.is_available():
            mem_allocated = torch.cuda.memory_allocated() / 1e9
            # Total VRAM used (matches nvidia-smi/rocm-smi) for cross-platform consistency
            free, total = torch.cuda.mem_get_info()
            mem_reserved = (total - free) / 1e9
            self.memory_allocated.append(mem_allocated)
            self.memory_reserved.append(mem_reserved)
        
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
        is_main_process = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
        if not is_main_process:
            return
            
        total_time = time.time() - self.train_start_time
        
        if len(self.step_times) > 1:
            step_times_no_warmup = self.step_times[1:]
            avg_step_time = sum(step_times_no_warmup) / len(step_times_no_warmup)
            steps_per_second = len(step_times_no_warmup) / sum(step_times_no_warmup)
            
            tokens_per_second = None
            tokens_per_second_per_gpu = None
            
            if self.global_batch_size and self.sequence_length:
                tokens_per_step = self.global_batch_size * self.sequence_length
                tokens_per_second = tokens_per_step / avg_step_time
                tokens_per_second_per_gpu = tokens_per_second / self.num_gpus if self.num_gpus else None
            
            results = {
                "platform": self.platform,
                "dataset": self.dataset,
                "gpu_info": self.gpu_info,
                "timestamp": datetime.now().isoformat(),
                "training_config": {
                    "max_steps": args.max_steps,
                    "global_batch_size": self.global_batch_size,
                    "micro_batch_size": args.per_device_train_batch_size,
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
            
            if self.memory_allocated:
                results["memory_metrics"] = {
                    "peak_memory_allocated_gb": max(self.memory_allocated),
                    "avg_memory_allocated_gb": sum(self.memory_allocated) / len(self.memory_allocated),
                    "min_memory_allocated_gb": min(self.memory_allocated),
                    "measurement_method": "torch.cuda.memory_allocated",
                }
            if self.memory_reserved:
                results.setdefault("memory_metrics", {}).update({
                    "peak_memory_reserved_gb": max(self.memory_reserved),
                    "avg_memory_reserved_gb": sum(self.memory_reserved) / len(self.memory_reserved),
                    "min_memory_reserved_gb": min(self.memory_reserved),
                })
            
            dataset_suffix = f"_{self.dataset}" if self.dataset else ""
            if self.framework and self.model_name:
                filename = f"train_{self.framework}_{self.model_name}{dataset_suffix}.json"
            elif self.model_name:
                software_stack = self.gpu_info.get("software_stack", self.platform)
                filename = f"train_{software_stack}_{self.model_name}{dataset_suffix}.json"
            else:
                software_stack = self.gpu_info.get("software_stack", self.platform)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"train_{software_stack}_{timestamp}{dataset_suffix}.json"
            
            filepath = self.output_dir / filename
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
            
            mem_metrics = results.get('memory_metrics', {})
            if mem_metrics:
                alloc = mem_metrics.get('avg_memory_allocated_gb')
                resv = mem_metrics.get('avg_memory_reserved_gb')
                print(f"\nMemory (avg per GPU):")
                if alloc:
                    print(f"  Allocated: {alloc:.2f} GB")
                if resv:
                    print(f"  Reserved (VRAM used): {resv:.2f} GB")
            
            print(f"\nResults saved to: {filepath}")
            print(f"{'='*60}\n")
