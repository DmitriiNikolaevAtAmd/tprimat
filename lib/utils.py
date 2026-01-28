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
    memory_values = []
    
    with open(log_file, 'r') as f:
        for line in f:
            match = re.search(r'hip mem (?:usage|allocated)[^:]*:\s*([0-9.]+)\s*GB', line, re.IGNORECASE)
            if match:
                try:
                    memory_gb = float(match.group(1))
                    if 0 < memory_gb < 1000:
                        memory_values.append(memory_gb)
                        continue
                except (ValueError, IndexError):
                    pass
            
            match = re.search(r'(?:max )?allocated:\s*([0-9.]+)\s*GB', line, re.IGNORECASE)
            if match:
                try:
                    memory_gb = float(match.group(1))
                    if 0 < memory_gb < 1000:
                        memory_values.append(memory_gb)
                        continue
                except (ValueError, IndexError):
                    pass
            
            match = re.search(r'memory\s+(?:usage)?:?\s*([0-9.]+)\s*GB', line, re.IGNORECASE)
            if match:
                try:
                    memory_gb = float(match.group(1))
                    if 0 < memory_gb < 1000:
                        memory_values.append(memory_gb)
                        continue
                except (ValueError, IndexError):
                    pass
    
    return memory_values


def get_parallelism_config(strategy: str, model: str, platform: str) -> Dict[str, Any]:
    return {
        "strategy": strategy or "unknown",
        "tensor_parallel_size": int(os.environ.get("TP", 1)),
        "pipeline_parallel_size": int(os.environ.get("PP", 1)),
        "data_parallel_size": int(os.environ.get("DP", 4)),
        "gradient_accumulation_steps": int(os.environ.get("GRAD_ACCUM", 32)),
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
                 parallel_strategy: str = "unknown", framework: str = None):
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
        
    def on_train_start(self, trainer, pl_module):
        self.train_start_time = time.time()
        self.num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        
        if hasattr(trainer, 'datamodule'):
            self.global_batch_size = getattr(trainer.datamodule, 'global_batch_size', None)
            self.sequence_length = getattr(trainer.datamodule, 'seq_length', 
                                          getattr(trainer.datamodule, 'sequence_length', 2048))
        
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
            print(f"{'='*60}\n")
    
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self.step_start_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
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
            mem_reserved = torch.cuda.memory_reserved() / 1e9
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
        
        try:
            lr = trainer.optimizers[0].param_groups[0]['lr'] if trainer.optimizers else None
            if lr is not None:
                self.learning_rates.append(float(lr))
        except (IndexError, KeyError, AttributeError):
            pass
        
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
        if not trainer.is_global_zero:
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
                }
            
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


class BenchmarkCallbackTran(TrainerCallback):
    def __init__(self, output_dir: str = "./output", platform: str = "auto", model_name: str = None, 
                 parallel_strategy: str = "unknown", framework: str = None):
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
            mem_reserved = torch.cuda.memory_reserved() / 1e9
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
                }
            
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
