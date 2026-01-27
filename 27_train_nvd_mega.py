#!/usr/bin/env python3
import os
import sys

os.environ['PYTHONUNBUFFERED'] = '1'
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import torch
import random
import numpy as np
import logging
import json
import time
from pathlib import Path
from datetime import datetime

try:
    import bitsandbytes as bnb
    HAS_BITSANDBYTES = True
except (ImportError, RuntimeError):
    HAS_BITSANDBYTES = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ],
    force=True
)
logger = logging.getLogger(__name__)


class PretrainingDataset:
    """Simple dataset class for Megatron training"""
    def __init__(self, tokenizer, seq_length=2048, use_real_data=False, data_path=None):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.indexed_dataset = None
        
        if use_real_data and data_path:
            try:
                from indexed_dataset import IndexedDataset
                self.indexed_dataset = IndexedDataset(data_path)
                logger.info(f"✓ Loaded real indexed dataset from {data_path}")
                logger.info(f"  Dataset contains {len(self.indexed_dataset)} sequences")
                self.real_data_available = True
            except Exception as e:
                logger.warning(f"⚠ Could not load real data: {e}")
                logger.warning(f"  Falling back to synthetic data")
                self.real_data_available = False
        else:
            self.real_data_available = False
        
        self.iteration = 0
    
    def get_batch(self, batch_size, device):
        """Get a batch of data (either real or synthetic)"""
        if self.real_data_available and self.indexed_dataset is not None:
            # Load real data
            batch_tokens = []
            for _ in range(batch_size):
                base_idx = self.iteration
                tokens = None
                last_error = None
                for attempt in range(3):
                    dataset_idx = (base_idx + attempt) % len(self.indexed_dataset)
                    try:
                        tokens = self.indexed_dataset[dataset_idx]
                        self.iteration = base_idx + attempt + 1
                        break
                    except Exception as e:
                        last_error = e
                        if not getattr(self, "_read_error_logged", False):
                            logger.warning(f"⚠ Real data read failed: {e}")
                            logger.warning("  Retrying with next sequence")
                            self._read_error_logged = True
                if tokens is None:
                    raise IOError(f"Real data read failed after retries: {last_error}")
                
                # Pad or truncate to seq_length
                if len(tokens) < self.seq_length:
                    pad_token = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id else 0
                    padding = torch.full((self.seq_length - len(tokens),), pad_token, dtype=torch.long)
                    input_ids = torch.cat([tokens, padding])
                else:
                    input_ids = tokens[:self.seq_length]
                
                batch_tokens.append(input_ids)
            
            return torch.stack(batch_tokens).to(device)
        else:
            # Synthetic data fallback
            return torch.randint(
                0, self.tokenizer.vocab_size,
                (batch_size, self.seq_length),
                device=device
            )


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def train_model(model_name: str, model_config: dict):
    logger.info(f"=" * 80)
    logger.info(f"Starting Mega-LM training for {model_name}")
    logger.info(f"=" * 80)
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend='nccl')
        logger.info(f"Initialized distributed training: rank={rank}, world_size={world_size}")
    else:
        logger.info("Running in single-GPU mode")
    set_seed(42)
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    if torch.cuda.is_available():
        device_props = torch.cuda.get_device_properties(0)
        device_name = torch.cuda.get_device_name(0)
        platform = "nvd"
        software_stack = "nemo"
        software_version = torch.version.cuda if hasattr(torch.version, 'cuda') else "unknown"
        gpu_cores = 16896 if "h100" in device_name.lower() else 6912
        gpu_info = {
            "device_count": world_size,
            "device_name": device_name,
            "total_memory_gb": device_props.total_memory / 1e9,
            "gpu_cores": gpu_cores,
            "pytorch_version": torch.__version__,
            "software_stack": software_stack,
            "software_version": software_version,
        }
    else:
        platform = "cpu"
        gpu_info = {}
    step_times = []
    loss_values = []
    learning_rates = []
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
        import torch.distributed as dist
        
        logger.info(f"Loading tokenizer: {model_config['hf_model']}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_config['hf_model'],
            trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Loading model: {model_config['hf_model']}")
        if world_size == 1:
            logger.info("Single GPU detected - using memory optimization")
            model = AutoModelForCausalLM.from_pretrained(
                model_config['hf_model'],
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                device_map="auto"
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_config['hf_model'],
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            )
            device = torch.device(f'cuda:{local_rank}')
            model = model.to(device)
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            logger.info("Enabled gradient checkpointing")
        if world_size > 1:
            from torch.nn.parallel import DistributedDataParallel as DDP
            device = torch.device(f'cuda:{local_rank}')
            model = DDP(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=False
            )
            logger.info(f"Wrapped model with DDP on device {local_rank}")
        dataset_path = f"/data/tprimat/allenai-c4-100k-{model_name}"
        use_real_data = os.path.exists(dataset_path + ".idx") and os.path.exists(dataset_path + ".bin")
        
        # Create dataset loader
        dataset = PretrainingDataset(
            tokenizer=tokenizer,
            seq_length=model_config['seq_length'],
            use_real_data=use_real_data,
            data_path=dataset_path
        )
        
        seq_length = model_config['seq_length']
        batch_size = model_config['micro_batch_size']
        num_steps = model_config['num_steps']
        if HAS_BITSANDBYTES:
            optimizer = bnb.optim.Adam8bit(
                model.parameters(),
                lr=model_config['learning_rate'],
                betas=(0.9, 0.95),
                eps=1e-8
            )
            logger.info("Using 8-bit Adam optimizer (saves ~75% optimizer memory)")
        else:
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=model_config['learning_rate'],
                betas=(0.9, 0.95),
                eps=1e-8
            )
            logger.warning("bitsandbytes not available, using standard Adam (higher memory)")
        
        # Add LR scheduler with warmup and cosine decay
        from transformers import get_cosine_schedule_with_warmup
        num_warmup_steps = 10
        num_training_steps = model_config['num_steps']
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        
        logger.info(f"Configuration:")
        logger.info(f"  Sequence length: {seq_length}")
        logger.info(f"  Micro batch size: {batch_size}")
        logger.info(f"  Global batch size: {batch_size * world_size * model_config['grad_accum_steps']}")
        logger.info(f"  Gradient accumulation: {model_config['grad_accum_steps']}")
        logger.info(f"  Training steps: {num_steps}")
        logger.info(f"  Learning rate: {model_config['learning_rate']}")
        logger.info(f"  LR scheduler: Cosine with {num_warmup_steps} warmup steps")
        logger.info(f"  Precision: bfloat16")
        model.train()
        logger.info("Starting training...")
        training_start = time.time()
        
        for step in range(num_steps):
            step_start = time.time()
            optimizer.zero_grad()
            
            step_losses = []
            for micro_step in range(model_config['grad_accum_steps']):
                if world_size == 1:
                    device = next(model.parameters()).device
                
                # Get batch from dataset (real or synthetic)
                input_ids = dataset.get_batch(batch_size, device)
                labels = input_ids.clone()
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    outputs = model(input_ids=input_ids, labels=labels)
                    loss = outputs.loss / model_config['grad_accum_steps']
                loss.backward()
                step_losses.append(loss.item() * model_config['grad_accum_steps'])
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            step_time = time.time() - step_start
            avg_loss = sum(step_losses) / len(step_losses)
            tokens_per_step = batch_size * seq_length * model_config['grad_accum_steps'] * world_size
            throughput = tokens_per_step / step_time
            step_times.append(step_time)
            loss_values.append(avg_loss)
            current_lr = scheduler.get_last_lr()[0] if scheduler is not None else optimizer.param_groups[0]['lr']
            learning_rates.append(current_lr)
            if rank == 0:
                logger.info(
                    f"Step {step + 1}/{num_steps} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"Time: {step_time:.3f}s | "
                    f"Throughput: {throughput:.0f} tokens/s"
                )
            if rank == 0 and (step + 1) % 5 == 0:
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated(device) / 1e9
                    reserved = torch.cuda.memory_reserved(device) / 1e9
                    logger.info(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
        training_time = time.time() - training_start
        if rank == 0 and len(step_times) > 10:
            warmup_steps = min(10, len(step_times))
            step_times_no_warmup = step_times[warmup_steps:]
            if not step_times_no_warmup:
                step_times_no_warmup = step_times
            avg_step_time = sum(step_times_no_warmup) / len(step_times_no_warmup)
            steps_per_second = len(step_times_no_warmup) / sum(step_times_no_warmup)
            global_batch_size = batch_size * world_size * model_config['grad_accum_steps']
            tokens_per_step = global_batch_size * seq_length
            tokens_per_second = tokens_per_step / avg_step_time
            tokens_per_second_per_gpu = tokens_per_second / world_size if world_size else None
            results = {
                "platform": platform,
                "gpu_info": gpu_info,
                "timestamp": datetime.now().isoformat(),
                "training_config": {
                    "max_steps": num_steps,
                    "global_batch_size": global_batch_size,
                    "micro_batch_size": batch_size,
                    "sequence_length": seq_length,
                    "num_gpus": world_size,
                    "parallel_strategy": "ddp",
                    "gradient_accumulation_steps": model_config['grad_accum_steps'],
                },
                "performance_metrics": {
                    "total_steps": len(step_times),
                    "total_time_seconds": training_time,
                    "avg_step_time_seconds": avg_step_time,
                    "min_step_time_seconds": min(step_times_no_warmup),
                    "max_step_time_seconds": max(step_times_no_warmup),
                    "steps_per_second": steps_per_second,
                    "tokens_per_second": tokens_per_second,
                    "tokens_per_second_per_gpu": tokens_per_second_per_gpu,
                    "throughput_per_gpu_core": steps_per_second / gpu_info["gpu_cores"] if gpu_info.get("gpu_cores", 0) > 0 else 0,
                },
                "step_times": step_times,
                "loss_values": loss_values,
                "learning_rates": learning_rates,
            }
            
            logger.info(f"=" * 80)
            logger.info(f"Training completed for {model_name}")
            logger.info(f"Total time: {training_time:.2f}s")
            logger.info(f"Average step time: {avg_step_time:.3f}s")
            logger.info(f"Throughput: {tokens_per_second:,.0f} tokens/sec")
            logger.info(f"Per-GPU Throughput: {tokens_per_second_per_gpu:,.0f} tokens/sec/GPU")
            logger.info(f"=" * 80)
            output_dir = Path("output")
            output_dir.mkdir(exist_ok=True)
            output_file = output_dir / f"train_{platform}_mega_{model_name}.json"
            from utils import round_floats
            results_rounded = round_floats(results, precision=5)
            
            with open(output_file, 'w') as f:
                json.dump(results_rounded, f, indent=2)
            
            logger.info(f"Results saved to {output_file}")
    
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        if rank == 0 and len(step_times) > 0:
            # Save partial results on error
            results = {
                "platform": platform,
                "gpu_info": gpu_info,
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "partial_results": {
                    "steps_completed": len(step_times),
                    "step_times": step_times,
                    "loss_values": loss_values,
                },
            }
            output_dir = Path("output")
            output_dir.mkdir(exist_ok=True)
            output_file = output_dir / f"train_nvd_mega_{model_name}.json"
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
        raise
    
    finally:
        if world_size > 1:
            torch.distributed.destroy_process_group()


def train_llama():
    config = {
        'hf_model': 'meta-llama/Llama-3.1-8B',
        'seq_length': 2048,
        'micro_batch_size': 1,
        'grad_accum_steps': 8,
        'num_steps': 50,
        'learning_rate': 3e-4,
        'tensor_parallel': 1,
        'pipeline_parallel': 1,
    }
    train_model('llama', config)


def train_qwen():
    config = {
        'hf_model': 'Qwen/Qwen2.5-7B',
        'seq_length': 2048,
        'micro_batch_size': 1,
        'grad_accum_steps': 8,
        'num_steps': 50,
        'learning_rate': 3e-4,
        'tensor_parallel': 1,
        'pipeline_parallel': 1,
    }
    train_model('qwen', config)


def main():
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    os.environ['PYTHONHASHSEED'] = '42'
    os.environ['HSA_NO_SCRATCH_RECLAIM'] = '1'
    os.environ['HSA_ENABLE_SDMA'] = '1'
    os.environ['HSA_FORCE_FINE_GRAIN_PCIE'] = '1'
    os.environ['RCCL_DEBUG'] = 'INFO'
    os.environ['NCCL_DEBUG'] = 'INFO'
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    
    logger.info("Environment configured for GPU training")
    logger.info(f"Output directory: {output_dir}")
    
    if len(sys.argv) < 2:
        logger.info("No model specified, training all models")
        train_llama()
        train_qwen()
    else:
        model = sys.argv[1]
        logger.info(f"Training model: {model}")
        if model == "llama":
            train_llama()
        elif model == "qwen":
            train_qwen()
        else:
            logger.error(f"Unknown model: {model}")
            logger.error("\nUsage:")
            logger.error("  python train_nvd_mega.py              # Train both models")
            logger.error("  python train_nvd_mega.py llama        # Train only Llama")
            logger.error("  python train_nvd_mega.py qwen         # Train only Qwen")
            sys.exit(1)


if __name__ == "__main__":
    main()
