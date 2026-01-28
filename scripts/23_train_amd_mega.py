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

DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", str(Path(__file__).parent.parent / "output")))

SEED = int(os.environ.get("SEED", 42))
MBS = int(os.environ.get("MBS", 1))
GBS = int(os.environ.get("GBS", 128))
SEQ_LEN = int(os.environ.get("SEQ_LEN", 2048))
LR = float(os.environ.get("LR", 3e-4))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", 0.1))
BETA1 = float(os.environ.get("BETA1", 0.9))
BETA2 = float(os.environ.get("BETA2", 0.95))
PRECISION = os.environ.get("PRECISION", "bf16")
WARMUP_STEPS = int(os.environ.get("WARMUP_STEPS", 50))
TRAIN_ITERS = int(os.environ.get("TRAIN_ITERS", 10))
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", 32))
TP = int(os.environ.get("TP", 1))
PP = int(os.environ.get("PP", 1))
DP = int(os.environ.get("DP", 4))

try:
    import bitsandbytes as bnb
    HAS_BITSANDBYTES = True
except (ImportError, RuntimeError):
    HAS_BITSANDBYTES = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True
)
logger = logging.getLogger(__name__)


class PretrainingDataset:
    def __init__(self, tokenizer, seq_length, data_path):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        
        if not data_path:
            raise ValueError("data_path is required - synthetic data is not allowed")
        
        from mega_dataset import IndexedDataset
        self.indexed_dataset = IndexedDataset(data_path)
        logger.info(f"✓ Loaded indexed dataset from {data_path}")
        logger.info(f"  Dataset contains {len(self.indexed_dataset)} sequences")
        self.iteration = 0
    
    def get_batch(self, batch_size, device):
        batch_tokens = []
        for _ in range(batch_size):
            dataset_idx = self.iteration % len(self.indexed_dataset)
            tokens = self.indexed_dataset[dataset_idx]
            self.iteration += 1
            
            if len(tokens) < self.seq_length:
                pad_token = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id else 0
                padding = torch.full((self.seq_length - len(tokens),), pad_token, dtype=torch.long)
                input_ids = torch.cat([tokens, padding])
            else:
                input_ids = tokens[:self.seq_length]
            
            batch_tokens.append(input_ids)
        
        return torch.stack(batch_tokens).to(device)


def set_seed(seed):
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
    
    set_seed(SEED)
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    grad_accum = model_config['grad_accum_steps']
    global_batch_size = MBS * grad_accum * world_size
    
    if torch.cuda.is_available():
        device_props = torch.cuda.get_device_properties(0)
        device_name = torch.cuda.get_device_name(0)
        platform = "amd"
        software_stack = "megatron"
        software_version = torch.version.hip if hasattr(torch.version, 'hip') else "unknown"
        
        if "mi300" in device_name.lower():
            gpu_cores = 14592
        else:
            gpu_cores = 6912
        
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
        
        logger.info(f"Loading tokenizer: {model_config['hf_model']}")
        tokenizer = AutoTokenizer.from_pretrained(model_config['hf_model'], trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        logger.info(f"Initializing model: {model_config['hf_model']} (random weights)")
        torch_dtype = torch.bfloat16 if PRECISION == "bf16" else torch.float16 if PRECISION == "fp16" else torch.float32
        config = AutoConfig.from_pretrained(model_config['hf_model'], trust_remote_code=True)
        model = AutoModelForCausalLM.from_config(config, torch_dtype=torch_dtype)
        model.config.use_cache = False
        
        if world_size == 1:
            logger.info("Single GPU detected")
            model = model.to('cuda')
        else:
            device = torch.device(f'cuda:{local_rank}')
            model = model.to(device)
        
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            logger.info("Enabled gradient checkpointing")
        
        if world_size > 1:
            from torch.nn.parallel import DistributedDataParallel as DDP
            device = torch.device(f'cuda:{local_rank}')
            model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
            logger.info(f"Wrapped model with DDP on device {local_rank}")
        
        dataset_path = str(DATA_DIR / f"allenai-c4-{model_name}-mega")
        
        idx_file = dataset_path + ".idx"
        bin_file = dataset_path + ".bin"
        if not os.path.exists(idx_file) or not os.path.exists(bin_file):
            raise FileNotFoundError(
                f"Real data not found at {dataset_path}\n"
                f"  Missing: {idx_file if not os.path.exists(idx_file) else ''} "
                f"{bin_file if not os.path.exists(bin_file) else ''}\n"
                f"  Run data preparation first: python scripts/01_fetch_deps.py && "
                f"python scripts/02_clean_data.py && python scripts/03_encode_data.py"
            )
        
        logger.info(f"Dataset: {dataset_path}")
        
        dataset = PretrainingDataset(tokenizer=tokenizer, seq_length=SEQ_LEN, data_path=dataset_path)
        
        batch_size = MBS
        num_steps = TRAIN_ITERS
        
        if HAS_BITSANDBYTES:
            optimizer = bnb.optim.Adam8bit(model.parameters(), lr=LR, betas=(BETA1, BETA2), eps=1e-8)
            logger.info("Using 8-bit Adam optimizer (saves ~75% optimizer memory)")
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(BETA1, BETA2), eps=1e-8)
            logger.warning("bitsandbytes not available, using standard Adam (higher memory)")
        
        from transformers import get_cosine_schedule_with_warmup
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=num_steps)
        
        logger.info(f"Configuration:")
        logger.info(f"  Sequence length: {SEQ_LEN}")
        logger.info(f"  Micro batch size: {batch_size}")
        logger.info(f"  Global batch size: {global_batch_size}")
        logger.info(f"  Gradient accumulation: {grad_accum}")
        logger.info(f"  Training steps: {num_steps}")
        logger.info(f"  Learning rate: {LR}")
        logger.info(f"  LR scheduler: Cosine with {WARMUP_STEPS} warmup steps")
        logger.info(f"  Precision: {PRECISION}")
        
        model.train()
        logger.info("Starting training...")
        training_start = time.time()
        
        for step in range(num_steps):
            step_start = time.time()
            optimizer.zero_grad()
            
            step_losses = []
            for micro_step in range(grad_accum):
                if world_size == 1:
                    device = next(model.parameters()).device
                
                input_ids = dataset.get_batch(batch_size, device)
                labels = input_ids.clone()
                
                with torch.amp.autocast('cuda', dtype=torch_dtype):
                    outputs = model(input_ids=input_ids, labels=labels)
                    loss = outputs.loss / grad_accum
                
                loss.backward()
                step_losses.append(loss.item() * grad_accum)
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            step_time = time.time() - step_start
            avg_loss = sum(step_losses) / len(step_losses)
            tokens_per_step = batch_size * SEQ_LEN * grad_accum * world_size
            throughput = tokens_per_step / step_time
            
            step_times.append(step_time)
            loss_values.append(avg_loss)
            current_lr = scheduler.get_last_lr()[0] if scheduler is not None else optimizer.param_groups[0]['lr']
            learning_rates.append(current_lr)
            
            if rank == 0:
                logger.info(f"Step {step + 1}/{num_steps} | Loss: {avg_loss:.4f} | Time: {step_time:.3f}s | Throughput: {throughput:.0f} tokens/s")
            
            if rank == 0 and (step + 1) % 5 == 0:
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated(device) / 1e9
                    reserved = torch.cuda.memory_reserved(device) / 1e9
                    logger.info(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
        
        training_time = time.time() - training_start
        
        if rank == 0 and len(step_times) > 10:
            warmup_skip = min(10, len(step_times))
            step_times_no_warmup = step_times[warmup_skip:]
            if not step_times_no_warmup:
                step_times_no_warmup = step_times
            
            avg_step_time = sum(step_times_no_warmup) / len(step_times_no_warmup)
            steps_per_second = len(step_times_no_warmup) / sum(step_times_no_warmup)
            tokens_per_step = global_batch_size * SEQ_LEN
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
                    "sequence_length": SEQ_LEN,
                    "num_gpus": world_size,
                    "parallel_strategy": "ddp",
                    "gradient_accumulation_steps": grad_accum,
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
            
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            output_file = OUTPUT_DIR / f"train_{platform}_mega_{model_name}.json"
            from utils import round_floats
            results_rounded = round_floats(results, precision=5)
            
            with open(output_file, 'w') as f:
                json.dump(results_rounded, f, indent=2)
            
            logger.info(f"Results saved to {output_file}")
    
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        if rank == 0 and len(step_times) > 0:
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
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            output_file = OUTPUT_DIR / f"train_amd_mega_{model_name}.json"
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
        raise
    
    finally:
        if world_size > 1:
            torch.distributed.destroy_process_group()


def train_llama():
    config = {
        'hf_model': 'meta-llama/Llama-3.1-8B',
        'grad_accum_steps': GRAD_ACCUM,
        'tensor_parallel': TP,
        'pipeline_parallel': PP,
    }
    train_model('llama', config)


def train_qwen():
    config = {
        'hf_model': 'Qwen/Qwen2.5-7B',
        'grad_accum_steps': GRAD_ACCUM,
        'tensor_parallel': TP,
        'pipeline_parallel': PP,
    }
    train_model('qwen', config)


def main():
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    os.environ['PYTHONHASHSEED'] = str(SEED)
    os.environ['HSA_NO_SCRATCH_RECLAIM'] = '1'
    os.environ['HSA_ENABLE_SDMA'] = '1'
    os.environ['HSA_FORCE_FINE_GRAIN_PCIE'] = '1'
    os.environ['RCCL_DEBUG'] = 'INFO'
    os.environ['NCCL_DEBUG'] = 'INFO'
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    logger.info("Environment configured for GPU training")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    
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
            logger.error("  python train_amd_mega.py              # Train both models")
            logger.error("  python train_amd_mega.py llama        # Train only Llama")
            logger.error("  python train_amd_mega.py qwen         # Train only Qwen")
            sys.exit(1)


if __name__ == "__main__":
    main()
