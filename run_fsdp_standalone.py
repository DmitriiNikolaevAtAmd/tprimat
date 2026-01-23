#!/usr/bin/env python3
"""
Option 2: PyTorch FSDP (Fully Sharded Data Parallel)
Native PyTorch distributed training with FSDP for memory-efficient training
"""
import os
import sys

# Force unbuffered output
os.environ['PYTHONUNBUFFERED'] = '1'
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
)
import random
import numpy as np
import logging
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.optim import AdamW
import time
from functools import partial
import json
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [Rank %(rank)s] - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ],
    force=True
)


class PretrainingDataset(Dataset):
    """Simple dataset for pretraining benchmarking"""
    def __init__(self, tokenizer, seq_length=2048, num_samples=640):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.num_samples = num_samples
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate synthetic data for benchmarking
        input_ids = torch.randint(0, self.tokenizer.vocab_size, (self.seq_length,))
        return {
            'input_ids': input_ids,
            'labels': input_ids.clone(),
        }


def setup_distributed():
    """Initialize distributed training"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        # Single GPU or torchrun not used
        rank = 0
        world_size = 1
        local_rank = 0
        
    torch.cuda.set_device(local_rank)
    
    if world_size > 1:
        dist.init_process_group(backend='nccl')
    
    return rank, world_size, local_rank


def cleanup_distributed():
    """Clean up distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()


def save_fsdp_benchmark_results(model_short_name, step_times, total_time, num_gpus, 
                                 global_batch_size, sequence_length, total_steps, output_dir):
    """Save FSDP benchmark results in the same format as other frameworks"""
    
    # Detect platform
    if torch.cuda.is_available():
        is_rocm = hasattr(torch.version, 'hip') and torch.version.hip is not None
        platform = "amd" if is_rocm else "nvd"
        device_name = torch.cuda.get_device_name(0)
        device_props = torch.cuda.get_device_properties(0)
        total_memory_gb = device_props.total_memory / 1e9
        software_stack = "primus" if is_rocm else "nemo"
        software_version = torch.version.hip if is_rocm else torch.version.cuda
    else:
        platform = "cpu"
        device_name = "CPU"
        total_memory_gb = 0
        software_stack = "cpu"
        software_version = "N/A"
    
    # Calculate metrics
    if len(step_times) > 0:
        avg_step_time = sum(step_times) / len(step_times)
        steps_per_second = len(step_times) / sum(step_times)
        tokens_per_step = global_batch_size * sequence_length
        tokens_per_second = tokens_per_step / avg_step_time
        tokens_per_second_per_gpu = tokens_per_second / num_gpus
    else:
        avg_step_time = 0
        steps_per_second = 0
        tokens_per_second = 0
        tokens_per_second_per_gpu = 0
    
    results = {
        "platform": platform,
        "gpu_info": {
            "device_count": num_gpus,
            "device_name": device_name,
            "total_memory_gb": total_memory_gb,
            "pytorch_version": torch.__version__,
            "software_stack": software_stack,
            "software_version": software_version,
        },
        "timestamp": datetime.now().isoformat(),
        "training_config": {
            "max_steps": total_steps,
            "global_batch_size": global_batch_size,
            "sequence_length": sequence_length,
            "num_gpus": num_gpus,
            "parallel_strategy": "fsdp",
        },
        "performance_metrics": {
            "total_steps": len(step_times) + 1,  # +1 for warmup step
            "total_time_seconds": total_time,
            "avg_step_time_seconds": avg_step_time,
            "min_step_time_seconds": min(step_times) if step_times else 0,
            "max_step_time_seconds": max(step_times) if step_times else 0,
            "steps_per_second": steps_per_second,
            "tokens_per_second": tokens_per_second,
            "tokens_per_second_per_gpu": tokens_per_second_per_gpu,
        },
    }
    
    # Save to file with format: train_fsdp_<model>.json
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    filename = f"train_fsdp_{model_short_name}.json"
    filepath = output_path / filename
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Benchmark results saved to: {filepath}")
    logger.info(f"Avg step time: {avg_step_time:.3f}s")
    logger.info(f"Throughput: {tokens_per_second:,.0f} tokens/sec ({tokens_per_second_per_gpu:,.0f} tokens/sec/GPU)")


def train_model(model_name, model_short_name):
    rank, world_size, local_rank = setup_distributed()
    
    # Setup logger with rank information
    logger = logging.getLogger(__name__)
    logger = logging.LoggerAdapter(logger, {'rank': rank})
    
    # Set seeds
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    random.seed(42)
    os.environ['PYTHONHASHSEED'] = '42'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    if rank == 0:
        logger.info(f"Loading model: {model_name}")
        logger.info(f"World size: {world_size}, Rank: {rank}, Local rank: {local_rank}")
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        use_cache=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()
    
    # Setup FSDP mixed precision
    bfloat16_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )
    
    # Get the transformer layer class for auto wrap
    # This will vary by model architecture
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer
    from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
    
    transformer_layer_cls = {
        LlamaDecoderLayer,
        Qwen2DecoderLayer,
    }
    
    auto_wrap_policy = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls=transformer_layer_cls,
    )
    
    # Wrap model with FSDP
    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=bfloat16_policy,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=torch.cuda.current_device(),
        limit_all_gathers=True,
    )
    
    # Create dataset and dataloader
    dataset = PretrainingDataset(
        tokenizer=tokenizer,
        seq_length=2048,
        num_samples=640  # 10 steps * 64 global batch size
    )
    
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=42
    ) if world_size > 1 else None
    
    dataloader = DataLoader(
        dataset,
        batch_size=1,  # Micro batch size
        sampler=sampler,
        num_workers=2,
        pin_memory=True,
    )
    
    # Setup optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=0.0003,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )
    
    # Setup learning rate scheduler
    total_steps = 10
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=1,
        num_training_steps=total_steps,
    )
    
    # Training loop
    if rank == 0:
        logger.info("Starting training...")
    
    model.train()
    gradient_accumulation_steps = 8  # 64 / 8 GPUs
    
    step = 0
    total_loss = 0.0
    start_time = time.time()
    step_times_no_warmup = []
    first_step = True
    step_start_time = time.time()
    
    for batch_idx, batch in enumerate(dataloader):
        if step >= total_steps:
            break
            
        input_ids = batch['input_ids'].to(local_rank)
        labels = batch['labels'].to(local_rank)
        
        # Forward pass
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss / gradient_accumulation_steps
        
        # Backward pass
        loss.backward()
        
        total_loss += loss.item() * gradient_accumulation_steps
        
        # Optimizer step
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            step_end_time = time.time()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            step += 1
            
            # Track step times (skip first step for warmup)
            if first_step:
                first_step = False
            else:
                step_time = step_end_time - step_start_time
                step_times_no_warmup.append(step_time)
            
            step_start_time = time.time()
            
            if rank == 0:
                avg_loss = total_loss / step
                elapsed = time.time() - start_time
                logger.info(f"Step {step}/{total_steps}, Loss: {avg_loss:.4f}, Time: {elapsed:.2f}s")
            
            if step >= total_steps:
                break
    
    if rank == 0:
        total_time = time.time() - start_time
        logger.info(f"Training completed! Total time: {total_time:.2f}s")
        
        # Save benchmark results
        save_fsdp_benchmark_results(
            model_short_name=model_short_name,
            step_times=step_times_no_warmup if len(step_times_no_warmup) > 0 else [0],
            total_time=total_time,
            num_gpus=world_size,
            global_batch_size=64,
            sequence_length=2048,
            total_steps=total_steps,
            output_dir="./output"
        )
    
    cleanup_distributed()


def train_llama():
    train_model("meta-llama/Llama-3.1-8B", "llama")


def train_qwen():
    train_model("Qwen/Qwen2.5-7B", "qwen")


def main():
    logger = logging.getLogger(__name__)
    os.makedirs("./output", exist_ok=True)
    
    if not torch.cuda.is_available():
        logger.error("CUDA is not available!")
        sys.exit(1)
    
    logger.info(f"CUDA devices available: {torch.cuda.device_count()}")
    
    if len(sys.argv) < 2:
        logger.error("Usage: python run_fsdp_standalone.py <model>")
        logger.error("  model: 'llama' or 'qwen'")
        sys.exit(1)
    
    model = sys.argv[1]
    if model == "llama":
        train_llama()
    elif model == "qwen":
        train_qwen()
    else:
        logger.error(f"Unknown model: {model}")
        sys.exit(1)


if __name__ == "__main__":
    main()
