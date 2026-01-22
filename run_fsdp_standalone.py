#!/usr/bin/env python3
"""
Option 2: PyTorch FSDP (Fully Sharded Data Parallel)
Native PyTorch distributed training with FSDP for memory-efficient training
"""
import os
import sys
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
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.optim import AdamW
import time
from functools import partial


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


def train_model(model_name, model_short_name):
    rank, world_size, local_rank = setup_distributed()
    
    # Set seeds
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    random.seed(42)
    os.environ['PYTHONHASHSEED'] = '42'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    if rank == 0:
        print(f"Loading model: {model_name}")
        print(f"World size: {world_size}, Rank: {rank}, Local rank: {local_rank}")
    
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
        print("Starting training...")
    
    model.train()
    gradient_accumulation_steps = 8  # 64 / 8 GPUs
    
    step = 0
    total_loss = 0.0
    start_time = time.time()
    
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
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            step += 1
            
            if rank == 0:
                avg_loss = total_loss / step
                elapsed = time.time() - start_time
                print(f"Step {step}/{total_steps}, Loss: {avg_loss:.4f}, Time: {elapsed:.2f}s")
            
            if step >= total_steps:
                break
    
    if rank == 0:
        total_time = time.time() - start_time
        print(f"Training completed! Total time: {total_time:.2f}s")
    
    cleanup_distributed()


def train_llama():
    train_model("meta-llama/Llama-3.1-8B", "llama")


def train_qwen():
    train_model("Qwen/Qwen2.5-7B", "qwen")


def main():
    os.makedirs("./output", exist_ok=True)
    
    if not torch.cuda.is_available():
        print("CUDA is not available!")
        sys.exit(1)
    
    if len(sys.argv) > 1:
        model = sys.argv[1]
        if model == "llama":
            train_llama()
        elif model == "qwen":
            train_qwen()
        else:
            print(f"Unknown model: {model}")
            sys.exit(1)
    else:
        import subprocess
        import time
        
        subprocess.run([sys.executable, __file__, "llama"])
        time.sleep(10)
        subprocess.run([sys.executable, __file__, "qwen"])


if __name__ == "__main__":
    main()
