#!/usr/bin/env python3
"""
Option 3: DeepSpeed ZeRO
Microsoft's DeepSpeed for memory-efficient large model training with ZeRO optimization
"""
import os
import sys
import torch
import random
import numpy as np
import deepspeed
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)
from torch.utils.data import Dataset, DataLoader
import json
import time


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


def get_deepspeed_config():
    """Get DeepSpeed configuration matching NeMo settings"""
    return {
        "train_batch_size": 64,  # Global batch size
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": 8,  # 64 / 8 GPUs
        "steps_per_print": 1,
        "gradient_clipping": 1.0,
        "prescale_gradients": False,
        "bf16": {
            "enabled": True
        },
        "zero_optimization": {
            "stage": 2,  # ZeRO stage 2: optimizer state + gradient partitioning
            "allgather_partitions": True,
            "allgather_bucket_size": 5e8,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
            "overlap_comm": True,
            "contiguous_gradients": True,
        },
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": 0.0003,
                "betas": [0.9, 0.95],
                "eps": 1e-8,
                "weight_decay": 0.1
            }
        },
        "scheduler": {
            "type": "WarmupCosineLR",
            "params": {
                "total_num_steps": 10,
                "warmup_min_lr": 0.00003,
                "warmup_max_lr": 0.0003,
                "warmup_num_steps": 1
            }
        },
        "wall_clock_breakdown": False,
        "dump_state": False
    }


def train_model(model_name, model_short_name):
    # Set seeds
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    random.seed(42)
    os.environ['PYTHONHASHSEED'] = '42'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # Initialize DeepSpeed distributed backend
    deepspeed.init_distributed()
    
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
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
    
    # Create dataset and dataloader
    dataset = PretrainingDataset(
        tokenizer=tokenizer,
        seq_length=2048,
        num_samples=640  # 10 steps * 64 global batch size
    )
    
    # Get DeepSpeed config
    ds_config = get_deepspeed_config()
    
    # Initialize DeepSpeed
    model_engine, optimizer, dataloader, lr_scheduler = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        training_data=dataset,
        config=ds_config,
    )
    
    # Training loop
    if rank == 0:
        print("Starting training...")
    
    model_engine.train()
    total_steps = 10
    step = 0
    total_loss = 0.0
    start_time = time.time()
    
    for batch_idx, batch in enumerate(dataloader):
        if step >= total_steps:
            break
        
        input_ids = batch['input_ids'].to(model_engine.device)
        labels = batch['labels'].to(model_engine.device)
        
        # Forward pass
        outputs = model_engine(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        
        # Backward pass
        model_engine.backward(loss)
        
        # Optimizer step
        model_engine.step()
        
        total_loss += loss.item()
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
        
        # Note: For Deep, you typically need to use deepspeed launcher
        # instead of python directly:
        # deepspeed --num_gpus=8 run_deep_plain.py llama
        subprocess.run([sys.executable, __file__, "llama"])
        time.sleep(10)
        subprocess.run([sys.executable, __file__, "qwen"])


if __name__ == "__main__":
    main()
