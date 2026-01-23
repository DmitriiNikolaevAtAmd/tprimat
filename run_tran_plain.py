#!/usr/bin/env python3
"""
Option 1: PyTorch + HuggingFace Transformers
Most portable and straightforward approach, good for single-node training
"""
import os
import sys

# Force unbuffered output
os.environ['PYTHONUNBUFFERED'] = '1'
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import torch
import random
import numpy as np
import logging
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from torch.utils.data import Dataset, IterableDataset
import json
from utils import BenchmarkCallbackTran

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ],
    force=True
)
logger = logging.getLogger(__name__)


class PretrainingDataset(IterableDataset):
    """Simple iterable dataset for pretraining"""
    def __init__(self, data_path, tokenizer, seq_length=2048, max_steps=10, global_batch_size=64):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.max_steps = max_steps
        self.global_batch_size = global_batch_size
        
    def __iter__(self):
        # Generate synthetic data for benchmarking
        for _ in range(self.max_steps * self.global_batch_size):
            # Create random tokens for benchmarking
            input_ids = torch.randint(0, self.tokenizer.vocab_size, (self.seq_length,))
            yield {
                'input_ids': input_ids,
                'labels': input_ids.clone(),
            }


def train_llama():
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    random.seed(42)
    os.environ['PYTHONHASHSEED'] = '42'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    model_name = "meta-llama/Llama-3.1-8B"
    
    # Load model and tokenizer
    logger.info(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",  # Use Flash Attention 2 if available
        use_cache=False,
        low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()
    logger.info("Enabled gradient checkpointing")
    
    # Create dataset
    dataset = PretrainingDataset(
        data_path="/data/llama_dataset_text_document",
        tokenizer=tokenizer,
        seq_length=2048,
        max_steps=10,
        global_batch_size=64
    )
    
    # Calculate batch size based on number of GPUs
    num_gpus = torch.cuda.device_count()
    global_batch_size = 64
    per_device_batch_size = 1
    gradient_accumulation_steps = global_batch_size // (per_device_batch_size * num_gpus)
    
    logger.info(f"Distributed training config: {num_gpus} GPUs")
    logger.info(f"  Per-device batch size: {per_device_batch_size}")
    logger.info(f"  Gradient accumulation steps: {gradient_accumulation_steps}")
    logger.info(f"  Global batch size: {global_batch_size}")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./output/llama_tran",
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=0.0003,
        weight_decay=0.1,
        adam_beta1=0.9,
        adam_beta2=0.95,
        max_steps=10,
        warmup_steps=1,
        lr_scheduler_type="cosine",
        logging_steps=1,
        save_strategy="no",
        bf16=True,  # Use bfloat16 for training
        bf16_full_eval=False,
        dataloader_num_workers=2,
        ddp_find_unused_parameters=False,
        gradient_checkpointing=True,
        optim="adamw_torch_fused",  # Use fused optimizer for better memory
        remove_unused_columns=False,
        report_to="none",
        # Distributed training settings
        local_rank=int(os.environ.get("LOCAL_RANK", -1)),
        ddp_backend="nccl",
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    
    # Add benchmark callback if available
    try:
        benchmark_callback = BenchmarkCallbackTran(
            output_dir="./output",
            platform="auto",
            model_name="llama",
            parallel_strategy="ddp",
            profiler_config={"enabled": False},
            framework="tran"
        )
        trainer.add_callback(benchmark_callback)
    except Exception as e:
        logger.warning(f"Could not add benchmark callback: {e}")
    
    # Train
    print("Starting training...")
    trainer.train()
    print("Training completed!")


def train_qwen():
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    random.seed(42)
    os.environ['PYTHONHASHSEED'] = '42'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    model_name = "Qwen/Qwen2.5-7B"
    
    # Load model and tokenizer
    logger.info(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",  # Use Flash Attention 2 if available
        use_cache=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()
    
    # Calculate batch size based on number of GPUs
    num_gpus = torch.cuda.device_count()
    global_batch_size = 64
    per_device_batch_size = 1
    gradient_accumulation_steps = global_batch_size // (per_device_batch_size * num_gpus)
    
    logger.info(f"Distributed training config: {num_gpus} GPUs")
    logger.info(f"  Per-device batch size: {per_device_batch_size}")
    logger.info(f"  Gradient accumulation steps: {gradient_accumulation_steps}")
    logger.info(f"  Global batch size: {global_batch_size}")
    
    # Create dataset
    dataset = PretrainingDataset(
        data_path="/data/llama_dataset_text_document",
        tokenizer=tokenizer,
        seq_length=2048,
        max_steps=10,
        global_batch_size=global_batch_size
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./output/qwen_tran",
        gradient_checkpointing=True,  # Enable gradient checkpointing
        optim="adamw_torch_fused",  # Use fused optimizer for better memory
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=0.0003,
        weight_decay=0.1,
        adam_beta1=0.9,
        adam_beta2=0.95,
        max_steps=10,
        warmup_steps=1,
        lr_scheduler_type="cosine",
        logging_steps=1,
        save_strategy="no",
        bf16=True,  # Use bfloat16 for training
        bf16_full_eval=False,
        dataloader_num_workers=2,
        ddp_find_unused_parameters=False,
        gradient_checkpointing=True,
        optim="adamw_torch_fused",  # Use fused optimizer for better memory
        remove_unused_columns=False,
        report_to="none",
        # Distributed training settings
        local_rank=int(os.environ.get("LOCAL_RANK", -1)),
        ddp_backend="nccl",
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    
    # Add benchmark callback if available
    try:
        benchmark_callback = BenchmarkCallbackTran(
            output_dir="./output",
            platform="auto",
            model_name="qwen",
            parallel_strategy="ddp",
            profiler_config={"enabled": False},
            framework="tran"
        )
        trainer.add_callback(benchmark_callback)
    except Exception as e:
        logger.warning(f"Could not add benchmark callback: {e}")
    
    # Train
    print("Starting training...")
    trainer.train()
    print("Training completed!")


def main():
    os.makedirs("./output", exist_ok=True)
    
    if not torch.cuda.is_available():
        logger.error("CUDA is not available!")
        sys.exit(1)
    
    num_gpus = torch.cuda.device_count()
    logger.info(f"CUDA devices available: {num_gpus}")
    
    if num_gpus > 1 and "RANK" not in os.environ:
        logger.warning(f"⚠️  Detected {num_gpus} GPUs but not using distributed training. Only GPU 0 will be used.")
        logger.warning(f"   To use all GPUs: torchrun --nproc_per_node={num_gpus} {sys.argv[0]}")
    
    if len(sys.argv) < 2:
        # No model specified, train all models
        logger.info("No model specified, training all models")
        train_llama()
        train_qwen()
    else:
        model = sys.argv[1]
        if model == "llama":
            train_llama()
        elif model == "qwen":
            train_qwen()
        else:
            logger.error(f"Unknown model: {model}")
            logger.error("Usage: python run_tran_plain.py [model]")
            logger.error("  model: 'llama' or 'qwen' (optional, trains all if omitted)")
            sys.exit(1)


if __name__ == "__main__":
    main()
