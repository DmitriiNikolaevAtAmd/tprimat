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
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    AutoConfig,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from torch.utils.data import Dataset, IterableDataset
import json
from utils import BenchmarkCallbackTran

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


class PretrainingDataset(IterableDataset):
    def __init__(self, data_path, tokenizer, seq_length=2048, max_steps=50, global_batch_size=64, use_real_data=False):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.max_steps = max_steps
        self.global_batch_size = global_batch_size
        self.use_real_data = use_real_data
        self.indexed_dataset = None
        self._real_data_failed = False
        
        if self.use_real_data:
            try:
                from indexed_dataset import IndexedDataset
                self.indexed_dataset = IndexedDataset(data_path)
                bin_size = self.indexed_dataset.bin_path.stat().st_size
                if bin_size == 0:
                    raise ValueError(f"Binary dataset file is empty: {self.indexed_dataset.bin_path}")
                logger.info(f"✓ Loaded real indexed dataset from {data_path}")
                logger.info(f"  Dataset contains {len(self.indexed_dataset)} sequences")
                self.real_data_available = True
            except Exception as e:
                raise RuntimeError(f"Real data required but could not be loaded: {e}") from e
        else:
            self.real_data_available = False
        
    def __iter__(self):
        total_samples = self.max_steps * self.global_batch_size
        for i in range(total_samples):
            if self.real_data_available and self.indexed_dataset is not None:
                # Load real data and pad/truncate to seq_length
                dataset_idx = i % len(self.indexed_dataset)
                tokens = None
                last_error = None
                for attempt in range(3):
                    try:
                        tokens = self.indexed_dataset[(dataset_idx + attempt) % len(self.indexed_dataset)]
                        break
                    except Exception as e:
                        last_error = e
                        if not self._real_data_failed:
                            logger.warning(f"⚠ Real data read failed: {e}")
                            logger.warning("  Retrying with next sequence")
                            self._real_data_failed = True
                if tokens is None:
                    raise IOError(f"Real data read failed after retries: {last_error}")
                
                # Pad or truncate to seq_length
                if len(tokens) < self.seq_length:
                    pad_token = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id else 0
                    padding = torch.full((self.seq_length - len(tokens),), pad_token, dtype=torch.long)
                    input_ids = torch.cat([tokens, padding])
                else:
                    input_ids = tokens[:self.seq_length]
            else:
                raise RuntimeError("Real data required for Transformers training")
            
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
    
    platform_prefix = "nvd"
    
    model_name = "meta-llama/Llama-3.1-8B"
    logger.info(f"Initializing model: {model_name} (random weights)")
    # Load config and create model with random weights (no pretrained download)
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_config(
        config,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()
    logger.info("Enabled gradient checkpointing")
    dataset_path = "/data/tprimat/allenai-c4-100k-llama"
    use_real_data = os.path.exists(dataset_path + ".idx") and os.path.exists(dataset_path + ".bin")
    
    if use_real_data:
        logger.info(f"Real data found at {dataset_path}")
    else:
        raise RuntimeError(f"Real data required but not found at {dataset_path}.idx/.bin")
    
    # Create dataset
    dataset = PretrainingDataset(
        data_path=dataset_path,
        tokenizer=tokenizer,
        seq_length=2048,
        max_steps=50,
        global_batch_size=64,
        use_real_data=use_real_data
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
    training_args = TrainingArguments(
        output_dir="./output/llama_tran",
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=0.0003,
        weight_decay=0.1,
        adam_beta1=0.9,
        adam_beta2=0.95,
        max_steps=50,
        warmup_steps=10,
        lr_scheduler_type="cosine",
        logging_steps=1,
        save_strategy="no",
        bf16=True,  # Use bfloat16 for training
        bf16_full_eval=False,
        dataloader_num_workers=0 if use_real_data else 2,
        ddp_find_unused_parameters=False,
        gradient_checkpointing=True,
        optim="adamw_bnb_8bit" if HAS_BITSANDBYTES else "adamw_torch_fused",  # Use 8-bit optimizer to save memory
        remove_unused_columns=False,
        report_to="none",
        # Distributed training settings
        local_rank=int(os.environ.get("LOCAL_RANK", -1)),
        ddp_backend="nccl",
    )
    
    if HAS_BITSANDBYTES:
        logger.info("Using 8-bit AdamW optimizer (saves ~75% optimizer memory)")
    else:
        logger.warning("bitsandbytes not available, using fused AdamW (higher memory)")
    
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
            platform="nvd",
            model_name="llama",
            parallel_strategy="ddp",
            framework=f"{platform_prefix}_tran"
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
    
    platform_prefix = "nvd"
    
    model_name = "Qwen/Qwen2.5-7B"
    logger.info(f"Initializing model: {model_name} (random weights)")
    # Load config and create model with random weights (no pretrained download)
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_config(
        config,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()
    logger.info("Enabled gradient checkpointing")
    num_gpus = torch.cuda.device_count()
    global_batch_size = 64
    per_device_batch_size = 1
    gradient_accumulation_steps = global_batch_size // (per_device_batch_size * num_gpus)
    
    logger.info(f"Distributed training config: {num_gpus} GPUs")
    logger.info(f"  Per-device batch size: {per_device_batch_size}")
    logger.info(f"  Gradient accumulation steps: {gradient_accumulation_steps}")
    logger.info(f"  Global batch size: {global_batch_size}")
    dataset_path = "/data/tprimat/allenai-c4-100k-qwen"
    use_real_data = os.path.exists(dataset_path + ".idx") and os.path.exists(dataset_path + ".bin")
    
    if use_real_data:
        logger.info(f"Real data found at {dataset_path}")
    else:
        raise RuntimeError(f"Real data required but not found at {dataset_path}.idx/.bin")
    
    # Create dataset
    dataset = PretrainingDataset(
        data_path=dataset_path,
        tokenizer=tokenizer,
        seq_length=2048,
        max_steps=50,
        global_batch_size=global_batch_size,
        use_real_data=use_real_data
    )
    training_args = TrainingArguments(
        output_dir="./output/qwen_tran",
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=0.0003,
        weight_decay=0.1,
        adam_beta1=0.9,
        adam_beta2=0.95,
        max_steps=50,
        warmup_steps=10,
        lr_scheduler_type="cosine",
        logging_steps=1,
        save_strategy="no",
        bf16=True,  # Use bfloat16 for training
        bf16_full_eval=False,
        dataloader_num_workers=0 if use_real_data else 2,
        ddp_find_unused_parameters=False,
        gradient_checkpointing=True,
        optim="adamw_bnb_8bit" if HAS_BITSANDBYTES else "adamw_torch_fused",  # Use 8-bit optimizer to save memory
        remove_unused_columns=False,
        report_to="none",
        # Distributed training settings
        local_rank=int(os.environ.get("LOCAL_RANK", -1)),
        ddp_backend="nccl",
    )
    
    if HAS_BITSANDBYTES:
        logger.info("Using 8-bit AdamW optimizer (saves ~75% optimizer memory)")
    else:
        logger.warning("bitsandbytes not available, using fused AdamW (higher memory)")
    
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
            platform="nvd",
            model_name="qwen",
            parallel_strategy="ddp",
            framework=f"{platform_prefix}_tran"
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
        logger.warning(f"Detected {num_gpus} GPUs but not using distributed training. Only GPU 0 will be used.")
        logger.warning(f"To use all GPUs: torchrun --nproc_per_node={num_gpus} {sys.argv[0]}")
    
    if len(sys.argv) < 2:
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
            logger.error("Usage: python train_all_tran.py [model]")
            logger.error("  model: 'llama' or 'qwen' (optional, trains all if omitted)")
            sys.exit(1)


if __name__ == "__main__":
    main()
