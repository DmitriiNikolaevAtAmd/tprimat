#!/usr/bin/env python3
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

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
from torch.utils.data import IterableDataset
from lib.utils import BenchmarkCallbackTran

DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
WORKSPACE_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", str(WORKSPACE_ROOT / "output")))

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


class PretrainingDataset(IterableDataset):
    def __init__(self, data_path, tokenizer, seq_length, max_steps, global_batch_size):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.max_steps = max_steps
        self.global_batch_size = global_batch_size
        
        if not data_path:
            raise ValueError("data_path is required - synthetic data is not allowed")
        
        from lib.mega_dataset import IndexedDataset
        self.indexed_dataset = IndexedDataset(data_path)
        bin_size = self.indexed_dataset.bin_path.stat().st_size
        if bin_size == 0:
            raise ValueError(f"Binary dataset file is empty: {self.indexed_dataset.bin_path}")
        logger.info(f"✓ Loaded indexed dataset from {data_path}")
        logger.info(f"  Dataset contains {len(self.indexed_dataset)} sequences")
        
    def __iter__(self):
        total_samples = self.max_steps * self.global_batch_size
        for i in range(total_samples):
            dataset_idx = i % len(self.indexed_dataset)
            tokens = self.indexed_dataset[dataset_idx]
            
            if len(tokens) < self.seq_length:
                pad_token = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id else 0
                padding = torch.full((self.seq_length - len(tokens),), pad_token, dtype=torch.long)
                input_ids = torch.cat([tokens, padding])
            else:
                input_ids = tokens[:self.seq_length]
            
            yield {
                'input_ids': input_ids,
                'labels': input_ids.clone(),
            }


def train_model(model_name, model_short_name):
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    platform_prefix = "amd"
    num_gpus = torch.cuda.device_count()
    
    logger.info(f"Initializing model: {model_name} (random weights)")
    
    torch_dtype = torch.bfloat16 if PRECISION == "bf16" else torch.float16 if PRECISION == "fp16" else torch.float32
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_config(
        config,
        torch_dtype=torch_dtype,
        attn_implementation="flash_attention_2",
    )
    model.config.use_cache = False
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    model.gradient_checkpointing_enable()
    logger.info("Enabled gradient checkpointing")
    
    dataset_path = str(DATA_DIR / f"allenai-c4-{model_short_name}-mega")
    
    idx_file = dataset_path + ".idx"
    bin_file = dataset_path + ".bin"
    if not os.path.exists(idx_file) or not os.path.exists(bin_file):
        raise FileNotFoundError(
            f"Real data not found at {dataset_path}\n"
            f"  Missing: {idx_file if not os.path.exists(idx_file) else ''} "
            f"{bin_file if not os.path.exists(bin_file) else ''}\n"
            f"  Run data preparation first: python prepare/fetch_deps.py && "
            f"python prepare/clean_data.py && python prepare/encode_data.py"
        )
    
    logger.info(f"Dataset: {dataset_path}")
    
    grad_accum = GRAD_ACCUM // num_gpus if num_gpus > 1 else GRAD_ACCUM
    global_batch_size = MBS * grad_accum * num_gpus
    
    dataset = PretrainingDataset(
        data_path=dataset_path,
        tokenizer=tokenizer,
        seq_length=SEQ_LEN,
        max_steps=TRAIN_ITERS,
        global_batch_size=global_batch_size
    )
    
    logger.info(f"Distributed training config: {num_gpus} GPUs")
    logger.info(f"  Per-device batch size: {MBS}")
    logger.info(f"  Gradient accumulation steps: {grad_accum}")
    logger.info(f"  Global batch size: {global_batch_size}")
    
    use_bf16 = PRECISION == "bf16"
    use_fp16 = PRECISION == "fp16"
    
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / f"{model_short_name}_tran"),
        per_device_train_batch_size=MBS,
        gradient_accumulation_steps=grad_accum,
        learning_rate=LR,
        weight_decay=WEIGHT_DECAY,
        adam_beta1=BETA1,
        adam_beta2=BETA2,
        max_steps=TRAIN_ITERS,
        warmup_steps=WARMUP_STEPS,
        lr_scheduler_type="cosine",
        logging_steps=1,
        save_strategy="no",
        bf16=use_bf16,
        fp16=use_fp16,
        bf16_full_eval=False,
        dataloader_num_workers=0,
        ddp_find_unused_parameters=False,
        gradient_checkpointing=True,
        optim="adamw_bnb_8bit" if HAS_BITSANDBYTES else "adamw_torch_fused",
        remove_unused_columns=False,
        report_to="none",
        local_rank=int(os.environ.get("LOCAL_RANK", -1)),
        ddp_backend="nccl",
    )
    
    if HAS_BITSANDBYTES:
        logger.info("Using 8-bit AdamW optimizer (saves ~75% optimizer memory)")
    else:
        logger.warning("bitsandbytes not available, using fused AdamW (higher memory)")
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    
    try:
        benchmark_callback = BenchmarkCallbackTran(
            output_dir=str(OUTPUT_DIR),
            platform="amd",
            model_name=model_short_name,
            parallel_strategy="ddp",
            framework=f"{platform_prefix}_tran"
        )
        trainer.add_callback(benchmark_callback)
    except Exception as e:
        logger.warning(f"Could not add benchmark callback: {e}")
    
    print("Starting training...")
    trainer.train()
    print("Training completed!")


def train_llama():
    train_model("meta-llama/Llama-3.1-8B", "llama")


def train_qwen():
    train_model("Qwen/Qwen2.5-7B", "qwen")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
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
            logger.error("Usage: python train_amd_tran.py [model]")
            logger.error("  model: 'llama' or 'qwen' (optional, trains all if omitted)")
            sys.exit(1)


if __name__ == "__main__":
    main()
