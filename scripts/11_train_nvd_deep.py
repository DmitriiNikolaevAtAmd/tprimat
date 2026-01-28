#!/usr/bin/env python3
import os
import sys
from pathlib import Path
import torch
import random
import math
import numpy as np
import deepspeed
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
)
from torch.utils.data import Dataset, DataLoader
import json
import time

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


class PretrainingDataset(Dataset):
    def __init__(self, tokenizer, seq_length, num_samples, data_path):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.num_samples = num_samples
        self.data_path = data_path
        
        if not data_path:
            raise ValueError("data_path is required - synthetic data is not allowed")
        
        from indexed_dataset import IndexedDataset
        self.indexed_dataset = IndexedDataset(data_path)
        print(f"✓ Loaded indexed dataset from {data_path}")
        print(f"  Dataset contains {len(self.indexed_dataset)} sequences")
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        dataset_idx = idx % len(self.indexed_dataset)
        tokens = self.indexed_dataset[dataset_idx]
        
        if len(tokens) < self.seq_length:
            pad_token = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id else 0
            padding = torch.full((self.seq_length - len(tokens),), pad_token, dtype=torch.long)
            input_ids = torch.cat([tokens, padding])
        else:
            input_ids = tokens[:self.seq_length]
        
        return {
            'input_ids': input_ids,
            'labels': input_ids.clone(),
        }


def get_deepspeed_config(world_size):
    grad_accum = GRAD_ACCUM // world_size if world_size > 1 else GRAD_ACCUM
    train_batch = MBS * grad_accum * world_size
    
    use_bf16 = PRECISION == "bf16"
    use_fp16 = PRECISION == "fp16"
    
    return {
        "train_batch_size": train_batch,
        "train_micro_batch_size_per_gpu": MBS,
        "gradient_accumulation_steps": grad_accum,
        "steps_per_print": 1,
        "gradient_clipping": 1.0,
        "prescale_gradients": False,
        "bf16": {"enabled": use_bf16},
        "fp16": {"enabled": use_fp16},
        "zero_optimization": {
            "stage": 3,
            "offload_optimizer": {"device": "cpu", "pin_memory": True},
            "offload_param": {"device": "cpu", "pin_memory": True},
            "overlap_comm": True,
            "contiguous_gradients": True,
            "sub_group_size": 1e9,
            "reduce_bucket_size": "auto",
            "stage3_prefetch_bucket_size": "auto",
            "stage3_param_persistence_threshold": "auto",
            "stage3_max_live_parameters": 1e9,
            "stage3_max_reuse_distance": 1e9,
            "stage3_gather_16bit_weights_on_model_save": True
        },
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": LR,
                "betas": [BETA1, BETA2],
                "eps": 1e-8,
                "weight_decay": WEIGHT_DECAY
            }
        },
        "wall_clock_breakdown": False,
        "dump_state": False
    }


class SimpleCosineScheduler:
    def __init__(self, optimizer, base_lr, warmup_steps, total_steps):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.warmup_steps = max(1, warmup_steps)
        self.total_steps = max(1, total_steps)
        self.step_num = 0
    
    def _lr_for_step(self, step):
        if step < self.warmup_steps:
            return self.base_lr * float(step + 1) / float(self.warmup_steps)
        progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        return 0.5 * self.base_lr * (1.0 + math.cos(math.pi * progress))
    
    def step(self):
        lr = self._lr_for_step(self.step_num)
        for group in self.optimizer.param_groups:
            group['lr'] = lr
        self.step_num += 1
        return lr
    
    def get_last_lr(self):
        return [self.optimizer.param_groups[0]['lr']]


def train_model(model_name, model_short_name):
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    deepspeed.init_distributed()
    
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    grad_accum = GRAD_ACCUM // world_size if world_size > 1 else GRAD_ACCUM
    global_batch_size = MBS * grad_accum * world_size
    
    if torch.cuda.is_available():
        device_props = torch.cuda.get_device_properties(0)
        device_name = torch.cuda.get_device_name(0)
        platform = "nvd"
        software_stack = "deepspeed"
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
    dataset_path = str(DATA_DIR / f"allenai-c4-{model_short_name}-mega")
    
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
    
    if rank == 0:
        print(f"Initializing model: {model_name} (random weights)")
        print(f"World size: {world_size}, Rank: {rank}, Local rank: {local_rank}")
        print(f"Batch config: mbs={MBS}, grad_accum={grad_accum}, gbs={global_batch_size}")
        print(f"Using ZeRO Stage 3 with CPU offloading for memory efficiency")
        print(f"Dataset: {dataset_path}")
    
    torch_dtype = torch.bfloat16 if PRECISION == "bf16" else torch.float16 if PRECISION == "fp16" else torch.float32
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_config(config, torch_dtype=torch_dtype)
    model.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model.gradient_checkpointing_enable()
    
    dataset = PretrainingDataset(
        tokenizer=tokenizer,
        seq_length=SEQ_LEN,
        num_samples=TRAIN_ITERS * global_batch_size,
        data_path=dataset_path
    )
    
    ds_config = get_deepspeed_config(world_size)
    
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True,
    ) if world_size > 1 else None
    
    dataloader = DataLoader(
        dataset,
        batch_size=ds_config["train_micro_batch_size_per_gpu"],
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )
    
    model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_config,
    )
    
    if rank == 0:
        print("Starting training...")
    
    model_engine.train()
    use_external_scheduler = False
    if lr_scheduler is None:
        base_lr = optimizer.param_groups[0]['lr'] if optimizer and optimizer.param_groups else LR
        lr_scheduler = SimpleCosineScheduler(optimizer, base_lr=base_lr, warmup_steps=WARMUP_STEPS, total_steps=TRAIN_ITERS)
        use_external_scheduler = True
    
    if rank == 0:
        print(f"LR schedule: cosine with {WARMUP_STEPS} warmup steps")
    
    step = 0
    start_time = time.time()
    
    for batch_idx, batch in enumerate(dataloader):
        if step >= TRAIN_ITERS:
            break
        
        step_start = time.time()
        
        input_ids = batch['input_ids'].to(model_engine.device)
        labels = batch['labels'].to(model_engine.device)
        outputs = model_engine(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        model_engine.backward(loss)
        model_engine.step()
        
        if use_external_scheduler and lr_scheduler is not None:
            lr_scheduler.step()
        
        step_time = time.time() - step_start
        step_times.append(step_time)
        loss_values.append(loss.item())
        
        if lr_scheduler is not None:
            current_lr = lr_scheduler.get_last_lr()[0]
        else:
            current_lr = optimizer.param_groups[0]['lr'] if optimizer else LR
        learning_rates.append(current_lr)
        
        step += 1
        
        if rank == 0 and step % 10 == 0:
            avg_loss = sum(loss_values) / len(loss_values)
            elapsed = time.time() - start_time
            print(f"Step {step}/{TRAIN_ITERS}, Loss: {avg_loss:.4f}, Step Time: {step_time:.3f}s, Elapsed: {elapsed:.2f}s")
        
        if step >= TRAIN_ITERS:
            break
    
    if rank == 0:
        total_time = time.time() - start_time
        print(f"Training completed! Total time: {total_time:.2f}s")
        
        if len(step_times) > 10:
            warmup_skip = min(10, len(step_times))
            step_times_no_warmup = step_times[warmup_skip:]
            if not step_times_no_warmup:
                step_times_no_warmup = step_times
            
            avg_step_time = sum(step_times_no_warmup) / len(step_times_no_warmup)
            steps_per_second = len(step_times_no_warmup) / sum(step_times_no_warmup)
            tokens_per_step = global_batch_size * SEQ_LEN
            tokens_per_second = tokens_per_step / avg_step_time
            tokens_per_second_per_gpu = tokens_per_second / world_size if world_size else None
            
            from datetime import datetime
            from utils import round_floats
            
            results = {
                "platform": platform,
                "gpu_info": gpu_info,
                "timestamp": datetime.now().isoformat(),
                "training_config": {
                    "max_steps": TRAIN_ITERS,
                    "global_batch_size": global_batch_size,
                    "micro_batch_size": MBS,
                    "sequence_length": SEQ_LEN,
                    "num_gpus": world_size,
                    "parallel_strategy": "zero3",
                    "gradient_accumulation_steps": grad_accum,
                },
                "performance_metrics": {
                    "total_steps": len(step_times),
                    "total_time_seconds": total_time,
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
            
            print(f"Average step time: {avg_step_time:.3f}s")
            print(f"Throughput: {tokens_per_second:,.0f} tokens/sec")
            print(f"Per-GPU Throughput: {tokens_per_second_per_gpu:,.0f} tokens/sec/GPU")
            
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            output_file = OUTPUT_DIR / f"train_nvd_deep_{model_short_name}.json"
            results_rounded = round_floats(results, precision=5)
            
            with open(output_file, 'w') as f:
                json.dump(results_rounded, f, indent=2)
            
            print(f"Results saved to {output_file}")


def train_llama():
    train_model("meta-llama/Llama-3.1-8B", "llama")


def train_qwen():
    train_model("Qwen/Qwen2.5-7B", "qwen")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    if not torch.cuda.is_available():
        print("CUDA is not available!")
        sys.exit(1)
    
    model_args = [arg for arg in sys.argv[1:] if not arg.startswith('--')]
    
    if len(model_args) > 0:
        model = model_args[0]
        if model == "llama":
            train_llama()
        elif model == "qwen":
            train_qwen()
        else:
            print(f"Unknown model: {model}")
            sys.exit(1)
    else:
        import subprocess
        subprocess.run([sys.executable, __file__, "llama"])
        time.sleep(10)
        subprocess.run([sys.executable, __file__, "qwen"])


if __name__ == "__main__":
    main()
