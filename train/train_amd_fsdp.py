#!/usr/bin/env python3
import os
import sys
from pathlib import Path

# Resolve relative paths before importing transformers (which uses HF_HOME)
_workspace_root = Path(__file__).parent.parent
sys.path.insert(0, str(_workspace_root))
for _env_var in ("HF_HOME", "HF_DATASETS_CACHE"):
    _val = os.environ.get(_env_var)
    if _val and not os.path.isabs(_val):
        os.environ[_env_var] = str(_workspace_root / _val)

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload,
    BackwardPrefetch,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
import functools
import random
import numpy as np
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    get_cosine_schedule_with_warmup,
)
from torch.utils.data import Dataset, DataLoader
import json
import time
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
WORKSPACE_ROOT = Path(__file__).parent.parent
_output_dir = os.environ.get("OUTPUT_DIR", "output")
OUTPUT_DIR = Path(_output_dir) if os.path.isabs(_output_dir) else WORKSPACE_ROOT / _output_dir

os.environ['PYTHONUNBUFFERED'] = '1'
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

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
GA = int(os.environ.get("GA", 32))
DATASET = os.environ.get("DATASET", "bc")  # bc or c4


class PretrainingDataset(Dataset):
    def __init__(self, tokenizer, seq_length, num_samples, data_path):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.num_samples = num_samples
        
        if not data_path:
            raise ValueError("data_path is required - synthetic data is not allowed")
        
        from lib.mega_dataset import IndexedDataset
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


def setup_distributed():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        print("Not running in distributed mode")
        rank = 0
        world_size = 1
        local_rank = 0
    
    if world_size > 1:
        dist.init_process_group(backend='nccl')
    
    torch.cuda.set_device(local_rank)
    
    return rank, world_size, local_rank


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def train_model(model_name, model_short_name):
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    rank, world_size, local_rank = setup_distributed()
    
    grad_accum = GA // world_size if world_size > 1 else GA
    global_batch_size = MBS * grad_accum * world_size
    
    if torch.cuda.is_available():
        device_props = torch.cuda.get_device_properties(0)
        device_name = torch.cuda.get_device_name(0)
        platform = "amd"
        software_stack = "fsdp"
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
    
    if rank == 0:
        print(f"Loading model: {model_name}")
        print(f"World size: {world_size}, Rank: {rank}, Local rank: {local_rank}")
        print(f"Batch config: mbs={MBS}, grad_accum={grad_accum}, gbs={global_batch_size}")
        print(f"Using FSDP with FULL_SHARD strategy")
        print(f"Dataset: {dataset_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    dataset = PretrainingDataset(
        tokenizer=tokenizer,
        seq_length=SEQ_LEN,
        num_samples=TRAIN_ITERS * global_batch_size,
        data_path=dataset_path
    )
    
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True,
    ) if world_size > 1 else None
    
    dataloader = DataLoader(
        dataset,
        batch_size=MBS,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=0,
        pin_memory=True,
    )
    
    if rank == 0:
        print("Initializing model (random weights)...")
    
    torch_dtype = torch.bfloat16 if PRECISION == "bf16" else torch.float16 if PRECISION == "fp16" else torch.float32
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_config(config, torch_dtype=torch_dtype)
    model.config.use_cache = False
    
    auto_wrap_policy = functools.partial(size_based_auto_wrap_policy, min_num_params=100_000_000)
    
    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        cpu_offload=CPUOffload(offload_params=False),
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        device_id=torch.cuda.current_device(),
        limit_all_gathers=True,
        use_orig_params=True,
    )
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        betas=(BETA1, BETA2),
        eps=1e-8,
        weight_decay=WEIGHT_DECAY,
    )
    
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=TRAIN_ITERS,
    )
    
    if rank == 0:
        print("Starting training...")
    
    model.train()
    step = 0
    start_time = time.time()
    data_iter = iter(dataloader)
    
    for step in range(TRAIN_ITERS):
        step_start = time.time()
        optimizer.zero_grad()
        
        for _ in range(grad_accum):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)
            
            input_ids = batch['input_ids'].to(torch.cuda.current_device())
            labels = batch['labels'].to(torch.cuda.current_device())
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss / grad_accum
            loss.backward()
        
        optimizer.step()
        lr_scheduler.step()
        
        step_time = time.time() - step_start
        step_times.append(step_time)
        loss_values.append(loss.item() * grad_accum)
        current_lr = lr_scheduler.get_last_lr()[0] if lr_scheduler is not None else optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        
        if rank == 0 and (step + 1) % 10 == 0:
            avg_loss = sum(loss_values[-10:]) / min(10, len(loss_values))
            elapsed = time.time() - start_time
            print(f"Step {step+1}/{TRAIN_ITERS}, Loss: {avg_loss:.4f}, Step Time: {step_time:.3f}s, LR: {learning_rates[-1]:.6f}")
    
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
            
            from lib.utils import round_floats
            
            results = {
                "platform": platform,
                "dataset": DATASET,
                "gpu_info": gpu_info,
                "timestamp": datetime.now().isoformat(),
                "training_config": {
                    "max_steps": TRAIN_ITERS,
                    "global_batch_size": global_batch_size,
                    "micro_batch_size": MBS,
                    "sequence_length": SEQ_LEN,
                    "num_gpus": world_size,
                    "parallel_strategy": "fsdp",
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
            output_file = OUTPUT_DIR / f"train_{platform}_fsdp_{model_short_name}_{DATASET}.json"
            results_rounded = round_floats(results, precision=5)
            
            with open(output_file, 'w') as f:
                json.dump(results_rounded, f, indent=2)
            
            print(f"Results saved to {output_file}")
    
    cleanup_distributed()


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
            print("Usage: torchrun --nproc_per_node=N train_amd_fsdp.py [model]")
            print("  model: 'llama' or 'qwen'")
            sys.exit(1)
    else:
        print("Usage: torchrun --nproc_per_node=N train_amd_fsdp.py [model]")
        print("  model: 'llama' or 'qwen'")
        sys.exit(1)


if __name__ == "__main__":
    main()
