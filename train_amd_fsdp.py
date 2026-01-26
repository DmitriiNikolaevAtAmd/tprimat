#!/usr/bin/env python3
import os
import sys
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload,
    BackwardPrefetch,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
)
import functools
import random
import numpy as np
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)
from torch.utils.data import Dataset, DataLoader
import json
import time
from datetime import datetime
from pathlib import Path

os.environ['PYTHONUNBUFFERED'] = '1'
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)


class PretrainingDataset(Dataset):
    def __init__(self, tokenizer, seq_length=2048, num_samples=32000, use_real_data=False, data_path=None):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.num_samples = num_samples
        self.use_real_data = use_real_data
        self.data_path = data_path
        
        if self.use_real_data and data_path:
            print(f"Using real data from {data_path}")
            self.real_data_available = True
        else:
            self.real_data_available = False
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        input_ids = torch.randint(0, self.tokenizer.vocab_size, (self.seq_length,))
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
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    random.seed(42)
    os.environ['PYTHONHASHSEED'] = '42'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    rank, world_size, local_rank = setup_distributed()
    
    if torch.cuda.is_available():
        device_props = torch.cuda.get_device_properties(0)
        device_name = torch.cuda.get_device_name(0)
        platform = "amd"
        software_stack = "rocm"
        software_version = torch.version.hip if hasattr(torch.version, 'hip') else "unknown"
        
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
    
    micro_batch = 1
    grad_accum = 8
    global_batch_size = micro_batch * grad_accum * world_size
    seq_length = 2048
    total_steps = 50
    
    step_times = []
    loss_values = []
    learning_rates = []
    
    dataset_path = "/data/llama_dataset_text_document"
    use_real_data = os.path.exists(dataset_path + ".idx")
    
    if rank == 0:
        print(f"Loading model: {model_name}")
        print(f"World size: {world_size}, Rank: {rank}, Local rank: {local_rank}")
        print(f"Batch config: micro_batch={micro_batch}, grad_accum={grad_accum}, global_batch={global_batch_size}")
        print(f"Using FSDP with FULL_SHARD strategy")
        if use_real_data:
            print(f"Real data found at {dataset_path}")
        else:
            print("Real data not found, using synthetic data for benchmarking")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    dataset = PretrainingDataset(
        tokenizer=tokenizer,
        seq_length=seq_length,
        num_samples=32000,
        use_real_data=use_real_data,
        data_path=dataset_path
    )
    
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
    ) if world_size > 1 else None
    
    dataloader = DataLoader(
        dataset,
        batch_size=micro_batch,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=2,
        pin_memory=True,
    )
    
    if rank == 0:
        print("Loading model...")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        use_cache=False,
    )
    
    model.gradient_checkpointing_enable()
    
    auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy,
        min_num_params=100_000_000,
    )
    
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
        lr=0.0003,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.1,
    )
    
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=10,
        num_training_steps=total_steps,
    )
    
    if rank == 0:
        print("Starting training...")
    
    model.train()
    step = 0
    start_time = time.time()
    data_iter = iter(dataloader)
    
    for step in range(total_steps):
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
        learning_rates.append(optimizer.param_groups[0]['lr'])
        
        if rank == 0 and (step + 1) % 10 == 0:
            avg_loss = sum(loss_values[-10:]) / min(10, len(loss_values))
            elapsed = time.time() - start_time
            print(f"Step {step+1}/{total_steps}, Loss: {avg_loss:.4f}, Step Time: {step_time:.3f}s, LR: {learning_rates[-1]:.6f}")
    
    if rank == 0:
        total_time = time.time() - start_time
        print(f"Training completed! Total time: {total_time:.2f}s")
        if len(step_times) > 10:
            step_times_no_warmup = step_times[10:]
            
            avg_step_time = sum(step_times_no_warmup) / len(step_times_no_warmup)
            steps_per_second = len(step_times_no_warmup) / sum(step_times_no_warmup)
            
            tokens_per_step = global_batch_size * seq_length
            tokens_per_second = tokens_per_step / avg_step_time
            tokens_per_second_per_gpu = tokens_per_second / world_size if world_size else None
            
            from utils import round_floats
            
            results = {
                "platform": platform,
                "gpu_info": gpu_info,
                "timestamp": datetime.now().isoformat(),
                "training_config": {
                    "max_steps": total_steps,
                    "global_batch_size": global_batch_size,
                    "micro_batch_size": micro_batch,
                    "sequence_length": seq_length,
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
            
            output_dir = Path("output")
            output_dir.mkdir(exist_ok=True)
            output_file = output_dir / f"train_{platform}_fsdp_{model_short_name}.json"
            
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
    os.makedirs("./output", exist_ok=True)
    
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
            print("Usage: torchrun --nproc_per_node=N train_all_fsdp.py [model]")
            print("  model: 'llama' or 'qwen'")
            sys.exit(1)
    else:
        print("Usage: torchrun --nproc_per_node=N train_all_fsdp.py [model]")
        print("  model: 'llama' or 'qwen'")
        sys.exit(1)


if __name__ == "__main__":
    main()
