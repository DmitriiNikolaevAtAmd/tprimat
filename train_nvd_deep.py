#!/usr/bin/env python3
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
    def __init__(self, tokenizer, seq_length=2048, num_samples=640, use_real_data=False, data_path=None):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.num_samples = num_samples
        self.use_real_data = use_real_data
        self.data_path = data_path
        self.indexed_dataset = None
        
        if self.use_real_data and data_path:
            try:
                from indexed_dataset import IndexedDataset
                self.indexed_dataset = IndexedDataset(data_path)
                print(f"✓ Loaded real indexed dataset from {data_path}")
                print(f"  Dataset contains {len(self.indexed_dataset)} sequences")
                self.real_data_available = True
            except Exception as e:
                print(f"⚠ Could not load real data: {e}")
                print(f"  Falling back to synthetic data")
                self.real_data_available = False
        else:
            self.real_data_available = False
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        if self.real_data_available and self.indexed_dataset is not None:
            # Load real data and pad/truncate to seq_length
            dataset_idx = idx % len(self.indexed_dataset)
            tokens = self.indexed_dataset[dataset_idx]
            
            # Pad or truncate to seq_length
            if len(tokens) < self.seq_length:
                # Pad with tokenizer pad token
                pad_token = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id else 0
                padding = torch.full((self.seq_length - len(tokens),), pad_token, dtype=torch.long)
                input_ids = torch.cat([tokens, padding])
            else:
                input_ids = tokens[:self.seq_length]
        else:
            # Synthetic data fallback
            input_ids = torch.randint(0, self.tokenizer.vocab_size, (self.seq_length,))
        
        return {
            'input_ids': input_ids,
            'labels': input_ids.clone(),
        }


def get_deepspeed_config(world_size=1):
    micro_batch = 1
    grad_accum = 8
    train_batch = micro_batch * grad_accum * world_size
    
    return {
        "train_batch_size": train_batch,
        "train_micro_batch_size_per_gpu": micro_batch,
        "gradient_accumulation_steps": grad_accum,
        "steps_per_print": 1,
        "gradient_clipping": 1.0,
        "prescale_gradients": False,
        "bf16": {
            "enabled": True
        },
        "zero_optimization": {
            "stage": 3,
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True
            },
            "offload_param": {
                "device": "cpu",
                "pin_memory": True
            },
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
                "lr": 0.0003,
                "betas": [0.9, 0.95],
                "eps": 1e-8,
                "weight_decay": 0.1
            }
        },
        "wall_clock_breakdown": False,
        "dump_state": False
    }


def train_model(model_name, model_short_name):
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    random.seed(42)
    os.environ['PYTHONHASHSEED'] = '42'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    deepspeed.init_distributed()
    
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    if torch.cuda.is_available():
        device_props = torch.cuda.get_device_properties(0)
        device_name = torch.cuda.get_device_name(0)
        platform = "nvd"
        software_stack = "nemo"
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
    dataset_path = "/data/llama_dataset_text_document"
    use_real_data = os.path.exists(dataset_path + ".idx")
    
    if rank == 0:
        print(f"Loading model: {model_name}")
        print(f"World size: {world_size}, Rank: {rank}, Local rank: {local_rank}")
        print(f"Batch config: micro_batch=1, grad_accum=8, train_batch={1*8*world_size}")
        print(f"Using ZeRO Stage 3 with CPU offloading for memory efficiency")
        if use_real_data:
            print(f"Real data found at {dataset_path}")
        else:
            print("Real data not found, using synthetic data for benchmarking")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        use_cache=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model.gradient_checkpointing_enable()
    dataset = PretrainingDataset(
        tokenizer=tokenizer,
        seq_length=2048,
        num_samples=32000,
        use_real_data=False,  # Disabled: use synthetic for consistent benchmarking
        data_path=None
    )
    ds_config = get_deepspeed_config(world_size)
    model_engine, optimizer, dataloader, lr_scheduler = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        training_data=dataset,
        config=ds_config,
    )
    if rank == 0:
        print("Starting training...")
    model_engine.train()
    total_steps = 500
    step = 0
    start_time = time.time()
    
    for batch_idx, batch in enumerate(dataloader):
        if step >= total_steps:
            break
        
        step_start = time.time()
        
        input_ids = batch['input_ids'].to(model_engine.device)
        labels = batch['labels'].to(model_engine.device)
        outputs = model_engine(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        model_engine.backward(loss)
        model_engine.step()
        step_time = time.time() - step_start
        step_times.append(step_time)
        loss_values.append(loss.item())
        current_lr = optimizer.param_groups[0]['lr'] if optimizer else 0.0003
        learning_rates.append(current_lr)
        
        step += 1
        
        if rank == 0 and step % 10 == 0:
            avg_loss = sum(loss_values) / len(loss_values)
            elapsed = time.time() - start_time
            print(f"Step {step}/{total_steps}, Loss: {avg_loss:.4f}, Step Time: {step_time:.3f}s, Elapsed: {elapsed:.2f}s")
        
        if step >= total_steps:
            break
    
    if rank == 0:
        total_time = time.time() - start_time
        print(f"Training completed! Total time: {total_time:.2f}s")
        if len(step_times) > 10:
            step_times_no_warmup = step_times[50:]
            avg_step_time = sum(step_times_no_warmup) / len(step_times_no_warmup)
            steps_per_second = len(step_times_no_warmup) / sum(step_times_no_warmup)
            micro_batch = 1
            grad_accum = 8
            global_batch_size = micro_batch * grad_accum * world_size
            seq_length = 2048
            tokens_per_step = global_batch_size * seq_length
            tokens_per_second = tokens_per_step / avg_step_time
            tokens_per_second_per_gpu = tokens_per_second / world_size if world_size else None
            from datetime import datetime
            import json
            from pathlib import Path
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
            output_dir = Path("output")
            output_dir.mkdir(exist_ok=True)
            output_file = output_dir / f"train_nvd_deep_{model_short_name}.json"
            results_rounded = round_floats(results, precision=5)
            
            with open(output_file, 'w') as f:
                json.dump(results_rounded, f, indent=2)
            
            print(f"Results saved to {output_file}")


def train_llama():
    train_model("meta-llama/Llama-3.1-8B", "llama")


def train_qwen():
    train_model("Qwen/Qwen2.5-7B", "qwen")


def main():
    os.makedirs("./output", exist_ok=True)
    
    if not torch.cuda.is_available():
        print("CUDA is not available!")
        sys.exit(1)
    
    # Filter out DeepSpeed arguments (--local_rank, etc.)
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
        import time
        
        # Note: For Deep, you typically need to use deepspeed launcher
        # instead of python directly:
        # deepspeed --num_gpus=8 train_deep.py llama
        subprocess.run([sys.executable, __file__, "llama"])
        time.sleep(10)
        subprocess.run([sys.executable, __file__, "qwen"])


if __name__ == "__main__":
    main()
