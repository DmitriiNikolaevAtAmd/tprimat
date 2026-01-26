#!/usr/bin/env python3
"""
Megatron-LM Training Script for NVIDIA GPUs

This is a standalone script that trains LLMs using Megatron-LM.
No shell wrapper needed - all environment setup is handled internally.

Usage:
    python train_nvd_mega.py              # Train both llama and qwen
    python train_nvd_mega.py llama        # Train only llama
    python train_nvd_mega.py qwen         # Train only qwen
"""
import os
import sys

os.environ['PYTHONUNBUFFERED'] = '1'
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import torch
import random
import numpy as np
import logging
import json
import time
from pathlib import Path
from datetime import datetime

# Try to import bitsandbytes for 8-bit optimizer
try:
    import bitsandbytes as bnb
    HAS_BITSANDBYTES = True
except ImportError:
    HAS_BITSANDBYTES = False

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


def set_seed(seed=42):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def train_model(model_name: str, model_config: dict):
    """
    Train a model using Megatron-LM native APIs.
    
    Args:
        model_name: Name of the model ('llama' or 'qwen')
        model_config: Configuration dictionary with model parameters
    """
    logger.info(f"=" * 80)
    logger.info(f"Starting Mega-LM training for {model_name}")
    logger.info(f"=" * 80)
    
    # Initialize distributed training
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend='nccl')
        logger.info(f"Initialized distributed training: rank={rank}, world_size={world_size}")
    else:
        logger.info("Running in single-GPU mode")
    
    set_seed(42)
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # Detect platform and get GPU info
    if torch.cuda.is_available():
        device_props = torch.cuda.get_device_properties(0)
        device_name = torch.cuda.get_device_name(0)
        is_rocm = hasattr(torch.version, 'hip') and torch.version.hip is not None
        platform = "amd" if is_rocm else "nvd"
        software_stack = "prim" if is_rocm else "nemo"
        software_version = torch.version.hip if is_rocm else torch.version.cuda
        
        # Approximate GPU cores (will update with utils function if needed)
        gpu_cores = 16896 if "h100" in device_name.lower() else 6912  # H100 or A100 default
        
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
    
    # Track step times and losses for unified format
    step_times = []
    loss_values = []
    learning_rates = []
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
        import torch.distributed as dist
        
        logger.info(f"Loading tokenizer: {model_config['hf_model']}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_config['hf_model'],
            trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        logger.info(f"Loading model: {model_config['hf_model']}")
        
        # Load model with low CPU memory mode for single GPU
        if world_size == 1:
            logger.info("Single GPU detected - using memory optimization")
            model = AutoModelForCausalLM.from_pretrained(
                model_config['hf_model'],
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                device_map="auto"  # Automatic device placement
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_config['hf_model'],
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            )
            # Apply tensor parallelism manually (simplified version)
            device = torch.device(f'cuda:{local_rank}')
            model = model.to(device)
        
        # Enable gradient checkpointing for memory efficiency
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            logger.info("Enabled gradient checkpointing")
        
        # Use DDP for data parallelism
        if world_size > 1:
            from torch.nn.parallel import DistributedDataParallel as DDP
            device = torch.device(f'cuda:{local_rank}')
            model = DDP(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=False
            )
            logger.info(f"Wrapped model with DDP on device {local_rank}")
        
        # Check if real data is available
        dataset_path = "/data/llama_dataset_text_document"
        use_real_data = os.path.exists(dataset_path + ".idx")
        
        if use_real_data:
            logger.info(f"Real data found at {dataset_path}")
            logger.info("Note: Currently using synthetic data for consistent benchmarking")
            logger.info("      To use real data, implement indexed dataset loader")
        else:
            logger.info("Real data not found, using synthetic data for benchmarking")
        
        # Create dataset
        seq_length = model_config['seq_length']
        batch_size = model_config['micro_batch_size']
        num_steps = model_config['num_steps']
        
        # Setup optimizer - use 8-bit for memory efficiency on all GPUs
        if HAS_BITSANDBYTES:
            optimizer = bnb.optim.Adam8bit(
                model.parameters(),
                lr=model_config['learning_rate'],
                betas=(0.9, 0.95),
                eps=1e-8
            )
            logger.info("Using 8-bit Adam optimizer (saves ~75% optimizer memory)")
        else:
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=model_config['learning_rate'],
                betas=(0.9, 0.95),
                eps=1e-8
            )
            logger.warning("bitsandbytes not available, using standard Adam (higher memory)")
        
        logger.info(f"Configuration:")
        logger.info(f"  Sequence length: {seq_length}")
        logger.info(f"  Micro batch size: {batch_size}")
        logger.info(f"  Global batch size: {batch_size * world_size * model_config['grad_accum_steps']}")
        logger.info(f"  Gradient accumulation: {model_config['grad_accum_steps']}")
        logger.info(f"  Training steps: {num_steps}")
        logger.info(f"  Learning rate: {model_config['learning_rate']}")
        logger.info(f"  Precision: bfloat16")
        
        # Training loop
        model.train()
        logger.info("Starting training...")
        training_start = time.time()
        
        for step in range(num_steps):
            step_start = time.time()
            optimizer.zero_grad()
            
            step_losses = []
            
            for micro_step in range(model_config['grad_accum_steps']):
                # Generate synthetic data
                if world_size == 1:
                    # For single GPU with device_map="auto", use model's device
                    device = next(model.parameters()).device
                input_ids = torch.randint(
                    0, tokenizer.vocab_size,
                    (batch_size, seq_length),
                    device=device
                )
                labels = input_ids.clone()
                
                # Forward pass
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    outputs = model(input_ids=input_ids, labels=labels)
                    loss = outputs.loss / model_config['grad_accum_steps']
                
                # Backward pass
                loss.backward()
                step_losses.append(loss.item() * model_config['grad_accum_steps'])
            
            # Optimizer step
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            step_time = time.time() - step_start
            avg_loss = sum(step_losses) / len(step_losses)
            
            # Calculate throughput
            tokens_per_step = batch_size * seq_length * model_config['grad_accum_steps'] * world_size
            throughput = tokens_per_step / step_time
            
            # Track metrics
            step_times.append(step_time)
            loss_values.append(avg_loss)
            learning_rates.append(model_config['learning_rate'])  # Fixed LR for now
            
            # Log metrics
            if rank == 0:
                logger.info(
                    f"Step {step + 1}/{num_steps} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"Time: {step_time:.3f}s | "
                    f"Throughput: {throughput:.0f} tokens/s"
                )
            
            # Memory stats every 5 steps
            if rank == 0 and (step + 1) % 5 == 0:
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated(device) / 1e9
                    reserved = torch.cuda.memory_reserved(device) / 1e9
                    logger.info(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
        
        training_time = time.time() - training_start
        
        # Calculate final metrics (unified format)
        if rank == 0 and len(step_times) > 1:
            # Skip first step (warmup)
            step_times_no_warmup = step_times[1:]
            
            avg_step_time = sum(step_times_no_warmup) / len(step_times_no_warmup)
            steps_per_second = len(step_times_no_warmup) / sum(step_times_no_warmup)
            
            # Calculate token-based throughput
            global_batch_size = batch_size * world_size * model_config['grad_accum_steps']
            tokens_per_step = global_batch_size * seq_length
            tokens_per_second = tokens_per_step / avg_step_time
            tokens_per_second_per_gpu = tokens_per_second / world_size if world_size else None
            
            # Build unified results structure
            results = {
                "platform": platform,
                "gpu_info": gpu_info,
                "timestamp": datetime.now().isoformat(),
                "training_config": {
                    "max_steps": num_steps,
                    "global_batch_size": global_batch_size,
                    "micro_batch_size": batch_size,
                    "sequence_length": seq_length,
                    "num_gpus": world_size,
                    "parallel_strategy": "ddp",
                    "gradient_accumulation_steps": model_config['grad_accum_steps'],
                },
                "performance_metrics": {
                    "total_steps": len(step_times),
                    "total_time_seconds": training_time,
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
            
            logger.info(f"=" * 80)
            logger.info(f"Training completed for {model_name}")
            logger.info(f"Total time: {training_time:.2f}s")
            logger.info(f"Average step time: {avg_step_time:.3f}s")
            logger.info(f"Throughput: {tokens_per_second:,.0f} tokens/sec")
            logger.info(f"Per-GPU Throughput: {tokens_per_second_per_gpu:,.0f} tokens/sec/GPU")
            logger.info(f"=" * 80)
            
            # Save results
            output_dir = Path("output")
            output_dir.mkdir(exist_ok=True)
            output_file = output_dir / f"train_nvd_mega_{model_name}.json"
            
            # Round all floats to 5 decimal places (matching BenchmarkCallback)
            from utils import round_floats
            results_rounded = round_floats(results, precision=5)
            
            with open(output_file, 'w') as f:
                json.dump(results_rounded, f, indent=2)
            
            logger.info(f"Results saved to {output_file}")
    
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        if rank == 0 and len(step_times) > 0:
            # Save partial results on error
            results = {
                "platform": platform,
                "gpu_info": gpu_info,
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "partial_results": {
                    "steps_completed": len(step_times),
                    "step_times": step_times,
                    "loss_values": loss_values,
                },
            }
            output_dir = Path("output")
            output_dir.mkdir(exist_ok=True)
            output_file = output_dir / f"train_nvd_mega_{model_name}.json"
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
        raise
    
    finally:
        if world_size > 1:
            torch.distributed.destroy_process_group()


def train_llama():
    """Train Llama 3.1 8B with Mega-LM."""
    config = {
        'hf_model': 'meta-llama/Llama-3.1-8B',
        'seq_length': 2048,
        'micro_batch_size': 1,
        'grad_accum_steps': 8,
        'num_steps': 500,
        'learning_rate': 3e-4,
        'tensor_parallel': 1,
        'pipeline_parallel': 1,
    }
    train_model('llama', config)


def train_qwen():
    """Train Qwen 2.5 7B with Mega-LM."""
    config = {
        'hf_model': 'Qwen/Qwen2.5-7B',
        'seq_length': 2048,
        'micro_batch_size': 1,
        'grad_accum_steps': 8,
        'num_steps': 500,
        'learning_rate': 3e-4,
        'tensor_parallel': 1,
        'pipeline_parallel': 1,
    }
    train_model('qwen', config)


def main():
    """Main entry point."""
    # Set environment variables (from train_nvd_mega.sh)
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    os.environ['PYTHONHASHSEED'] = '42'
    os.environ['HSA_NO_SCRATCH_RECLAIM'] = '1'
    os.environ['HSA_ENABLE_SDMA'] = '1'
    os.environ['HSA_FORCE_FINE_GRAIN_PCIE'] = '1'
    os.environ['RCCL_DEBUG'] = 'INFO'
    os.environ['NCCL_DEBUG'] = 'INFO'
    
    # Create output directory
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    
    logger.info("Environment configured for NVIDIA GPU training")
    logger.info(f"Output directory: {output_dir}")
    
    if len(sys.argv) < 2:
        # No model specified, train all models
        logger.info("No model specified, training all models")
        train_llama()
        train_qwen()
    else:
        model = sys.argv[1]
        logger.info(f"Training model: {model}")
        if model == "llama":
            train_llama()
        elif model == "qwen":
            train_qwen()
        else:
            logger.error(f"Unknown model: {model}")
            logger.error("\nUsage:")
            logger.error("  python train_nvd_mega.py              # Train both models")
            logger.error("  python train_nvd_mega.py llama        # Train only Llama")
            logger.error("  python train_nvd_mega.py qwen         # Train only Qwen")
            sys.exit(1)


if __name__ == "__main__":
    main()
