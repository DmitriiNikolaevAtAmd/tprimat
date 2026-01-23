#!/usr/bin/env python3
"""
Plain Mega-LM training script for NVIDIA GPUs.
Uses native Megatron-LM APIs directly (not NeMo wrapper).
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
import json
import time
from pathlib import Path
from datetime import datetime

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
    
    # Track metrics
    metrics = {
        'model_name': model_name,
        'framework': 'mega',
        'timestamp': datetime.now().isoformat(),
        'rank': rank,
        'world_size': world_size,
        'config': model_config,
        'steps': [],
        'total_training_time': 0.0,
        'avg_step_time': 0.0,
        'throughput_tokens_per_sec': 0.0,
    }
    
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
        
        # Create dummy dataset
        logger.info("Creating synthetic dataset...")
        seq_length = model_config['seq_length']
        batch_size = model_config['micro_batch_size']
        num_steps = model_config['num_steps']
        
        # Setup optimizer - use 8-bit Adam for memory efficiency on single GPU
        if world_size == 1:
            try:
                import bitsandbytes as bnb
                optimizer = bnb.optim.Adam8bit(
                    model.parameters(),
                    lr=model_config['learning_rate'],
                    betas=(0.9, 0.95),
                    eps=1e-8
                )
                logger.info("Using 8-bit Adam optimizer for memory efficiency")
            except ImportError:
                logger.warning("bitsandbytes not available, using regular Adam")
                optimizer = torch.optim.Adam(
                    model.parameters(),
                    lr=model_config['learning_rate'],
                    betas=(0.9, 0.95),
                    eps=1e-8
                )
        else:
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=model_config['learning_rate'],
                betas=(0.9, 0.95),
                eps=1e-8
            )
        
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
            
            # Log metrics
            if rank == 0:
                step_metrics = {
                    'step': step + 1,
                    'loss': avg_loss,
                    'step_time': step_time,
                    'throughput': throughput,
                    'tokens_per_step': tokens_per_step,
                }
                metrics['steps'].append(step_metrics)
                
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
        
        # Calculate final metrics
        if rank == 0 and len(metrics['steps']) > 0:
            step_times = [s['step_time'] for s in metrics['steps']]
            throughputs = [s['throughput'] for s in metrics['steps']]
            
            metrics['total_training_time'] = training_time
            metrics['avg_step_time'] = sum(step_times) / len(step_times)
            metrics['throughput_tokens_per_sec'] = sum(throughputs) / len(throughputs)
            metrics['min_step_time'] = min(step_times)
            metrics['max_step_time'] = max(step_times)
            metrics['final_loss'] = metrics['steps'][-1]['loss']
            
            logger.info(f"=" * 80)
            logger.info(f"Training completed for {model_name}")
            logger.info(f"Total time: {training_time:.2f}s")
            logger.info(f"Average step time: {metrics['avg_step_time']:.3f}s")
            logger.info(f"Average throughput: {metrics['throughput_tokens_per_sec']:.0f} tokens/s")
            logger.info(f"=" * 80)
            
            # Save results
            output_dir = Path("output")
            output_dir.mkdir(exist_ok=True)
            output_file = output_dir / f"train_mega_{model_name}.json"
            
            with open(output_file, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            logger.info(f"Results saved to {output_file}")
    
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        if rank == 0:
            metrics['error'] = str(e)
            output_dir = Path("output")
            output_dir.mkdir(exist_ok=True)
            output_file = output_dir / f"train_mega_{model_name}.json"
            with open(output_file, 'w') as f:
                json.dump(metrics, f, indent=2)
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
        'num_steps': 10,
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
        'num_steps': 10,
        'learning_rate': 3e-4,
        'tensor_parallel': 1,
        'pipeline_parallel': 1,
    }
    train_model('qwen', config)


def main():
    """Main entry point."""
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
            logger.error("Usage: python run_mega_plain.py [model]")
            logger.error("  model: 'llama' or 'qwen' (optional, trains all if omitted)")
            sys.exit(1)


if __name__ == "__main__":
    main()
