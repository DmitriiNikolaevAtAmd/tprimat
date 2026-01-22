#!/usr/bin/env python3
import os
import sys
import torch
import random
import numpy as np
from nemo.collections import llm
import nemo_run as run
from nemo.lightning import MegatronStrategy
from utils import BenchmarkCallback


def train_llama():
    """Train Llama 3.1 8B with hardcoded parameters"""
    print("\n" + "=" * 60)
    print("Training Llama 3.1 8B")
    print("=" * 60)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    random.seed(42)
    os.environ['PYTHONHASHSEED'] = '42'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    recipe = llm.llama31_8b.pretrain_recipe(
        name="llama31_8b_pretrain",
        dir="/data",
        num_nodes=1,
        num_gpus_per_node=8,
    )
    
    print("  * Parallelism: TP=1, PP=1, DP=8, GradAccum=16")
    recipe.trainer.strategy = MegatronStrategy(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        virtual_pipeline_model_parallel_size=None,
        sequence_parallel=False,
    )
    
    # Data configuration
    dataset_path = "/data/llama_dataset_text_document"
    tokenizer_path = "meta-llama/Llama-3.1-8B"
    
    if os.path.exists(dataset_path + ".idx"):
        print(f"  * Using real data: {dataset_path}")
        from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer as NeMoAutoTokenizer
        print(f"  * Loading tokenizer from: {tokenizer_path}")
        tokenizer = NeMoAutoTokenizer(tokenizer_path)
        
        from nemo.collections.llm.gpt.data.pre_training import PreTrainingDataModule
        recipe.data = PreTrainingDataModule(
            paths=[dataset_path],
            seq_length=2048,
            micro_batch_size=1,
            global_batch_size=128,
            tokenizer=tokenizer,
            num_workers=2,
        )
    else:
        print("  * Using mock data for benchmarking")
        recipe.data.micro_batch_size = 1
        recipe.data.global_batch_size = 128
        recipe.data.seq_length = 2048
    
    # Training duration
    recipe.trainer.max_steps = 500
    
    # Optimizer configuration
    recipe.optim.config.lr = 0.0003
    recipe.optim.config.min_lr = 0.00003
    recipe.optim.config.weight_decay = 0.1
    recipe.optim.config.adam_beta1 = 0.9
    recipe.optim.config.adam_beta2 = 0.95
    
    # Warmup scheduler
    recipe.optim.lr_scheduler.warmup_steps = 50
    recipe.optim.lr_scheduler.constant_steps = 0
    
    # FP8 precision settings (NVIDIA H100 optimizations)
    recipe.model.config.fp8 = "hybrid"
    recipe.model.config.fp8_param = True
    
    # Disable checkpointing (benchmark only)
    recipe.trainer.enable_checkpointing = False
    recipe.log.ckpt = None
    recipe.resume = None
    recipe.log.tensorboard = None
    recipe.log.wandb = None
    recipe.trainer.val_check_interval = None
    recipe.trainer.check_val_every_n_epoch = None
    
    # Add benchmark callback
    benchmark_callback = BenchmarkCallback(
        output_dir="./output",
        platform="auto",
        model_name="llama",
        parallel_strategy="minimal_communication",
        profiler_config={"enabled": False}
    )
    if recipe.trainer.callbacks is None:
        recipe.trainer.callbacks = []
    recipe.trainer.callbacks.append(benchmark_callback)
    
    print("  * Starting training...")
    run.run(recipe, direct=True)
    print("  ✓ Training completed")


def train_qwen():
    """Train Qwen 2.5 7B with hardcoded parameters"""
    print("\n" + "=" * 60)
    print("Training Qwen 2.5 7B")
    print("=" * 60)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    random.seed(42)
    os.environ['PYTHONHASHSEED'] = '42'
    
    # Set PyTorch memory allocator
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # Initialize recipe
    recipe = llm.qwen25_7b.pretrain_recipe(
        name="qwen25_7b_pretrain",
        dir="/data",
        num_nodes=1,
        num_gpus_per_node=8,
    )
    
    print("  * Parallelism: TP=1, PP=1, DP=8, GradAccum=16")
    recipe.trainer.strategy = MegatronStrategy(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        virtual_pipeline_model_parallel_size=None,
        sequence_parallel=False,
    )
    
    dataset_path = "/data/llama_dataset_text_document"
    tokenizer_path = "Qwen/Qwen2.5-7B"
    
    if os.path.exists(dataset_path + ".idx"):
        print(f"  * Using real data: {dataset_path}")
        from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer as NeMoAutoTokenizer
        print(f"  * Loading tokenizer from: {tokenizer_path}")
        tokenizer = NeMoAutoTokenizer(tokenizer_path)
        
        from nemo.collections.llm.gpt.data.pre_training import PreTrainingDataModule
        recipe.data = PreTrainingDataModule(
            paths=[dataset_path],
            seq_length=2048,
            micro_batch_size=1,
            global_batch_size=128,
            tokenizer=tokenizer,
            num_workers=2,
        )
    else:
        print("  * Using mock data for benchmarking")
        recipe.data.micro_batch_size = 1
        recipe.data.global_batch_size = 128
        recipe.data.seq_length = 2048
    
    recipe.trainer.max_steps = 500
    
    recipe.optim.config.lr = 0.0003  # 3.0e-4
    recipe.optim.config.min_lr = 0.00003  # 3.0e-5 (10% of peak LR)
    recipe.optim.config.weight_decay = 0.1
    recipe.optim.config.adam_beta1 = 0.9
    recipe.optim.config.adam_beta2 = 0.95
    
    recipe.optim.lr_scheduler.warmup_steps = 50
    recipe.optim.lr_scheduler.constant_steps = 0
    
    recipe.model.config.fp8 = "hybrid"
    recipe.model.config.fp8_param = True
    
    recipe.trainer.enable_checkpointing = False
    recipe.log.ckpt = None
    recipe.resume = None
    recipe.log.tensorboard = None
    recipe.log.wandb = None
    recipe.trainer.val_check_interval = None
    recipe.trainer.check_val_every_n_epoch = None
    
    benchmark_callback = BenchmarkCallback(
        output_dir="./output",
        platform="auto",
        model_name="qwen",
        parallel_strategy="minimal_communication",
        profiler_config={"enabled": False}
    )
    if recipe.trainer.callbacks is None:
        recipe.trainer.callbacks = []
    recipe.trainer.callbacks.append(benchmark_callback)
    
    print("  * Starting training...")
    run.run(recipe, direct=True)
    print("  ✓ Training completed")


def main():
    """Main entry point - runs both models sequentially"""
    print("=" * 60)
    print("NVIDIA LLM Training Benchmark - Standalone")
    print("=" * 60)
    print("Hardware: 8x NVIDIA H100 (80GB each)")
    print("Strategy: Minimal Communication (TP=1, PP=1, DP=8)")
    print("Batch size: 128 (micro=1, grad_accum=16)")
    print("Sequence length: 2048 tokens")
    print("Training steps: 500")
    print("Precision: FP8 Hybrid")
    print()
    
    os.makedirs("./output", exist_ok=True)
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available!")
        sys.exit(1)
    
    gpu_count = torch.cuda.device_count()
    print(f"Detected {gpu_count} GPUs")
    if gpu_count != 8:
        print(f"WARNING: Expected 8 GPUs, found {gpu_count}")
    
    for i in range(min(gpu_count, 8)):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    print()
    
    try:
        train_llama()
        train_qwen()
        
        print("\n" + "=" * 60)
        print("Summary")
        print("=" * 60)
        print("✓ Both models trained successfully")
        print("  - Llama 3.1 8B: completed")
        print("  - Qwen 2.5 7B: completed")
        print()
        
        sys.exit(0)
        
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
