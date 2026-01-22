#!/usr/bin/env python3
"""
Standalone AMD/ROCm Training Script for LLM Benchmarking
All parameters are hardcoded (no config loading)
Runs both Llama 3.1 8B and Qwen 2.5 7B models using Primus framework
"""

import os
import sys
import subprocess
import yaml
from pathlib import Path


def create_primus_config(model_name, output_dir, primus_path):
    """Create Primus YAML config with hardcoded parameters"""
    
    # Model-specific settings
    if model_name == "llama":
        num_layers = 32
        hidden_size = 4096
        num_attention_heads = 32
        ffn_hidden_size = 14336  # Llama uses SwiGLU: 4 * hidden_size * 7/8
        base_config = "llama3.1_8B-BF16-pretrain.yaml"
    elif model_name == "qwen":
        num_layers = 28
        hidden_size = 3584
        num_attention_heads = 28
        ffn_hidden_size = 18944  # Qwen 2.5 7B FFN dimension
        base_config = "qwen2.5_7B-BF16-pretrain.yaml"
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Try to load base config from Primus if it exists
    base_config_path = os.path.join(primus_path, "examples/megatron/configs/MI300X", base_config)
    if os.path.exists(base_config_path):
        print(f"  * Loading base config: {base_config_path}")
        with open(base_config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        print(f"  * Base config not found, creating from scratch")
        config = {}
    
    # Override with hardcoded parameters
    config.update({
        # Model architecture
        'num_layers': num_layers,
        'hidden_size': hidden_size,
        'num_attention_heads': num_attention_heads,
        'ffn_hidden_size': ffn_hidden_size,
        
        # Parallelism (minimal_communication strategy)
        'tensor_model_parallel_size': 1,
        'pipeline_model_parallel_size': 1,
        'gradient_accumulation_steps': 16,
        
        # Data configuration
        'micro_batch_size': 1,
        'global_batch_size': 128,
        'seq_length': 2048,
        
        # Training duration
        'train_iters': 500,
        'eval_iters': 0,
        'eval_interval': 500,
        
        # Optimizer
        'optimizer': 'adam',
        'lr': 0.0003,  # 3.0e-4
        'min_lr': 0.00003,  # 3.0e-5
        'lr_warmup_iters': 50,
        'lr_decay_style': 'cosine',
        'lr_decay_iters': 500,
        'weight_decay': 0.1,
        'adam_beta1': 0.9,
        'adam_beta2': 0.95,
        'clip_grad': 1.0,
        
        # Precision (FP8 hybrid mode for MI300X)
        'bf16': True,
        'fp8': 'hybrid',
        'fp8_param': True,
        
        # Optimizations
        'use_distributed_optimizer': True,
        'use_flash_attn': True,
        'use_fused_rmsnorm': True,
        'fp32_residual_connection': False,
        
        # Memory logging (for benchmarking)
        'log_memory_usage': True,
        'log_interval': 1,
        'log_timers_to_tensorboard': False,
        'tensorboard_log_interval': 1,
        
        # Checkpointing (disabled for benchmark)
        'save_interval': 10000,  # Effectively disabled
        'eval_interval': 10000,
        
        # Seed
        'seed': 42,
        
        # Profiling (disabled by default)
        'profile': False,
    })
    
    # Save config
    config_path = os.path.join(output_dir, f"{model_name}_primus_config.yaml")
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"  * Config saved to: {config_path}")
    return config_path


def train_model(model_name, primus_path, output_dir):
    """Train a model using Primus with hardcoded parameters"""
    print("\n" + "=" * 60)
    print(f"Training {model_name.upper()} with Primus")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set environment variables
    env = os.environ.copy()
    env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    env['PYTHONHASHSEED'] = '42'
    
    # Create Primus config
    config_path = create_primus_config(model_name, output_dir, primus_path)
    env['EXP'] = config_path
    
    # Model-specific settings
    if model_name == "llama":
        print("  * Model: Llama 3.1 8B")
    elif model_name == "qwen":
        print("  * Model: Qwen 2.5 7B")
    
    # Parallelism configuration
    print("  * Hardware: 8x AMD MI300X (192GB each)")
    print("  * Parallelism: TP=1, PP=1, DP=8, GradAccum=16")
    print("  * Batch size: 128 (micro=1)")
    print("  * Sequence length: 2048 tokens")
    print("  * Training steps: 500")
    print("  * Precision: FP8 Hybrid")
    print("  * Learning rate: 0.0003")
    print("  * Warmup steps: 50")
    print()
    
    # Log files
    log_file = os.path.join(output_dir, f"training_main_{model_name}.log")
    backup_log = os.path.join(output_dir, f"primus_training_{model_name}.log")
    
    print(f"  * Starting training...")
    print(f"  * Log files:")
    print(f"    - Primary: {log_file}")
    print(f"    - Backup:  {backup_log}")
    print()
    
    # Check if Primus exists
    if not os.path.isdir(primus_path):
        print(f"  ✗ Primus not found at {primus_path}")
        print(f"  Please set PRIMUS_PATH environment variable or ensure Primus is installed")
        return False
    
    # Check if run script exists
    run_script = os.path.join(primus_path, "examples/run_pretrain.sh")
    if not os.path.isfile(run_script):
        print(f"  ✗ Primus run script not found: {run_script}")
        return False
    
    # Build training command
    cmd = [
        "bash", run_script,
        "--train_iters", "500",
        "--lr", "0.0003",
        "--min_lr", "0.00003",
        "--lr_warmup_iters", "50",
        "--lr_decay_style", "cosine",
        "--lr_decay_iters", "500",
        "--weight_decay", "0.1",
    ]
    
    # Run training with tee to capture logs
    try:
        with open(log_file, 'w') as log_f, open(backup_log, 'w') as backup_f:
            process = subprocess.Popen(
                cmd,
                cwd=primus_path,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            # Stream output to both console and log files
            for line in process.stdout:
                print(line, end='')
                log_f.write(line)
                log_f.flush()
                backup_f.write(line)
                backup_f.flush()
            
            process.wait()
            exit_code = process.returncode
        
        if exit_code == 0:
            print("\n  ✓ Training completed successfully")
            print(f"  * Logs saved to: {log_file}")
            
            # Extract metrics
            print("\n  * Extracting metrics...")
            extract_cmd = [
                "python3", "extract_primus_metrics.py",
                "--log-file", log_file,
                "--model-name", model_name,
                "--output", os.path.join(output_dir, f"benchmark_rocm_{model_name}.json"),
                "--num-gpus", "8",
                "--global-batch-size", "128",
                "--sequence-length", "2048",
                "--parallel-strategy", "minimal_communication",
            ]
            
            try:
                result = subprocess.run(extract_cmd, cwd=os.path.dirname(__file__), check=True)
                print(f"  ✓ Metrics extracted successfully")
                print(f"  * Results saved to: {output_dir}/benchmark_rocm_{model_name}.json")
                return True
            except subprocess.CalledProcessError as e:
                print(f"  ✗ Metric extraction failed: {e}")
                return False
        else:
            print(f"\n  ✗ Training failed with exit code {exit_code}")
            print(f"  * Check log file: {log_file}")
            return False
            
    except Exception as e:
        print(f"\n  ✗ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_rocm():
    """Check if ROCm is available"""
    try:
        import torch
        if not torch.cuda.is_available():
            print("ERROR: CUDA/ROCm not available!")
            return False
        
        is_rocm = hasattr(torch.version, 'hip') and torch.version.hip is not None
        if not is_rocm:
            print("WARNING: Detected CUDA instead of ROCm. This script is for AMD GPUs.")
            response = input("Continue anyway? (y/n): ")
            if response.lower() != 'y':
                return False
        
        gpu_count = torch.cuda.device_count()
        print(f"Detected {gpu_count} GPU(s)")
        
        if gpu_count != 8:
            print(f"WARNING: Expected 8 GPUs, found {gpu_count}")
        
        for i in range(min(gpu_count, 8)):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        
        print()
        return True
        
    except ImportError:
        print("ERROR: PyTorch not found")
        return False


def main():
    """Main entry point - runs both models sequentially"""
    print("=" * 60)
    print("AMD/ROCm LLM Training Benchmark - Standalone")
    print("=" * 60)
    print("Hardware: 8x AMD MI300X (192GB each)")
    print("Framework: Primus (ROCm)")
    print("Strategy: Minimal Communication (TP=1, PP=1, DP=8)")
    print("Batch size: 128 (micro=1, grad_accum=16)")
    print("Sequence length: 2048 tokens")
    print("Training steps: 500")
    print("Precision: FP8 Hybrid")
    print()
    
    # Check ROCm availability
    if not check_rocm():
        sys.exit(1)
    
    # Get Primus path
    primus_path = os.environ.get('PRIMUS_PATH', '/workspace/Primus')
    print(f"Primus path: {primus_path}")
    
    if not os.path.isdir(primus_path):
        print(f"\nERROR: Primus not found at {primus_path}")
        print("Please set PRIMUS_PATH environment variable:")
        print("  export PRIMUS_PATH=/path/to/Primus")
        sys.exit(1)
    
    # Output directory
    output_dir = "./output"
    print(f"Output directory: {output_dir}")
    print()
    
    # Train both models
    results = {}
    models = ["llama", "qwen"]
    
    for model in models:
        success = train_model(model, primus_path, output_dir)
        results[model] = success
        print()
    
    # Summary
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    
    successful = [m for m, s in results.items() if s]
    failed = [m for m, s in results.items() if not s]
    
    if successful:
        print(f"✓ Successful: {len(successful)}")
        for model in successful:
            print(f"  - {model.upper()}")
    
    if failed:
        print(f"✗ Failed: {len(failed)}")
        for model in failed:
            print(f"  - {model.upper()}")
    
    print()
    
    if failed:
        sys.exit(1)
    else:
        print("All models trained successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()
