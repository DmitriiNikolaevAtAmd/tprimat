#!/usr/bin/env python3
"""
Megatron-DeepSpeed Training
Combines Megatron-LM's model parallelism with DeepSpeed's memory optimizations
"""
import os
import sys
import subprocess
from pathlib import Path

# Force unbuffered output
os.environ['PYTHONUNBUFFERED'] = '1'
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)


def check_megatron_deepspeed():
    """Check if Megatron-DeepSpeed is available"""
    mgds_path = os.environ.get('MEGATRON_DEEPSPEED_PATH', '/workspace/Megatron-DeepSpeed')
    
    if not os.path.exists(mgds_path):
        print(f"[!] Megatron-DeepSpeed not found at {mgds_path}")
        print(f"   Please set MEGATRON_DEEPSPEED_PATH environment variable")
        print(f"   Or clone: git clone https://github.com/microsoft/Megatron-DeepSpeed.git")
        return None
    
    pretrain_script = Path(mgds_path) / "pretrain_gpt.py"
    if not pretrain_script.exists():
        print(f"[!] pretrain_gpt.py not found in {mgds_path}")
        return None
    
    return mgds_path


def get_model_args(model_name):
    """Get model-specific arguments"""
    if model_name == "llama":
        return {
            "model_name": "meta-llama/Llama-3.1-8B",
            "num_layers": 32,
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "seq_length": 2048,
            "max_position_embeddings": 8192,
            "vocab_size": 128256,
        }
    elif model_name == "qwen":
        return {
            "model_name": "Qwen/Qwen2.5-7B",
            "num_layers": 28,
            "hidden_size": 3584,
            "num_attention_heads": 28,
            "seq_length": 2048,
            "max_position_embeddings": 32768,
            "vocab_size": 151936,
        }
    else:
        raise ValueError(f"Unknown model: {model_name}")


def train_model(model_short_name):
    """Train model with Megatron-DeepSpeed"""
    mgds_path = check_megatron_deepspeed()
    if mgds_path is None:
        return False
    
    model_args = get_model_args(model_short_name)
    
    # Detect platform
    import torch
    is_rocm = hasattr(torch.version, 'hip') and torch.version.hip is not None
    platform = "amd" if is_rocm else "nvd"
    
    # Get configuration
    world_size = int(os.environ.get('WORLD_SIZE', torch.cuda.device_count()))
    
    # Parallelism settings (for 8 GPUs)
    tp_size = 2  # Tensor parallelism
    pp_size = 1  # Pipeline parallelism
    dp_size = world_size // (tp_size * pp_size)  # Data parallelism
    
    micro_batch = 1
    grad_accum = 8
    global_batch = micro_batch * grad_accum * dp_size
    
    print(f"Training {model_args['model_name']} with Megatron-DeepSpeed")
    print(f"  Platform: {platform.upper()}")
    print(f"  GPUs: {world_size}")
    print(f"  Parallelism: TP={tp_size}, PP={pp_size}, DP={dp_size}")
    print(f"  Batch: micro={micro_batch}, grad_accum={grad_accum}, global={global_batch}")
    print()
    
    # DeepSpeed config
    ds_config = {
        "train_batch_size": global_batch,
        "train_micro_batch_size_per_gpu": micro_batch,
        "gradient_accumulation_steps": grad_accum,
        "steps_per_print": 1,
        "gradient_clipping": 1.0,
        "bf16": {
            "enabled": True
        },
        "zero_optimization": {
            "stage": 1,  # ZeRO Stage 1 (optimizer state sharding only, compatible with model parallelism)
            "reduce_bucket_size": 5e8,
            "allgather_bucket_size": 5e8,
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
        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": 0.00003,
                "warmup_max_lr": 0.0003,
                "warmup_num_steps": 1
            }
        },
        "wall_clock_breakdown": False,
    }
    
    # Save DeepSpeed config
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    ds_config_file = output_dir / f"ds_config_mgds_{model_short_name}.json"
    import json
    with open(ds_config_file, 'w') as f:
        json.dump(ds_config, f, indent=2)
    
    # Check if real data is available
    dataset_path = "/data/llama_dataset_text_document"
    use_real_data = os.path.exists(dataset_path + ".idx")
    
    if use_real_data:
        print(f"Using real data: {dataset_path}")
        data_args = [
            "--data-path", dataset_path,
        ]
    else:
        print("Real data not found, using mock data")
        # For benchmarking without real data, we'll create a small mock dataset
        # In production, you'd need actual preprocessed data
        data_args = [
            "--mock-data",  # This may not be supported, adjust as needed
        ]
    
    # Megatron-DeepSpeed command
    cmd = [
        "deepspeed",
        "--num_gpus", str(world_size),
        "--num_nodes", "1",
        os.path.join(mgds_path, "pretrain_gpt.py"),
        # Model architecture
        "--num-layers", str(model_args["num_layers"]),
        "--hidden-size", str(model_args["hidden_size"]),
        "--num-attention-heads", str(model_args["num_attention_heads"]),
        "--seq-length", str(model_args["seq_length"]),
        "--max-position-embeddings", str(model_args["max_position_embeddings"]),
        # Training config
        "--micro-batch-size", str(micro_batch),
        "--global-batch-size", str(global_batch),
        "--train-iters", "500",
        # Parallelism
        "--tensor-model-parallel-size", str(tp_size),
        "--pipeline-model-parallel-size", str(pp_size),
        # Optimizer
        "--lr", "0.0003",
        "--lr-decay-style", "cosine",
        "--min-lr", "0.00003",
        "--lr-warmup-iters", "1",
        "--weight-decay", "0.1",
        "--clip-grad", "1.0",
        # Precision
        "--bf16",
        # DeepSpeed
        "--deepspeed",
        "--deepspeed_config", str(ds_config_file),
        "--zero-stage", "1",
        # Checkpointing (disabled for benchmarking)
        "--no-save-optim",
        "--no-save-rng",
        # Logging
        "--log-interval", "1",
        "--tensorboard-dir", str(output_dir / "tensorboard"),
    ] + data_args
    
    print(f"Running: {' '.join(cmd)}")
    print()
    
    # Run training
    try:
        result = subprocess.run(cmd, check=True)
        print(f"\n✓ Megatron-DeepSpeed {model_short_name} training completed!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Training failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print("\n✗ DeepSpeed launcher not found. Is DeepSpeed installed?")
        print("   Install with: pip install deepspeed")
        return False


def main():
    os.makedirs("./output", exist_ok=True)
    
    import torch
    if not torch.cuda.is_available():
        print("CUDA is not available!")
        sys.exit(1)
    
    if len(sys.argv) < 2:
        print("Usage: python train_all_mgds.py [model]")
        print("  model: 'llama' or 'qwen'")
        sys.exit(1)
    
    model = sys.argv[1]
    if model not in ["llama", "qwen"]:
        print(f"Unknown model: {model}")
        print("Usage: python train_all_mgds.py [model]")
        print("  model: 'llama' or 'qwen'")
        sys.exit(1)
    
    success = train_model(model)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
