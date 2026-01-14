from nemo.collections import llm
import nemo_run as run
from utils import BenchmarkCallback
from config_loader import load_config
import os

def run_pretrain():
    # Load configuration
    config = load_config()
    
    # Detect platform
    import torch
    import random
    import numpy as np
    is_rocm = hasattr(torch.version, 'hip') and torch.version.hip is not None
    platform = "amd" if is_rocm else "nvidia"
    
    # Set random seed for reproducibility (must be done before recipe creation)
    seed = config.training.general.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # Set deterministic behavior for reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Get model name and parallelism settings from config
    model_name = "llama"
    # Allow overriding parallelism strategy via environment variable (for different configurations)
    parallel_strategy = os.environ.get('TPRIMAT_PARALLEL', config.get_methodology())
    print(f"🔧 Using parallelism strategy: {parallel_strategy}")
    parallelism = config.get_parallelism(model_name, platform, methodology=parallel_strategy)
    platform_opts = config.get_platform_optimizations(platform)
    
    # Set PyTorch memory allocator for better fragmentation handling
    if 'cuda_alloc_conf' in platform_opts:
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = platform_opts['cuda_alloc_conf']
    else:
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # 1. Initialize the recipe
    recipe = llm.llama31_8b.pretrain_recipe(
        name=f"{config.models.llama.name}_pretrain",
        dir="/checkpoints",
        num_nodes=1,
        num_gpus_per_node=config.hardware.platforms[platform].num_gpus,
    )
    
    # 2. PARALLELISM CONFIGURATION (from config.yaml)
    # TP * PP * DP = num_gpus
    print(f"🔧 Parallelism: TP={parallelism['tensor_model_parallel_size']}, "
          f"PP={parallelism['pipeline_model_parallel_size']}, "
          f"DP={parallelism['data_parallel_size']}, "
          f"GradAccum={parallelism['gradient_accumulation_steps']}")
    recipe.trainer.strategy.tensor_model_parallel_size = parallelism['tensor_model_parallel_size']
    recipe.trainer.strategy.pipeline_model_parallel_size = parallelism['pipeline_model_parallel_size']
    
    # Explicitly set trainer devices to match config
    recipe.trainer.devices = config.hardware.platforms[platform].num_gpus
    recipe.trainer.num_nodes = 1
    
    # 3. DATA CONFIGURATION (from config.yaml)
    recipe.data.micro_batch_size = config.training.data.micro_batch_size
    recipe.data.global_batch_size = config.training.data.global_batch_size
    recipe.data.seq_length = config.training.data.seq_length
    
    # 4. OPTIMIZATIONS & DURATION (from config.yaml)
    recipe.trainer.max_steps = config.training.duration.max_steps
    
    # 5. OPTIMIZER & LEARNING RATE CONFIGURATION (from config.yaml)
    # Apply learning rate and warmup settings
    recipe.optim.config.lr = config.training.optimizer.learning_rate
    recipe.optim.config.min_lr = config.training.optimizer.learning_rate * 0.1  # Decay to 10% of peak LR
    recipe.optim.config.weight_decay = config.training.optimizer.weight_decay
    recipe.optim.config.adam_beta1 = config.training.optimizer.beta1
    recipe.optim.config.adam_beta2 = config.training.optimizer.beta2
    
    # Configure warmup scheduler (it's lr_scheduler, not sched)
    recipe.optim.lr_scheduler.warmup_steps = config.training.optimizer.warmup_steps
    recipe.optim.lr_scheduler.constant_steps = 0  # No constant phase, go straight to decay after warmup
    
    # Apply platform-specific precision settings
    if platform_opts.get('fp8_hybrid'):
        recipe.model.config.fp8 = "hybrid"
        recipe.model.config.fp8_param = platform_opts.get('fp8_param', True)
    
    # Apply activation checkpointing if configured
    if platform_opts.get('activation_checkpointing'):
        # NeMo handles this automatically based on model config
        pass
    
    # 6. DISABLE ALL CHECKPOINTING AND INTERMEDIATE SAVES
    # Only logs and benchmark profiles will be saved
    recipe.trainer.enable_checkpointing = False
    recipe.log.ckpt = None  # Disable checkpoint callback
    recipe.resume = None    # No resume from checkpoint
    
    # Disable TensorBoard and other loggers (keep only benchmark)
    recipe.log.tensorboard = None
    recipe.log.wandb = None
    
    # Set validation to None to avoid validation checkpoints
    recipe.trainer.val_check_interval = None
    recipe.trainer.check_val_every_n_epoch = None
    
    # 7. ADD BENCHMARK CALLBACK (configured from config.yaml)
    benchmark_callback = BenchmarkCallback(
        output_dir=config.get_output_dir(),
        platform="auto",  # Auto-detects CUDA or ROCm
        model_name=model_name,
        parallel_strategy=parallel_strategy
    )
    if recipe.trainer.callbacks is None:
        recipe.trainer.callbacks = []
    recipe.trainer.callbacks.append(benchmark_callback)
    
    # 8. EXECUTE
    run.run(recipe, direct=True)

if __name__ == "__main__":
    run_pretrain()

