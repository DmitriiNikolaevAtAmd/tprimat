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
    parallel_strategy = os.environ.get('PARALLEL', config.get_methodology())
    print(f"  * Using parallelism strategy: {parallel_strategy}")
    parallelism = config.get_parallelism(model_name, platform, methodology=parallel_strategy)
    platform_opts = config.get_platform_optimizations(platform)
    
    # Set PyTorch memory allocator for better fragmentation handling
    if 'cuda_alloc_conf' in platform_opts:
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = platform_opts['cuda_alloc_conf']
    else:
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # 1. Initialize the recipe
    num_gpus = config.hardware.platforms[platform].num_gpus
    tp_size = parallelism['tensor_model_parallel_size']
    pp_size = parallelism['pipeline_model_parallel_size']
    
    recipe = llm.llama31_8b.train_recipe(
        name=f"{config.models.llama.name}_pretrain",
        dir=config.paths.nemo.data_dir,
        num_nodes=1,
        num_gpus_per_node=num_gpus,
    )
    
    # 2. PARALLELISM CONFIGURATION (from config.yaml)
    print(f"  * Parallelism: TP={tp_size}, "
          f"PP={pp_size}, "
          f"DP={parallelism['data_parallel_size']}, "
          f"GradAccum={parallelism['gradient_accumulation_steps']}")
    
    # Import MegatronStrategy to properly configure parallelism
    from nemo.lightning import MegatronStrategy
    
    # Create a new strategy with the desired parallelism
    recipe.trainer.strategy = MegatronStrategy(
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=pp_size,
        context_parallel_size=1,
        virtual_pipeline_model_parallel_size=None,
        sequence_parallel=False,
    )
    
    # 3. DATA CONFIGURATION (from config.yaml)
    # Check for model-specific data paths first, then fall back to global paths
    dataset_path = None
    tokenizer_path = None
    
    # Priority 1: Model-specific paths
    if hasattr(config.models.llama, 'dataset_path') and config.models.llama.dataset_path:
        dataset_path = config.models.llama.dataset_path
        tokenizer_path = config.models.llama.tokenizer_path
    # Priority 2: Global training data paths
    elif hasattr(config.training.data, 'dataset_path') and config.training.data.dataset_path:
        dataset_path = config.training.data.dataset_path
        tokenizer_path = config.training.data.tokenizer_path
    
    # Check if real data paths are provided - need to replace MockDataModule with PreTrainingDataModule
    if dataset_path and tokenizer_path:
        print(f"  * Using real data: {dataset_path}")
        
        # Use NeMo's AutoTokenizer wrapper for Llama 3.1 (has unique_identifiers attribute)
        # The tokenizer must match what was used to preprocess the data
        from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer as NeMoAutoTokenizer
        print(f"  * Loading Llama 3.1 tokenizer from: {tokenizer_path}")
        tokenizer = NeMoAutoTokenizer(tokenizer_path)
        
        # Replace MockDataModule with PreTrainingDataModule for real data
        from nemo.collections.llm.gpt.data.pre_training import PreTrainingDataModule
        recipe.data = PreTrainingDataModule(
            paths=[dataset_path],
            seq_length=config.training.data.seq_length,
            micro_batch_size=config.training.data.micro_batch_size,
            global_batch_size=config.training.data.global_batch_size,
            tokenizer=tokenizer,
            num_workers=2,
        )
    else:
        # Use default MockDataModule for benchmarking
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
        parallel_strategy=parallel_strategy,
        profiler_config=config.get_profiler_config()
    )
    if recipe.trainer.callbacks is None:
        recipe.trainer.callbacks = []
    recipe.trainer.callbacks.append(benchmark_callback)
    
    # 8. EXECUTE
    run.run(recipe, direct=True)

if __name__ == "__main__":
    import sys
    import shutil
    import subprocess
    
    # Check if we should wrap with Nsight profiler
    config = load_config()
    profiler_config = config.get_profiler_config()
    
    # Only wrap if: profiling enabled and not already running under nsys
    if (profiler_config.get('enabled', False) and 
        os.environ.get('NSYS_PROFILING_SESSION_ID') is None):
        
        # Check if nsys is available
        if shutil.which('nsys') is None:
            print("[!] Nsight profiling enabled but 'nsys' not found in PATH")
            print("   Running without profiling...")
            run_pretrain()
        else:
            # Build profile output path
            parallel_strategy = os.environ.get('PARALLEL', config.get_methodology())
            output_dir = config.get_output_dir()
            profile_output = os.path.join(output_dir, f"profile_cuda_llama_{parallel_strategy}")
            
            # Get nsight config
            trace = profiler_config.get('trace', 'cuda,nvtx,osrt,cudnn,cublas')
            cuda_memory = 'true' if profiler_config.get('cuda_memory_usage', True) else 'false'
            capture_range = profiler_config.get('capture_range', 'cudaProfilerApi')
            stats = 'true' if profiler_config.get('stats', True) else 'false'
            
            print(f"  * NVIDIA Nsight Systems profiling enabled")
            print(f"   Output: {profile_output}.nsys-rep")
            print()
            
            # Re-launch with nsys wrapper
            nsys_cmd = [
                'nsys', 'profile',
                '-o', profile_output,
                '--trace', trace,
                f'--cuda-memory-usage={cuda_memory}',
                '--capture-range', capture_range,
                f'--stats={stats}',
                '--force-overwrite=true',
                '--', sys.executable, __file__
            ]
            
            # Pass through environment
            env = os.environ.copy()
            env['NSYS_PROFILING_SESSION_ID'] = '1'  # Prevent infinite recursion
            
            result = subprocess.run(nsys_cmd, env=env)
            sys.exit(result.returncode)
    else:
        run_pretrain()

