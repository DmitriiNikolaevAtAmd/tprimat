from nemo.collections import llm
import nemo_run as run
from benchmark_utils import BenchmarkCallback
import os

def run_pretrain():
    # Set PyTorch memory allocator for better fragmentation handling
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # 1. Initialize the recipe for Mixtral 8x7B
    recipe = llm.mixtral_8x7b.pretrain_recipe(
        name="mixtral_8x7b_pretrain_fp8",
        dir="/checkpoints",
        num_nodes=1,
        num_gpus_per_node=8,
    )
    
    # 2. PARALLELISM CONFIGURATION
    # Mixtral 8x7B is a Mixture of Experts (MoE) model
    # TP (4) * PP (1) = 4 GPUs per model instance
    # Total GPUs (8) / 4 = 2-way Data Parallelism
    # Expert Parallelism can be enabled for better efficiency
    recipe.trainer.strategy.tensor_model_parallel_size = 4
    recipe.trainer.strategy.pipeline_model_parallel_size = 1
    recipe.trainer.strategy.expert_model_parallel_size = 1  # Can increase for MoE
    
    # 3. DATA CONFIGURATION
    # Global Batch Size (128) / Data Parallel (2) = 64 samples per DP group
    # With Micro Batch Size = 1, this means 64 accumulation steps
    recipe.data.micro_batch_size = 1
    recipe.data.global_batch_size = 128
    recipe.data.seq_length = 2048
    
    # 4. OPTIMIZATIONS & DURATION
    recipe.trainer.max_steps = 10
    recipe.model.config.fp8 = "hybrid"  
    recipe.model.config.fp8_param = True
    
    # 5. DISABLE ALL CHECKPOINTING AND INTERMEDIATE SAVES
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
    
    # 6. ADD BENCHMARK CALLBACK FOR AMD vs NVIDIA COMPARISON
    # This only saves lightweight JSON logs to ./output/
    benchmark_callback = BenchmarkCallback(
        output_dir="./output",
        platform="auto",  # Auto-detects CUDA or ROCm
        model_name="mixtral"
    )
    if recipe.trainer.callbacks is None:
        recipe.trainer.callbacks = []
    recipe.trainer.callbacks.append(benchmark_callback)
    
    # 7. EXECUTE
    run.run(recipe, direct=True)

if __name__ == "__main__":
    run_pretrain()
