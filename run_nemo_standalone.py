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
    
    recipe.trainer.strategy = MegatronStrategy(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        virtual_pipeline_model_parallel_size=None,
        sequence_parallel=False,
    )
    
    dataset_path = "/data/llama_dataset_text_document"
    tokenizer_path = "meta-llama/Llama-3.1-8B"
    
    if os.path.exists(dataset_path + ".idx"):
        from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer as NeMoAutoTokenizer
        tokenizer = NeMoAutoTokenizer(tokenizer_path)
        from nemo.collections.llm.gpt.data.pre_training import PreTrainingDataModule
        recipe.data = PreTrainingDataModule(
            paths=[dataset_path],
            seq_length=2048,
            micro_batch_size=1,
            global_batch_size=64,
            tokenizer=tokenizer,
            num_workers=2,
        )
    else:
        recipe.data.micro_batch_size = 1
        recipe.data.global_batch_size = 64
        recipe.data.seq_length = 2048
    
    recipe.trainer.max_steps = 10
    recipe.optim.config.lr = 0.0003
    recipe.optim.config.min_lr = 0.00003
    recipe.optim.config.weight_decay = 0.1
    recipe.optim.config.adam_beta1 = 0.9
    recipe.optim.config.adam_beta2 = 0.95
    recipe.optim.lr_scheduler.warmup_steps = 1
    recipe.optim.lr_scheduler.constant_steps = 0
    recipe.model.config.fp8 = "hybrid"
    recipe.model.config.fp8_param = True
    recipe.model.config.recompute_granularity = "selective"
    recipe.model.config.recompute_method = "uniform"
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
        model_name="llama",
        parallel_strategy="minimal_communication",
        profiler_config={"enabled": False}
    )
    if recipe.trainer.callbacks is None:
        recipe.trainer.callbacks = []
    recipe.trainer.callbacks.append(benchmark_callback)
    
    run.run(recipe, direct=True)


def train_qwen():
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    random.seed(42)
    os.environ['PYTHONHASHSEED'] = '42'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    recipe = llm.qwen25_7b.pretrain_recipe(
        name="qwen25_7b_pretrain",
        dir="/data",
        num_nodes=1,
        num_gpus_per_node=8,
    )
    
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
        from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer as NeMoAutoTokenizer
        tokenizer = NeMoAutoTokenizer(tokenizer_path)
        from nemo.collections.llm.gpt.data.pre_training import PreTrainingDataModule
        recipe.data = PreTrainingDataModule(
            paths=[dataset_path],
            seq_length=2048,
            micro_batch_size=1,
            global_batch_size=64,
            tokenizer=tokenizer,
            num_workers=2,
        )
    else:
        recipe.data.micro_batch_size = 1
        recipe.data.global_batch_size = 64
        recipe.data.seq_length=2048
    
    recipe.trainer.max_steps = 10
    recipe.optim.config.lr = 0.0003
    recipe.optim.config.min_lr = 0.00003
    recipe.optim.config.weight_decay = 0.1
    recipe.optim.config.adam_beta1 = 0.9
    recipe.optim.config.adam_beta2 = 0.95
    recipe.optim.lr_scheduler.warmup_steps = 1
    recipe.optim.lr_scheduler.constant_steps = 0
    recipe.model.config.fp8 = "hybrid"
    recipe.model.config.fp8_param = True
    recipe.model.config.recompute_granularity = "selective"
    recipe.model.config.recompute_method = "uniform"
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
    
    run.run(recipe, direct=True)


def main():
    os.makedirs("./output", exist_ok=True)
    
    if not torch.cuda.is_available():
        sys.exit(1)
    
    if len(sys.argv) > 1:
        model = sys.argv[1]
        if model == "llama":
            train_llama()
        elif model == "qwen":
            train_qwen()
    else:
        import subprocess
        import time
        
        subprocess.run([sys.executable, __file__, "llama"])
        time.sleep(10)
        subprocess.run([sys.executable, __file__, "qwen"])


if __name__ == "__main__":
    main()
