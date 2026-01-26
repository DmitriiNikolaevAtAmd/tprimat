#!/usr/bin/env python3
from nemo.collections import llm
import nemo_run as run
from utils import BenchmarkCallback
from config_loader import load_config
import os
import sys


def get_model_config(model_name: str, config):
    configs = {
        "llama": {
            "display_name": "Llama 3.1 8B",
            "recipe_fn": llm.llama31_8b.train_recipe,
            "config_obj": config.models.llama,
        },
        "qwen": {
            "display_name": "Qwen 2.5 7B",
            "recipe_fn": llm.qwen25_7b.train_recipe,
            "config_obj": config.models.qwen,
        }
    }
    
    if model_name not in configs:
        print(f"[!] Unknown model: {model_name}. Supported: {list(configs.keys())}")
        sys.exit(1)
    
    return configs[model_name]


def run_pretrain(model_name: str):
    config = load_config()
    
    import torch
    import random
    import numpy as np
    is_rocm = hasattr(torch.version, 'hip') and torch.version.hip is not None
    platform = "amd" if is_rocm else "nvidia"
    seed = config.training.general.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    model_cfg = get_model_config(model_name, config)
    print(f"  * Training model: {model_cfg['display_name']}")
    parallel_strategy = os.environ.get('PARALLEL', config.get_methodology())
    print(f"  * Using parallelism strategy: {parallel_strategy}")
    parallelism = config.get_parallelism(model_name, platform, methodology=parallel_strategy)
    platform_opts = config.get_platform_optimizations(platform)
    if 'cuda_alloc_conf' in platform_opts:
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = platform_opts['cuda_alloc_conf']
    else:
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    num_gpus = config.hardware.platforms[platform].num_gpus
    tp_size = parallelism['tensor_model_parallel_size']
    pp_size = parallelism['pipeline_model_parallel_size']
    
    recipe = model_cfg['recipe_fn'](
        name=f"{model_cfg['config_obj'].name}_pretrain",
        dir=config.paths.nemo.data_dir,
        num_nodes=1,
        num_gpus_per_node=num_gpus,
    )
    print(f"  * Parallelism: TP={tp_size}, "
          f"PP={pp_size}, "
          f"DP={parallelism['data_parallel_size']}, "
          f"GradAccum={parallelism['gradient_accumulation_steps']}")
    from nemo.lightning import MegatronStrategy
    recipe.trainer.strategy = MegatronStrategy(
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=pp_size,
        context_parallel_size=1,
        virtual_pipeline_model_parallel_size=None,
        sequence_parallel=False,
    )
    dataset_path = None
    tokenizer_path = None
    if hasattr(model_cfg['config_obj'], 'dataset_path') and model_cfg['config_obj'].dataset_path:
        dataset_path = model_cfg['config_obj'].dataset_path
        tokenizer_path = model_cfg['config_obj'].tokenizer_path
    elif hasattr(config.training.data, 'dataset_path') and config.training.data.dataset_path:
        dataset_path = config.training.data.dataset_path
        tokenizer_path = config.training.data.tokenizer_path
    if dataset_path and tokenizer_path:
        print(f"  * Using real data: {dataset_path}")
        from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer as NeMoAutoTokenizer
        print(f"  * Loading {model_cfg['display_name']} tokenizer from: {tokenizer_path}")
        tokenizer = NeMoAutoTokenizer(tokenizer_path)
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
        recipe.data.micro_batch_size = config.training.data.micro_batch_size
        recipe.data.global_batch_size = config.training.data.global_batch_size
        recipe.data.seq_length = config.training.data.seq_length
    recipe.trainer.max_steps = 50
    recipe.optim.config.lr = config.training.optimizer.learning_rate
    recipe.optim.config.min_lr = config.training.optimizer.learning_rate * 0.1
    recipe.optim.config.weight_decay = config.training.optimizer.weight_decay
    recipe.optim.config.adam_beta1 = config.training.optimizer.beta1
    recipe.optim.config.adam_beta2 = config.training.optimizer.beta2
    recipe.optim.lr_scheduler.warmup_steps = 10
    recipe.optim.lr_scheduler.constant_steps = 0
    if platform_opts.get('fp8_hybrid'):
        recipe.model.config.fp8 = "hybrid"
        recipe.model.config.fp8_param = platform_opts.get('fp8_param', True)
    if platform_opts.get('activation_checkpointing'):
        pass
    recipe.trainer.enable_checkpointing = False
    recipe.log.ckpt = None
    recipe.resume = None
    recipe.log.tensorboard = None
    recipe.log.wandb = None
    recipe.trainer.val_check_interval = None
    recipe.trainer.check_val_every_n_epoch = None
    benchmark_callback = BenchmarkCallback(
        output_dir=config.get_output_dir(),
        platform="auto",  # Auto-detects CUDA or ROCm
        model_name=model_name,
        parallel_strategy=parallel_strategy
    )
    if recipe.trainer.callbacks is None:
        recipe.trainer.callbacks = []
    recipe.trainer.callbacks.append(benchmark_callback)
    print(f"  * Starting {model_cfg['display_name']} training...")
    run.run(recipe, direct=True)
    print(f"  * {model_cfg['display_name']} training completed!")


def main():
    if len(sys.argv) < 2:
        print("No model specified, training both llama and qwen")
        models_to_train = ["llama", "qwen"]
    else:
        model_name = sys.argv[1].lower()
        models_to_train = [model_name]
    for model_name in models_to_train:
        run_pretrain(model_name)


if __name__ == "__main__":
    main()
