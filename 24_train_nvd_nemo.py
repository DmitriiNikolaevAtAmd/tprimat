#!/usr/bin/env python3
import os
import sys

os.environ['PYTHONUNBUFFERED'] = '1'
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import torch
import random
import numpy as np
import logging
from nemo.collections import llm
import nemo_run as run
from nemo.lightning import MegatronStrategy
from utils import BenchmarkCallback

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ],
    force=True
)
logger = logging.getLogger(__name__)


def load_env_file(env_path: str) -> None:
    if not os.path.exists(env_path):
        return
    with open(env_path, "r") as env_file:
        for line in env_file:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def ensure_hf_token_for_gated_repo(repo_id: str) -> None:
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if hf_token:
        os.environ.setdefault("HF_TOKEN", hf_token)
        os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", hf_token)
        return
    if repo_id.startswith("meta-llama/"):
        logger.error(
            "Missing Hugging Face token for gated repo: %s. "
            "Set HF_TOKEN (or HUGGINGFACE_HUB_TOKEN) in the environment or in secrets.env.",
            repo_id,
        )
        sys.exit(1)


def get_model_config(model_name: str):
    configs = {
        "llama": {
            "display_name": "Llama 3.1 8B",
            "recipe_fn": llm.llama31_8b.pretrain_recipe,
            "recipe_name": "llama31_8b_pretrain",
            "tokenizer_path": "meta-llama/Llama-3.1-8B",
        },
        "qwen": {
            "display_name": "Qwen 2.5 7B",
            "recipe_fn": llm.qwen25_7b.pretrain_recipe,
            "recipe_name": "qwen25_7b_pretrain",
            "tokenizer_path": "Qwen/Qwen2.5-7B",
        }
    }
    
    if model_name not in configs:
        logger.error(f"Unknown model: {model_name}. Supported: {list(configs.keys())}")
        sys.exit(1)
    
    return configs[model_name]


def train_model(model_name: str):
    load_env_file("secrets.env")
    os.makedirs("./output", exist_ok=True)
    
    if not torch.cuda.is_available():
        logger.error("CUDA is not available!")
        sys.exit(1)
    
    platform_prefix = "nvd"
    
    logger.info(f"CUDA devices available: {torch.cuda.device_count()}")
    
    config = get_model_config(model_name)
    ensure_hf_token_for_gated_repo(config["tokenizer_path"])
    
    logger.info(f"Setting up {config['display_name']} training...")
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    random.seed(42)
    os.environ['PYTHONHASHSEED'] = '42'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    logger.info(f"Creating {config['display_name']} training recipe...")
    recipe = config['recipe_fn'](
        name=config['recipe_name'],
        dir="/data",
        num_nodes=1,
        num_gpus_per_node=8,
    )
    
    recipe.trainer.strategy = MegatronStrategy(
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        virtual_pipeline_model_parallel_size=None,
        sequence_parallel=True,
    )
    
    dataset_path = f"/data/tprimat/allenai-c4-100k-{model_name}"
    
    if os.path.exists(dataset_path + ".idx"):
        logger.info(f"Using real data: {dataset_path}")
        from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer as NeMoAutoTokenizer
        tokenizer = NeMoAutoTokenizer(config['tokenizer_path'])
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
        logger.info("Using synthetic data for benchmarking")
        recipe.data.micro_batch_size = 1
        recipe.data.global_batch_size = 64
        recipe.data.seq_length = 2048
    
    recipe.trainer.max_steps = 50
    recipe.optim.config.lr = 0.0003
    recipe.optim.config.min_lr = 0.0
    recipe.optim.config.weight_decay = 0.1
    recipe.optim.config.adam_beta1 = 0.9
    recipe.optim.config.adam_beta2 = 0.95
    recipe.optim.lr_scheduler.warmup_steps = 10
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
        platform="nvd",
        model_name=model_name,
        parallel_strategy="minimal_communication",
        framework=f"{platform_prefix}_nemo"
    )
    if recipe.trainer.callbacks is None:
        recipe.trainer.callbacks = []
    recipe.trainer.callbacks.append(benchmark_callback)
    
    logger.info(f"Starting {config['display_name']} training...")
    run.run(recipe, direct=True)
    logger.info(f"{config['display_name']} training completed!")


def main():
    if len(sys.argv) < 2:
        logger.info("No model specified, training both llama and qwen")
        train_model("llama")
        train_model("qwen")
    else:
        model_name = sys.argv[1].lower()
        train_model(model_name)


if __name__ == "__main__":
    main()
