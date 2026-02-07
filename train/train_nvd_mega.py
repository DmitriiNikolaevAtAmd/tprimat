#!/usr/bin/env python3
import gc
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ['PYTHONUNBUFFERED'] = '1'
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import torch
import torch.profiler
import random
import numpy as np
import logging
from transformers import AutoTokenizer
from lightning.pytorch.callbacks import Callback
from nemo.collections import llm
import nemo_run as run
from nemo.lightning import MegatronStrategy
from lib.utils import BenchmarkCallback

DATA_DIR = Path(os.environ.get("DATA_DIR", "/data/tprimat"))
WORKSPACE_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", str(WORKSPACE_ROOT / "output")))

SEED = int(os.environ.get("SEED", 42))
MBS = int(os.environ.get("MBS", 1))
GBS = int(os.environ.get("GBS", 128))
SEQ_LEN = int(os.environ.get("SEQ_LEN", 2048))
LR = float(os.environ.get("LR", 3e-4))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", 0.1))
BETA1 = float(os.environ.get("BETA1", 0.9))
BETA2 = float(os.environ.get("BETA2", 0.95))
PRECISION = os.environ.get("PRECISION", "bf16")
FP8_HYBRID = os.environ.get("FP8_HYBRID", "false").lower() == "true"
FP8_PARAM = os.environ.get("FP8_PARAM", "false").lower() == "true"
WARMUP_STEPS = int(os.environ.get("WARMUP_STEPS", 50))
TRAIN_ITERS = int(os.environ.get("TRAIN_ITERS", 10))
GA = int(os.environ.get("GA", 32))
TP = int(os.environ.get("TP", 1))
PP = int(os.environ.get("PP", 1))
DP = int(os.environ.get("DP", 4))
DATASET = os.environ.get("DATASET", "bc")

PROFILING = os.environ.get("PROFILING", "false").lower() == "true"
PROFILE_WAIT = int(os.environ.get("PROFILE_WAIT", 1))
PROFILE_WARMUP = int(os.environ.get("PROFILE_WARMUP", 1))
PROFILE_ACTIVE = int(os.environ.get("PROFILE_ACTIVE", 3))
PROFILE_REPEAT = int(os.environ.get("PROFILE_REPEAT", 1))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True
)
logger = logging.getLogger(__name__)


class GCCallback(Callback):

    def on_train_start(self, trainer, pl_module):
        gc.disable()
        logger.info("GC disabled for training (avoiding step-time spikes)")

    def on_train_end(self, trainer, pl_module):
        gc.enable()
        gc.collect()
        logger.info("GC re-enabled after training")


class KinetoProfilerCallback(Callback):

    def __init__(self, output_dir: Path, model_name: str):
        self.output_dir = output_dir
        self.model_name = model_name
        self.profiler = None
        self.profile_dir = output_dir / "profiles"

    def on_train_start(self, trainer, pl_module):
        if not PROFILING:
            return

        self.profile_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Kineto GPU profiling enabled")
        logger.info(f"  Profile output: {self.profile_dir}")
        logger.info(f"  Schedule: wait={PROFILE_WAIT}, warmup={PROFILE_WARMUP}, "
                    f"active={PROFILE_ACTIVE}, repeat={PROFILE_REPEAT}")

        def trace_handler(prof):
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            trace_file = self.profile_dir / f"nvd_mega_{self.model_name}_rank{rank}.json"
            logger.info(f"Exporting trace to {trace_file}")
            prof.export_chrome_trace(str(trace_file))

            stacks_file = self.profile_dir / f"nvd_mega_{self.model_name}_rank{rank}_stacks.txt"
            try:
                prof.export_stacks(str(stacks_file), "self_cuda_time_total")
                logger.info(f"Exported CUDA stacks to {stacks_file}")
            except Exception as e:
                logger.warning(f"Could not export stacks: {e}")

            if trainer.is_global_zero:
                logger.info("\n" + prof.key_averages().table(
                    sort_by="cuda_time_total", row_limit=20
                ))

        self.profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=PROFILE_WAIT,
                warmup=PROFILE_WARMUP,
                active=PROFILE_ACTIVE,
                repeat=PROFILE_REPEAT,
            ),
            on_trace_ready=trace_handler,
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=True,
            with_modules=True,
        )
        self.profiler.__enter__()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.profiler is not None:
            self.profiler.step()

    def on_train_end(self, trainer, pl_module):
        if self.profiler is not None:
            self.profiler.__exit__(None, None, None)
            self.profiler = None
            logger.info(f"Kineto profiling completed. Traces saved to {self.profile_dir}")


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


def get_tokenizer_path(model_name: str) -> str:
    local_paths = {
        "llama": str(DATA_DIR / "tokenizers" / "llama"),
        "qwen": str(DATA_DIR / "tokenizers" / "qwen"),
    }
    hf_paths = {
        "llama": "meta-llama/Llama-3.1-8B",
        "qwen": "Qwen/Qwen2.5-7B",
    }

    local_path = local_paths.get(model_name)
    if local_path and os.path.isdir(local_path):
        logger.info(f"Using local tokenizer: {local_path}")
        return local_path

    hf_path = hf_paths.get(model_name)
    if hf_path:
        logger.info(f"Using HuggingFace tokenizer: {hf_path}")
        return hf_path

    raise ValueError(f"Unknown model for tokenizer: {model_name}")


def get_model_config(model_name: str):
    tokenizer_path = get_tokenizer_path(model_name)

    configs = {
        "llama": {
            "display_name": "Llama 3.1 8B",
            "recipe_fn": llm.llama31_8b.pretrain_recipe,
            "recipe_name": "llama31_8b_pretrain",
            "tokenizer_path": tokenizer_path,
        },
        "qwen": {
            "display_name": "Qwen 2.5 7B",
            "recipe_fn": llm.qwen25_7b.pretrain_recipe,
            "recipe_name": "qwen25_7b_pretrain",
            "tokenizer_path": tokenizer_path,
        }
    }

    if model_name not in configs:
        logger.error(f"Unknown model: {model_name}. Supported: {list(configs.keys())}")
        sys.exit(1)

    return configs[model_name]


def train_model(model_name: str):
    load_env_file("secrets.env")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        logger.error("CUDA is not available!")
        sys.exit(1)

    platform_prefix = "nvd"
    num_gpus = torch.cuda.device_count()

    logger.info(f"CUDA devices available: {num_gpus}")

    config = get_model_config(model_name)
    ensure_hf_token_for_gated_repo(config["tokenizer_path"])

    logger.info(f"Setting up {config['display_name']} training (Megatron-Core stack)...")

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    logger.info(f"Loading tokenizer for {config['display_name']}...")
    tokenizer = AutoTokenizer.from_pretrained(
        config["tokenizer_path"],
        trust_remote_code=True,
    )
    tokenizer_vocab_size = len(tokenizer)
    base_vocab_size = getattr(tokenizer, "vocab_size", tokenizer_vocab_size)
    if tokenizer_vocab_size != base_vocab_size:
        logger.info(
            "Tokenizer vocab size differs from base: len=%d base=%d",
            tokenizer_vocab_size,
            base_vocab_size,
        )

    logger.info(f"Creating {config['display_name']} training recipe (Megatron-Core)...")
    recipe = config['recipe_fn'](
        name=config['recipe_name'],
        dir="/data",
        num_nodes=1,
        num_gpus_per_node=num_gpus,
    )

    recipe.model.config.vocab_size = tokenizer_vocab_size

    try:
        from megatron.core.distributed import DistributedDataParallelConfig
        ddp_config = DistributedDataParallelConfig(
            grad_reduce_in_fp32=False,
            overlap_grad_reduce=False,
            overlap_param_gather=False,
            use_distributed_optimizer=True,
        )
        logger.info("Using DDP config (distributed_optimizer, no overlap)")
    except ImportError:
        ddp_config = "megatron"
        logger.info("Megatron DDP config not available, using default")

    recipe.trainer.strategy = MegatronStrategy(
        tensor_model_parallel_size=TP,
        pipeline_model_parallel_size=PP,
        context_parallel_size=1,
        virtual_pipeline_model_parallel_size=None,
        sequence_parallel=(TP > 1),
        ddp=ddp_config,
    )

    dataset_name = DATASET
    train_dataset_path = str(DATA_DIR / f"{dataset_name}-train")
    test_dataset_path = str(DATA_DIR / f"{dataset_name}-test")

    train_idx = train_dataset_path + ".idx"
    train_bin = train_dataset_path + ".bin"
    if not os.path.exists(train_idx) or not os.path.exists(train_bin):
        raise FileNotFoundError(
            f"Training dataset not found at {train_dataset_path}\n"
            f"  Missing: {train_idx if not os.path.exists(train_idx) else ''} "
            f"{train_bin if not os.path.exists(train_bin) else ''}\n"
            f"  Run data preparation: python prepare/encode_data.py"
        )

    test_idx = test_dataset_path + ".idx"
    test_bin = test_dataset_path + ".bin"
    if not os.path.exists(test_idx) or not os.path.exists(test_bin):
        raise FileNotFoundError(
            f"Test dataset not found at {test_dataset_path}\n"
            f"  Missing: {test_idx if not os.path.exists(test_idx) else ''} "
            f"{test_bin if not os.path.exists(test_bin) else ''}\n"
            f"  Run data preparation: python prepare/encode_data.py"
        )

    logger.info(f"Train dataset: {train_dataset_path}")
    logger.info(f"Test dataset:  {test_dataset_path}")
    logger.info("Using separate train/test files (validation uses test dataset)")

    data_paths = {
        "train": [train_dataset_path],
        "validation": [test_dataset_path],
        "test": [test_dataset_path],
    }

    from nemo.collections.llm.gpt.data.pre_training import PreTrainingDataModule
    recipe.data = PreTrainingDataModule(
        paths=data_paths,
        seq_length=SEQ_LEN,
        micro_batch_size=MBS,
        global_batch_size=GBS,
    )

    _WARMUP_ITERS = 0
    recipe.trainer.max_steps = TRAIN_ITERS + _WARMUP_ITERS
    recipe.optim.config.lr = LR
    recipe.optim.config.min_lr = 0.0
    recipe.optim.config.weight_decay = WEIGHT_DECAY
    recipe.optim.config.adam_beta1 = BETA1
    recipe.optim.config.adam_beta2 = BETA2
    recipe.optim.lr_scheduler.warmup_steps = WARMUP_STEPS + _WARMUP_ITERS
    recipe.optim.lr_scheduler.constant_steps = 0
    recipe.optim.lr_scheduler.max_steps = TRAIN_ITERS + _WARMUP_ITERS
    recipe.optim.lr_scheduler.min_lr = 0.0

    if FP8_HYBRID:
        recipe.model.config.fp8 = "hybrid"
    else:
        recipe.model.config.fp8 = None
    recipe.model.config.fp8_param = FP8_PARAM
    recipe.model.config.recompute_granularity = None
    recipe.model.config.recompute_method = None

    recipe.model.config.bias_activation_fusion = True
    recipe.model.config.bias_dropout_fusion = True
    recipe.model.config.masked_softmax_fusion = True
    recipe.model.config.persist_layer_norm = True
    recipe.model.config.apply_rope_fusion = True
    recipe.model.config.cross_entropy_loss_fusion = True
    recipe.model.config.gradient_accumulation_fusion = False

    recipe.trainer.enable_checkpointing = False
    recipe.log.ckpt = None
    recipe.resume = None
    recipe.log.tensorboard = None
    recipe.log.wandb = None
    recipe.trainer.val_check_interval = TRAIN_ITERS + _WARMUP_ITERS + 1
    recipe.trainer.check_val_every_n_epoch = None
    recipe.trainer.limit_val_batches = 0
    recipe.trainer.num_sanity_val_steps = 0

    benchmark_callback = BenchmarkCallback(
        output_dir=str(OUTPUT_DIR),
        platform="nvd",
        model_name=model_name,
        parallel_strategy="minimal_communication",
        framework=f"{platform_prefix}_mega",
        dataset=DATASET,
        warmup_steps=_WARMUP_ITERS,
    )
    gc_callback = GCCallback()
    if recipe.trainer.callbacks is None:
        recipe.trainer.callbacks = []
    recipe.trainer.callbacks.append(benchmark_callback)
    recipe.trainer.callbacks.append(gc_callback)
    if PROFILING:
        profiler_callback = KinetoProfilerCallback(OUTPUT_DIR, model_name)
        recipe.trainer.callbacks.append(profiler_callback)

    logger.info(f"Configuration:")
    logger.info(f"  Sequence length: {SEQ_LEN}")
    logger.info(f"  Micro batch size: {MBS}")
    logger.info(f"  Global batch size: {GBS}")
    logger.info(f"  Training steps: {TRAIN_ITERS}")
    logger.info(f"  Learning rate: {LR}")
    logger.info(f"  Warmup steps: {WARMUP_STEPS}")
    logger.info(f"  Precision: {PRECISION}")
    logger.info(f"  FP8 Hybrid: {FP8_HYBRID}")
    logger.info(f"  FP8 Param: {FP8_PARAM}")
    logger.info(f"  TP: {TP}, PP: {PP}, DP: {DP}")
    logger.info(f"  Stack: Megatron-Core + TransformerEngine + Lightning")
    logger.info(f"  Fusions: bias_act, bias_drop, masked_softmax, persist_ln, rope, xent")

    logger.info(f"Starting {config['display_name']} training (Megatron-Core stack)...")

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
