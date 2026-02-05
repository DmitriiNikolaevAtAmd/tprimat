#!/usr/bin/env python3
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

DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
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
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", 32))
TP = int(os.environ.get("TP", 1))
PP = int(os.environ.get("PP", 1))
DP = int(os.environ.get("DP", 4))
PROFILING = os.environ.get("PROFILING", "false").lower() == "true"
PROFILE_WAIT = int(os.environ.get("PROFILE_WAIT", 1))
PROFILE_WARMUP = int(os.environ.get("PROFILE_WARMUP", 1))
PROFILE_ACTIVE = int(os.environ.get("PROFILE_ACTIVE", 3))
PROFILE_REPEAT = int(os.environ.get("PROFILE_REPEAT", 1))
VERIFY_DATA = os.environ.get("VERIFY_DATA", "false").lower() == "true"
VERIFY_SAMPLES = int(os.environ.get("VERIFY_SAMPLES", 100))
VERIFY_FULL_SCAN = os.environ.get("VERIFY_FULL_SCAN", "false").lower() == "true"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True
)
logger = logging.getLogger(__name__)


class KinetoProfilerCallback(Callback):
    """Lightning callback for Kineto GPU profiling integrated with training loop."""
    
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
            trace_file = self.profile_dir / f"nvd_nemo_{self.model_name}_rank{rank}.json"
            logger.info(f"Exporting trace to {trace_file}")
            prof.export_chrome_trace(str(trace_file))
            
            # Also export stacks for flame graph generation
            stacks_file = self.profile_dir / f"nvd_nemo_{self.model_name}_rank{rank}_stacks.txt"
            try:
                prof.export_stacks(str(stacks_file), "self_cuda_time_total")
                logger.info(f"Exported CUDA stacks to {stacks_file}")
            except Exception as e:
                logger.warning(f"Could not export stacks: {e}")
            
            # Print summary table
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


def get_model_config(model_name: str):
    configs = {
        "llama": {
            "display_name": "Llama 3.1 8B",
            "recipe_fn": llm.llama31_8b.pretrain_recipe,
            "recipe_name": "llama31_8b_pretrain",
            "tokenizer_path": str(DATA_DIR / "tokenizers" / "llama"),
        },
        "qwen": {
            "display_name": "Qwen 2.5 7B",
            "recipe_fn": llm.qwen25_7b.pretrain_recipe,
            "recipe_name": "qwen25_7b_pretrain",
            "tokenizer_path": str(DATA_DIR / "tokenizers" / "qwen"),
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
    
    logger.info(f"Setting up {config['display_name']} training...")
    
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
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

    logger.info(f"Creating {config['display_name']} training recipe...")
    recipe = config['recipe_fn'](
        name=config['recipe_name'],
        dir="/data",
        num_nodes=1,
        num_gpus_per_node=num_gpus,
    )

    # Ensure model vocab matches tokenizer to avoid out-of-bounds embedding ids.
    recipe.model.config.vocab_size = tokenizer_vocab_size
    
    recipe.trainer.strategy = MegatronStrategy(
        tensor_model_parallel_size=TP,
        pipeline_model_parallel_size=PP,
        context_parallel_size=1,
        virtual_pipeline_model_parallel_size=None,
        sequence_parallel=False,
    )
    
    mega_dataset_path = str(DATA_DIR / "megatron" / "bookcorpus_text_sentence")
    idx_file = mega_dataset_path + ".idx"
    bin_file = mega_dataset_path + ".bin"
    if not os.path.exists(idx_file) or not os.path.exists(bin_file):
        raise FileNotFoundError(
            f"Data preparation not completed - dataset not found at {mega_dataset_path}\n"
            f"  Missing: {idx_file if not os.path.exists(idx_file) else ''} "
            f"{bin_file if not os.path.exists(bin_file) else ''}\n"
            f"  Run data preparation first: python prepare/fetch_deps.py && "
            f"python prepare/clean_data.py && python prepare/encode_data.py"
        )
    
    logger.info(f"Data validation passed: {mega_dataset_path}")
    logger.info("Using NeMo PreTrainingDataModule with BookCorpus dataset")

    if VERIFY_DATA:
        try:
            from prepare.verify_data import verify_dataset
            logger.info(
                "Verifying dataset tokens (samples=%d, full_scan=%s)",
                VERIFY_SAMPLES,
                VERIFY_FULL_SCAN,
            )
            ok = verify_dataset(
                mega_dataset_path,
                config["tokenizer_path"],
                VERIFY_SAMPLES,
                VERIFY_FULL_SCAN,
            )
            if not ok:
                logger.error("Dataset verification failed. Aborting training.")
                sys.exit(1)
        except Exception as e:
            logger.error("Dataset verification errored: %s", e)
            sys.exit(1)
    
    from nemo.collections.llm.gpt.data.pre_training import PreTrainingDataModule
    recipe.data = PreTrainingDataModule(
        paths=[mega_dataset_path],
        seq_length=SEQ_LEN,
        micro_batch_size=MBS,
        global_batch_size=GBS,
    )
    
    recipe.trainer.max_steps = TRAIN_ITERS
    recipe.optim.config.lr = LR
    recipe.optim.config.min_lr = 0.0
    recipe.optim.config.weight_decay = WEIGHT_DECAY
    recipe.optim.config.adam_beta1 = BETA1
    recipe.optim.config.adam_beta2 = BETA2
    recipe.optim.lr_scheduler.warmup_steps = WARMUP_STEPS
    recipe.optim.lr_scheduler.constant_steps = 0
    recipe.optim.lr_scheduler.max_steps = TRAIN_ITERS
    
    if FP8_HYBRID:
        recipe.model.config.fp8 = "hybrid"
    else:
        recipe.model.config.fp8 = None
    recipe.model.config.fp8_param = FP8_PARAM
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
        output_dir=str(OUTPUT_DIR),
        platform="nvd",
        model_name=model_name,
        parallel_strategy="minimal_communication",
        framework=f"{platform_prefix}_nemo"
    )
    if recipe.trainer.callbacks is None:
        recipe.trainer.callbacks = []
    recipe.trainer.callbacks.append(benchmark_callback)
    
    # Add Kineto profiler callback if profiling is enabled
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
    logger.info(f"  Profiling: {PROFILING}")
    logger.info(f"  Verify data: {VERIFY_DATA} (samples={VERIFY_SAMPLES}, full_scan={VERIFY_FULL_SCAN})")
    
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