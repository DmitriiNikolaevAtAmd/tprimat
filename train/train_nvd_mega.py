#!/usr/bin/env python3
import gc
import os
import sys
from pathlib import Path

# Resolve relative paths before importing transformers (which uses HF_HOME)
_workspace_root = Path(__file__).parent.parent
sys.path.insert(0, str(_workspace_root))
for _env_var in ("HF_HOME", "HF_DATASETS_CACHE"):
    _val = os.environ.get(_env_var)
    if _val and not os.path.isabs(_val):
        os.environ[_env_var] = str(_workspace_root / _val)

os.environ['PYTHONUNBUFFERED'] = '1'
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import torch
import random
import numpy as np
import logging
import json
import time
from pathlib import Path
from datetime import datetime
from contextlib import nullcontext

DATA_DIR = Path(os.environ.get("DATA_DIR", "/data/tprimat"))
WORKSPACE_ROOT = Path(__file__).parent.parent
_output_dir = os.environ.get("OUTPUT_DIR", "output")
OUTPUT_DIR = Path(_output_dir) if os.path.isabs(_output_dir) else WORKSPACE_ROOT / _output_dir

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
TRAIN_ITERS = int(os.environ.get("TRAIN_ITERS", 500))
GA = int(os.environ.get("GA", 8))
TP = int(os.environ.get("TP", 1))
PP = int(os.environ.get("PP", 1))
DP = int(os.environ.get("DP", 8))
DATASET = os.environ.get("DATASET", "bc")  # bc or c4

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Data pipeline
# ──────────────────────────────────────────────────────────────────────

class _TokenDataset(torch.utils.data.Dataset):
    """Map-style dataset wrapping IndexedDataset for use with DataLoader.

    The underlying IndexedDataset is opened lazily (per-process) so the
    class is safe for ``num_workers > 0`` via fork or spawn.
    """

    def __init__(self, data_path: str, seq_length: int, pad_id: int = 0):
        from lib.dataset import IndexedDataset
        self._ds = IndexedDataset(data_path)
        self._seq = seq_length
        self._pad = pad_id

    def __len__(self):
        return len(self._ds)

    def __getitem__(self, idx):
        t = self._ds[idx % len(self._ds)]
        n = len(t)
        if n >= self._seq:
            return t[: self._seq]
        out = torch.full((self._seq,), self._pad, dtype=torch.long)
        out[:n] = t
        return out


def _cycle(loader):
    """Yield batches forever, advancing the DistributedSampler epoch."""
    ep = 0
    while True:
        if hasattr(loader, "sampler") and hasattr(loader.sampler, "set_epoch"):
            loader.sampler.set_epoch(ep)
        yield from loader
        ep += 1


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def train_model(model_name: str, model_config: dict):
    logger.info(f"=" * 80)
    logger.info(f"Starting Mega-LM training for {model_name}")
    logger.info(f"=" * 80)

    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(
            backend="nccl",
            device_id=torch.device(f"cuda:{local_rank}"),
        )
        try:
            torch.distributed.barrier()
        except Exception:
            pass
        logger.info(f"Initialized distributed training: rank={rank}, world_size={world_size}")
    else:
        logger.info("Running in single-GPU mode")

    set_seed(SEED)
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    # ── Performance knobs ──────────────────────────────────────────────
    torch.set_float32_matmul_precision('high')          # TF32 matmuls
    torch.backends.cudnn.benchmark = True               # cuDNN autotuner
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # ── FIX: derive grad_accum from GBS so it matches NeMo ────────────
    # Previously GA was divided by world_size which collapsed GBS to
    # MBS*world_size (e.g. 8), while NeMo used the full GBS (e.g. 64).
    grad_accum = max(1, GBS // (MBS * world_size))
    global_batch_size = MBS * grad_accum * world_size
    logger.info(f"GBS={GBS} → grad_accum={grad_accum}, effective GBS={global_batch_size}")

    if torch.cuda.is_available():
        device_props = torch.cuda.get_device_properties(local_rank)
        device_name = torch.cuda.get_device_name(local_rank)
        platform = "nvd"
        software_stack = "megatron"
        software_version = torch.version.cuda if hasattr(torch.version, 'cuda') else "unknown"
        gpu_cores = 16896 if "h100" in device_name.lower() else 6912
        gpu_info = {
            "device_count": world_size,
            "device_name": device_name,
            "total_memory_gb": device_props.total_memory / 1e9,
            "gpu_cores": gpu_cores,
            "pytorch_version": torch.__version__,
            "software_stack": software_stack,
            "software_version": software_version,
        }
    else:
        platform = "cpu"
        gpu_info = {}

    step_times = []
    loss_values = []
    learning_rates = []

    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

        logger.info(f"Loading tokenizer: {model_config['hf_model']}")
        tokenizer = AutoTokenizer.from_pretrained(model_config['hf_model'], trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        logger.info(f"Initializing model: {model_config['hf_model']} (random weights)")
        torch_dtype = torch.bfloat16 if PRECISION == "bf16" else torch.float16 if PRECISION == "fp16" else torch.float32
        config = AutoConfig.from_pretrained(model_config['hf_model'], trust_remote_code=True)

        _device = torch.device(f'cuda:{local_rank}') if world_size > 1 else torch.device('cuda')

        # ── Attention: prefer flash_attention_2 (faster fused kernels),
        #    fall back to sdpa, then default.
        # Initialise directly on the target GPU via init_device to avoid
        # allocating a full CPU copy per rank that then races to .to(device).
        attn_impl_used = "flash_attention_2"
        try:
            with torch.device(_device):
                model = AutoModelForCausalLM.from_config(
                    config,
                    torch_dtype=torch_dtype,
                    attn_implementation="flash_attention_2",
                )
            logger.info("Using attention implementation: flash_attention_2")
        except Exception:
            try:
                with torch.device(_device):
                    model = AutoModelForCausalLM.from_config(
                        config,
                        torch_dtype=torch_dtype,
                        attn_implementation="sdpa",
                    )
                attn_impl_used = "sdpa"
                logger.info("flash_attention_2 not available, using sdpa")
            except Exception as attn_err:
                logger.warning(f"SDPA not available ({attn_err}), falling back to default")
                with torch.device(_device):
                    model = AutoModelForCausalLM.from_config(config, torch_dtype=torch_dtype)
                attn_impl_used = "default"
                logger.info("Using default attention implementation")
        model.config.use_cache = False

        # NOTE: gradient checkpointing intentionally disabled — with ZeRO-1
        # sharding, Llama-8B fits in H100 80 GB without recomputation.
        # Checkpointing halves activation memory but costs ~30-40 % throughput.

        # ── DDP ───────────────────────────────────────────────────────
        # NOTE: DDP must wrap the model BEFORE torch.compile so that
        # the autograd hooks for gradient all-reduce are part of the
        # compiled graph.  static_graph=True is also incompatible with
        # torch.compile — compile rewrites the autograd graph which
        # trips DDP's expect_autograd_hooks_ assertion in reducer.cpp.
        is_ddp = False
        if world_size > 1:
            from torch.nn.parallel import DistributedDataParallel as DDP
            model = DDP(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=False,
                gradient_as_bucket_view=True,
            )
            logger.info(f"Wrapped model with DDP (bucket_view)")
            is_ddp = True

        # ── torch.compile ─────────────────────────────────────────────
        # Disabled under DDP: HuggingFace flash-attention uses .item()
        # which forces graph breaks, and the resulting recompilation
        # storms add overhead instead of saving it.  Single-GPU still
        # benefits from reduce-overhead (CUDA-graph) mode.
        if not is_ddp:
            try:
                model = torch.compile(model, mode="reduce-overhead")
                logger.info("Enabled torch.compile (reduce-overhead)")
            except Exception as compile_err:
                logger.warning(f"torch.compile failed ({compile_err}), running eagerly")
        else:
            logger.info("Skipping torch.compile (incompatible with DDP + HF flash-attn graph breaks)")

        # ── Data ──────────────────────────────────────────────────────
        dataset_path = str(DATA_DIR / f"{DATASET}-train")
        idx_file = dataset_path + ".idx"
        bin_file = dataset_path + ".bin"
        if not os.path.exists(idx_file) or not os.path.exists(bin_file):
            raise FileNotFoundError(
                f"Dataset not found at {dataset_path}\n"
                f"  Missing: {idx_file if not os.path.exists(idx_file) else ''} "
                f"{bin_file if not os.path.exists(bin_file) else ''}\n"
                f"  Run data preparation: bash prepare/data.sh"
            )

        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        ds = _TokenDataset(dataset_path, SEQ_LEN, pad_id)
        logger.info(f"Dataset: {dataset_path} ({len(ds)} sequences)")

        sampler = None
        if world_size > 1:
            sampler = torch.utils.data.distributed.DistributedSampler(
                ds, num_replicas=world_size, rank=rank, shuffle=True,
            )
        loader = torch.utils.data.DataLoader(
            ds,
            batch_size=MBS,
            sampler=sampler,
            shuffle=(sampler is None),
            num_workers=2,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
            drop_last=True,
        )
        data_iter = iter(_cycle(loader))

        batch_size = MBS
        num_steps = TRAIN_ITERS

        # ── Optimizer ──────────────────────────────────────────────────
        # Llama-8B + full AdamW FP32 states ≈ 86 GB (params 14 + opt 57 + grads 14),
        # exceeding 80 GB per GPU. Use ZeroRedundancyOptimizer (ZeRO-1) to shard
        # optimizer states across the data-parallel group, reducing per-GPU opt
        # memory from ~57 GB to ~7 GB.
        if world_size > 1:
            from torch.distributed.optim import ZeroRedundancyOptimizer
            optimizer = ZeroRedundancyOptimizer(
                model.parameters(),
                optimizer_class=torch.optim.AdamW,
                lr=LR, betas=(BETA1, BETA2),
                eps=1e-8, weight_decay=WEIGHT_DECAY,
            )
            logger.info("Using ZeroRedundancyOptimizer (sharded AdamW, ZeRO-1)")
        else:
            try:
                optimizer = torch.optim.AdamW(
                    model.parameters(), lr=LR, betas=(BETA1, BETA2),
                    eps=1e-8, weight_decay=WEIGHT_DECAY, fused=True,
                )
                logger.info("Using fused AdamW optimizer")
            except Exception:
                optimizer = torch.optim.AdamW(
                    model.parameters(), lr=LR, betas=(BETA1, BETA2),
                    eps=1e-8, weight_decay=WEIGHT_DECAY,
                )
                logger.info("Using standard AdamW optimizer")

        from transformers import get_cosine_schedule_with_warmup
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=num_steps)

        logger.info(f"Configuration:")
        logger.info(f"  Sequence length: {SEQ_LEN}")
        logger.info(f"  Micro batch size: {batch_size}")
        logger.info(f"  Global batch size: {global_batch_size}")
        logger.info(f"  Gradient accumulation: {grad_accum}")
        logger.info(f"  Training steps: {num_steps}")
        logger.info(f"  Learning rate: {LR}")
        logger.info(f"  LR scheduler: Cosine with {WARMUP_STEPS} warmup steps")
        logger.info(f"  Precision: {PRECISION}")
        logger.info(f"  FP8 Hybrid: {FP8_HYBRID}")
        logger.info(f"  FP8 Param: {FP8_PARAM}")

        # ── FP8 setup (NVIDIA H100+ only) ─────────────────────────────
        fp8_enabled = False
        fp8_ctx_fn = nullcontext
        if FP8_HYBRID:
            try:
                import transformer_engine.pytorch as te
                fp8_recipe = te.recipe.DelayedScaling(
                    fp8_format=te.recipe.Format.HYBRID,
                    amax_history_len=4,
                    amax_compute_algo="most_recent",
                )
                fp8_ctx_fn = lambda: te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe)
                fp8_enabled = True
                logger.info("FP8 hybrid enabled via TransformerEngine")
            except ImportError:
                logger.warning("FP8 requested but transformer_engine not installed — running in BF16")
            except Exception as fp8_err:
                logger.warning(f"FP8 init failed ({fp8_err}) — running in BF16")

        model.train()

        # ── CUDA warmup (un-timed iterations to pre-compile kernels) ────
        # Running a few real forward+backward passes forces CUDA to JIT-
        # compile all kernels, warms the cuDNN autotuner, and allocates
        # memory pools.  This eliminates the large initial spike visible
        # in the first ~5 timed steps.
        _WARMUP_ITERS = 3
        logger.info(f"Running {_WARMUP_ITERS} warmup iterations to pre-compile CUDA kernels...")
        for _w in range(_WARMUP_ITERS):
            optimizer.zero_grad(set_to_none=True)
            for micro_step in range(grad_accum):
                input_ids = next(data_iter).to(_device, non_blocking=True)
                labels = input_ids
                sync_ctx = nullcontext()
                if is_ddp and micro_step < (grad_accum - 1):
                    sync_ctx = model.no_sync()
                with sync_ctx, fp8_ctx_fn():
                    with torch.amp.autocast('cuda', dtype=torch_dtype):
                        outputs = model(input_ids=input_ids, labels=labels)
                        loss = outputs.loss / grad_accum
                    loss.backward()
                del outputs, loss, input_ids, labels
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, foreach=True)
            optimizer.step()
            # NOTE: scheduler.step() intentionally NOT called during warmup
            # so the LR schedule starts cleanly at the first timed step.
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        logger.info("CUDA warmup complete, starting timed training...")

        logger.info("Starting training...")
        training_start = time.time()

        # Disable GC during training to avoid random pauses
        gc.disable()
        try:
            for step in range(num_steps):
                step_start = time.time()
                optimizer.zero_grad(set_to_none=True)

                step_losses = []
                for micro_step in range(grad_accum):
                    input_ids = next(data_iter).to(_device, non_blocking=True)
                    labels = input_ids  # HF shifts internally; no clone needed

                    # Skip allreduce on intermediate micro-steps
                    sync_ctx = nullcontext()
                    if is_ddp and micro_step < (grad_accum - 1):
                        sync_ctx = model.no_sync()

                    with sync_ctx, fp8_ctx_fn():
                        with torch.amp.autocast('cuda', dtype=torch_dtype):
                            outputs = model(input_ids=input_ids, labels=labels)
                            loss = outputs.loss / grad_accum
                        loss.backward()
                    step_losses.append(loss.item() * grad_accum)
                    # Free activation memory immediately to avoid OOM on next micro-step
                    del outputs, loss, input_ids, labels

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, foreach=True)
                optimizer.step()
                scheduler.step()

                step_time = time.time() - step_start
                avg_loss = sum(step_losses) / len(step_losses)
                tokens_per_step = batch_size * SEQ_LEN * grad_accum * world_size
                throughput = tokens_per_step / step_time

                step_times.append(step_time)
                loss_values.append(avg_loss)
                current_lr = scheduler.get_last_lr()[0] if scheduler is not None else optimizer.param_groups[0]['lr']
                learning_rates.append(current_lr)

                if rank == 0:
                    logger.info(f"Step {step + 1}/{num_steps} | Loss: {avg_loss:.4f} | Time: {step_time:.3f}s | Throughput: {throughput:.0f} tokens/s")

                if rank == 0 and (step + 1) % 5 == 0:
                    if torch.cuda.is_available():
                        allocated = torch.cuda.memory_allocated(_device) / 1e9
                        reserved = torch.cuda.memory_reserved(_device) / 1e9
                        logger.info(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
        finally:
            gc.enable()

        training_time = time.time() - training_start

        if rank == 0 and len(step_times) > 10:
            warmup_skip = min(10, len(step_times))
            step_times_no_warmup = step_times[warmup_skip:]
            if not step_times_no_warmup:
                step_times_no_warmup = step_times

            avg_step_time = sum(step_times_no_warmup) / len(step_times_no_warmup)
            steps_per_second = len(step_times_no_warmup) / sum(step_times_no_warmup)
            tokens_per_step = global_batch_size * SEQ_LEN
            tokens_per_second = tokens_per_step / avg_step_time
            tokens_per_second_per_gpu = tokens_per_second / world_size if world_size else None

            results = {
                "platform": platform,
                "dataset": DATASET,
                "gpu_info": gpu_info,
                "timestamp": datetime.now().isoformat(),
                "training_config": {
                    "max_steps": num_steps,
                    "global_batch_size": global_batch_size,
                    "micro_batch_size": batch_size,
                    "sequence_length": SEQ_LEN,
                    "num_gpus": world_size,
                    "parallel_strategy": "ddp",
                    "gradient_accumulation_steps": grad_accum,
                    "attn_implementation": attn_impl_used,
                    "fp8_hybrid": fp8_enabled,
                },
                "performance_metrics": {
                    "total_steps": len(step_times),
                    "total_time_seconds": training_time,
                    "avg_step_time_seconds": avg_step_time,
                    "min_step_time_seconds": min(step_times_no_warmup),
                    "max_step_time_seconds": max(step_times_no_warmup),
                    "steps_per_second": steps_per_second,
                    "tokens_per_second": tokens_per_second,
                    "tokens_per_second_per_gpu": tokens_per_second_per_gpu,
                    "throughput_per_gpu_core": steps_per_second / gpu_info["gpu_cores"] if gpu_info.get("gpu_cores", 0) > 0 else 0,
                },
                "step_times": step_times,
                "loss_values": loss_values,
                "learning_rates": learning_rates,
            }

            logger.info(f"=" * 80)
            logger.info(f"Training completed for {model_name}")
            logger.info(f"Total time: {training_time:.2f}s")
            logger.info(f"Average step time: {avg_step_time:.3f}s")
            logger.info(f"Throughput: {tokens_per_second:,.0f} tokens/sec")
            logger.info(f"Per-GPU Throughput: {tokens_per_second_per_gpu:,.0f} tokens/sec/GPU")
            logger.info(f"=" * 80)

            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            output_file = OUTPUT_DIR / f"train_{platform}_mega_{model_name}_{DATASET}.json"
            from lib.utils import round_floats
            results_rounded = round_floats(results, precision=5)

            with open(output_file, 'w') as f:
                json.dump(results_rounded, f, indent=2)

            logger.info(f"Results saved to {output_file}")

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        if rank == 0 and len(step_times) > 0:
            results = {
                "platform": platform,
                "gpu_info": gpu_info,
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "partial_results": {
                    "steps_completed": len(step_times),
                    "step_times": step_times,
                    "loss_values": loss_values,
                },
            }
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            output_file = OUTPUT_DIR / f"train_nvd_mega_{model_name}_{DATASET}.json"
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
        raise

    finally:
        if world_size > 1:
            torch.distributed.destroy_process_group()


def train_llama():
    config = {
        'hf_model': 'meta-llama/Llama-3.1-8B',
        'grad_accum_steps': GA,
        'tensor_parallel': TP,
        'pipeline_parallel': PP,
    }
    train_model('llama', config)


def train_qwen():
    config = {
        'hf_model': 'Qwen/Qwen2.5-7B',
        'grad_accum_steps': GA,
        'tensor_parallel': TP,
        'pipeline_parallel': PP,
    }
    train_model('qwen', config)


def main():
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    os.environ['PYTHONHASHSEED'] = str(SEED)
    os.environ['HSA_NO_SCRATCH_RECLAIM'] = '1'
    os.environ['HSA_ENABLE_SDMA'] = '1'
    os.environ['HSA_FORCE_FINE_GRAIN_PCIE'] = '1'
    os.environ.setdefault('RCCL_DEBUG', 'ERROR')
    os.environ.setdefault('NCCL_DEBUG', 'ERROR')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Environment configured for GPU training")
    logger.info(f"Output directory: {OUTPUT_DIR}")

    if len(sys.argv) < 2:
        logger.info("No model specified, training all models")
        train_llama()
        train_qwen()
    else:
        model = sys.argv[1]
        logger.info(f"Training model: {model}")
        if model == "llama":
            train_llama()
        elif model == "qwen":
            train_qwen()
        else:
            logger.error(f"Unknown model: {model}")
            logger.error("\nUsage:")
            logger.error("  python train_nvd_mega.py              # Train both models")
            logger.error("  python train_nvd_mega.py llama        # Train only Llama")
            logger.error("  python train_nvd_mega.py qwen         # Train only Qwen")
            sys.exit(1)


if __name__ == "__main__":
    main()
