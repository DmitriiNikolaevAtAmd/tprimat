#!/usr/bin/env python3
import argparse
import gc
import json
import logging
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from contextlib import nullcontext

import numpy as np
import torch

WORKSPACE_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))

for env_var in ("HF_HOME", "HF_DATASETS_CACHE"):
    val = os.environ.get(env_var)
    if val and not os.path.isabs(val):
        os.environ[env_var] = str(WORKSPACE_ROOT / val)

DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
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
WARMUP_STEPS = int(os.environ.get("WARMUP_STEPS", 50))
TRAIN_ITERS = int(os.environ.get("TRAIN_ITERS", 500))
GA = int(os.environ.get("GA", 8))

# Parallelism
TP = int(os.environ.get("TP", 1))
PP = int(os.environ.get("PP", 1))
DP = int(os.environ.get("DP", 8))

MODELS = {
    "llama": "meta-llama/Llama-3.1-8B",
    "qwen": "Qwen/Qwen2.5-7B",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Data pipeline
# ──────────────────────────────────────────────────────────────────────

class _TokenDataset(torch.utils.data.Dataset):
    """Map-style dataset wrapping IndexedDataset for use with DataLoader."""

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

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_gpu_info(world_size: int) -> dict:
    if not torch.cuda.is_available():
        return {}

    device_props = torch.cuda.get_device_properties(0)
    device_name = torch.cuda.get_device_name(0)

    gpu_cores = 19456 if "mi300" in device_name.lower() else 6912

    return {
        "device_count": world_size,
        "device_name": device_name,
        "total_memory_gb": device_props.total_memory / 1e9,
        "gpu_cores": gpu_cores,
        "pytorch_version": torch.__version__,
        "software_stack": "megatron",
        "software_version": getattr(torch.version, "hip", "unknown"),
    }


def train_model(model_name: str):
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend="nccl")

    set_seed(SEED)

    # ── Performance knobs ──────────────────────────────────────────────
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # ── FIX: derive grad_accum from GBS (same fix as NVIDIA script) ───
    grad_accum = max(1, GBS // (MBS * world_size))
    global_batch_size = MBS * grad_accum * world_size
    logger.info(f"GBS={GBS} → grad_accum={grad_accum}, effective GBS={global_batch_size}")

    gpu_info = get_gpu_info(world_size)
    torch_dtype = torch.bfloat16 if PRECISION == "bf16" else torch.float16 if PRECISION == "fp16" else torch.float32

    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, get_cosine_schedule_with_warmup

        hf_model = MODELS[model_name]
        tokenizer = AutoTokenizer.from_pretrained(hf_model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        config = AutoConfig.from_pretrained(hf_model, trust_remote_code=True)

        _device = torch.device(f"cuda:{local_rank}" if world_size > 1 else "cuda")

        # ── Attention: try sdpa first, then eager fallback
        attn_impl_used = "sdpa"
        try:
            with torch.device(_device):
                model = AutoModelForCausalLM.from_config(
                    config, torch_dtype=torch_dtype,
                    attn_implementation="sdpa",
                )
            logger.info("Using attention implementation: sdpa")
        except Exception:
            try:
                with torch.device(_device):
                    model = AutoModelForCausalLM.from_config(
                        config, torch_dtype=torch_dtype,
                        attn_implementation="eager",
                    )
                attn_impl_used = "eager"
                logger.info("sdpa not available, using eager")
            except Exception:
                with torch.device(_device):
                    model = AutoModelForCausalLM.from_config(config, torch_dtype=torch_dtype)
                attn_impl_used = "default"
                logger.info("Using default attention implementation")
        model.config.use_cache = False

        # NOTE: gradient checkpointing intentionally disabled for fair
        # benchmarking — it halves activation memory but costs ~30-40 %
        # throughput.  Disabled on NVIDIA side as well.

        # ── DDP ───────────────────────────────────────────────────────
        # NOTE: static_graph=True is incompatible with model.no_sync()
        # used during gradient accumulation (the autograd graph differs
        # between synced and no-sync micro-steps).
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
            logger.info("Wrapped model with DDP (bucket_view)")
            is_ddp = True

        # ── Data ──────────────────────────────────────────────────────
        dataset_name = os.environ.get("DATASET", "bc")
        dataset_path = str(DATA_DIR / f"{dataset_name}-train")

        if not os.path.exists(f"{dataset_path}.idx") or not os.path.exists(f"{dataset_path}.bin"):
            raise FileNotFoundError(
                f"Data not found. Expected:\n  {dataset_path}.idx\n  {dataset_path}.bin\n"
                f"Set DATA_DIR and DATASET (e.g. DATASET=bc) or create/symlink the indexed dataset."
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

        # ── Optimizer ─────────────────────────────────────────────────
        # Use ZeRO-1 for multi-GPU to match NVIDIA script — shards
        # optimizer states across DP ranks, reducing per-GPU memory.
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

        # Include _WARMUP_ITERS in the schedule so the LR curve aligns with
        # NeMo (where Lightning advances the scheduler during CUDA warmup).
        # scheduler.step() is called during warmup below, consuming the extra
        # steps before timed training begins.
        _WARMUP_ITERS = 0
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=WARMUP_STEPS + _WARMUP_ITERS,
            num_training_steps=TRAIN_ITERS + _WARMUP_ITERS,
        )

        model.train()
        step_times = []
        loss_values = []
        learning_rates = []

        # ── CUDA warmup (un-timed iterations to pre-compile kernels) ────
        logger.info(f"Running {_WARMUP_ITERS} warmup iterations to pre-compile CUDA kernels...")
        for _w in range(_WARMUP_ITERS):
            optimizer.zero_grad(set_to_none=True)
            for micro_step in range(grad_accum):
                input_ids = next(data_iter).to(_device, non_blocking=True)
                labels = input_ids
                sync_ctx = nullcontext()
                if is_ddp and micro_step < (grad_accum - 1):
                    sync_ctx = model.no_sync()
                with sync_ctx:
                    with torch.amp.autocast("cuda", dtype=torch_dtype):
                        outputs = model(input_ids=input_ids, labels=labels)
                        loss = outputs.loss / grad_accum
                    loss.backward()
                del outputs, loss, input_ids, labels
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, foreach=True)
            optimizer.step()
            scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        logger.info("CUDA warmup complete, starting timed training...")

        training_start = time.time()

        # Disable GC during training to avoid random pauses
        gc.disable()
        try:
            for step in range(TRAIN_ITERS):
                step_start = time.time()
                optimizer.zero_grad(set_to_none=True)

                step_losses = []
                for micro_step in range(grad_accum):
                    input_ids = next(data_iter).to(_device, non_blocking=True)
                    labels = input_ids

                    # Skip allreduce on intermediate micro-steps
                    sync_ctx = nullcontext()
                    if is_ddp and micro_step < (grad_accum - 1):
                        sync_ctx = model.no_sync()

                    with sync_ctx:
                        with torch.amp.autocast("cuda", dtype=torch_dtype):
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
                tokens_per_step = MBS * SEQ_LEN * grad_accum * world_size
                throughput = tokens_per_step / step_time

                step_times.append(step_time)
                loss_values.append(avg_loss)
                current_lr = scheduler.get_last_lr()[0]
                learning_rates.append(current_lr)

                if rank == 0:
                    logger.info(f"Step {step + 1}/{TRAIN_ITERS} | Loss: {avg_loss:.4f} | Time: {step_time:.3f}s | Throughput: {throughput:.0f} tokens/s")
        finally:
            gc.enable()

        training_time = time.time() - training_start

        if rank == 0 and len(step_times) > 10:
            warmup_skip = min(10, len(step_times))
            step_times_steady = step_times[warmup_skip:]

            avg_step_time = sum(step_times_steady) / len(step_times_steady)
            tokens_per_step = global_batch_size * SEQ_LEN
            tokens_per_second = tokens_per_step / avg_step_time

            results = {
                "platform": "amd",
                "dataset": dataset_name,
                "gpu_info": gpu_info,
                "timestamp": datetime.now().isoformat(),
                "training_config": {
                    "max_steps": TRAIN_ITERS,
                    "global_batch_size": global_batch_size,
                    "micro_batch_size": MBS,
                    "sequence_length": SEQ_LEN,
                    "num_gpus": world_size,
                    "parallel_strategy": f"TP{TP}_PP{PP}_DP{DP}",
                    "tensor_parallel_size": TP,
                    "pipeline_parallel_size": PP,
                    "data_parallel_size": DP,
                    "gradient_accumulation_steps": grad_accum,
                    "attn_implementation": attn_impl_used,
                },
                "performance_metrics": {
                    "total_steps": len(step_times),
                    "total_time_seconds": training_time,
                    "avg_step_time_seconds": avg_step_time,
                    "tokens_per_second": tokens_per_second,
                    "tokens_per_second_per_gpu": tokens_per_second / world_size,
                },
                "step_times": step_times,
                "loss_values": loss_values,
                "learning_rates": learning_rates,
            }

            # Merge memory log into results
            mem_log = os.environ.get("MEMORY_LOG")
            if mem_log and os.path.exists(mem_log):
                try:
                    from evaluate.extract_metrics import parse_memory_log
                    num_steps = len(step_times)
                    mem_data = parse_memory_log(mem_log, num_steps=num_steps)
                    if mem_data:
                        results["memory_metrics"] = {
                            "peak_memory_allocated_gb": mem_data["peak_memory_gb"],
                            "avg_memory_allocated_gb": mem_data["avg_memory_gb"],
                            "min_memory_allocated_gb": mem_data["min_memory_gb"],
                        }
                        results["memory_values"] = mem_data["memory_values"]
                        logger.info(
                            "Memory: peak %.2f GB, avg %.2f GB (%d samples)",
                            mem_data["peak_memory_gb"],
                            mem_data["avg_memory_gb"],
                            mem_data.get("raw_samples", len(mem_data["memory_values"])),
                        )
                except Exception as e:
                    logger.warning("Could not merge memory log: %s", e)

            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            output_file = OUTPUT_DIR / f"train_amd_mega_{model_name}_{dataset_name}.json"

            with open(output_file, "w") as f:
                json.dump(results, f, indent=2)

            logger.info(f"Results saved to {output_file}")

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise

    finally:
        if world_size > 1:
            torch.distributed.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="AMD Megatron-style training")
    parser.add_argument(
        "model",
        type=str,
        nargs="?",
        choices=["llama", "qwen"],
        help="Model to train (llama or qwen)",
    )

    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.model:
        train_model(args.model)
    else:
        train_model("llama")
        train_model("qwen")


if __name__ == "__main__":
    main()
