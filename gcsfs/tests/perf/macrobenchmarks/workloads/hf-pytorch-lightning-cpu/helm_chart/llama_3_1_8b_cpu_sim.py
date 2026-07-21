"""CPU-only IO simulator for the Llama 3.1 8B Lightning training benchmark.

Reproduces the GCS IO pattern (N_NODES x 4 ranks x 16 dataloader workers,
periodic checkpoint writes of the full bf16 8B state dict) without GPUs. The
real Llama model is loaded and held frozen so checkpoint file sizes match
production; GPU compute is replaced by ``time.sleep(SIMULATED_STEP_COMPUTE_SECONDS)``
in ``training_step``.

Single-node launch (smoke test):
    torchrun --nproc_per_node=4 --nnodes=1 llama_3_1_8b_cpu_sim.py

Multi-node launch (the production emulator: 2 c4-standard-192 VMs, each
running 4 processes that stand in for GPU chips -- capped at 4/node, down from
8, so a checkpoint-restoring run fits the 720GB host RAM). The Helm chart in
``helm_chart/templates/`` wires up the K8s JobSet and the per-pod launcher;
on each pod it ultimately runs:
    torchrun --nproc_per_node=4 --nnodes=$NNODES --node_rank=$NODE_RANK \\
             --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \\
             llama_3_1_8b_cpu_sim.py

Required env vars: ``DATASET_PATH``, ``RUN_ID`` (always); ``HF_TOKEN`` only
when ``MODEL_ID`` points at the HuggingFace gated repo (i.e. not gs://).
Optional env vars: ``MODEL_ID`` (default ``meta-llama/Llama-3.1-8B``; may be
``gs://bucket/path`` for a launcher pre-downloaded copy), ``CKPT_WRITE_PATH``,
``MAX_STEPS``, ``CHECKPOINT_WRITE_INTERVAL``, ``TRAINING_STRATEGY``, etc.

``TRAINING_STRATEGY`` (default ``ddp``; ``fsdp_sharded`` shards the model and
writes a per-rank sharded/distributed checkpoint; ``fsdp_full`` shards the
model but consolidates to a single rank-0-written checkpoint at save time,
like ``ddp``) selects the parallel-training strategy. A resume
(``CKPT_LOAD_PATH``) must point at a checkpoint produced by the same strategy
-- cross-strategy restore is unsupported.
"""

from datetime import timedelta
import logging
import os
import sys
import shutil
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import torch.multiprocessing

# forkserver before any other torch import that could spawn workers.
try:
    torch.multiprocessing.set_start_method("forkserver", force=True)
except RuntimeError:
    pass  # context already set

import datasets
import datasets.distributed
import fsspec
import lightning.pytorch as pl
import torch
import transformers
from lightning.pytorch.callbacks import Callback, DeviceStatsMonitor, ModelCheckpoint
from lightning.pytorch.loops.fetchers import _PrefetchDataFetcher
from lightning.pytorch.loops.fit_loop import _FitLoop as FitLoop
from lightning.pytorch.strategies import DDPStrategy, FSDPStrategy
from torch.utils.data import DataLoader
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

# ---- Logging --------------------------------------------------------------
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
storage_log_level = os.getenv("GCSFS_LOG_LEVEL", "INFO").upper()
if storage_log_level == "TRACE":
    storage_log_level = "DEBUG"
logging.getLogger("gcsfs").setLevel(storage_log_level)
logging.getLogger("fsspec").setLevel(storage_log_level)

run_id = os.environ.get("RUN_ID")
if not run_id:
    raise SystemExit("RUN_ID env var is required.")

log_format = (
    "%(asctime)s - %(levelname)s - %(name)s - [Thread: %(thread)d] - %(message)s"
)
logging.basicConfig(
    format=log_format,
    level=log_level,
    handlers=[logging.StreamHandler(sys.stdout)],
)

# ---- Simulated compute ----------------------------------------------------
# Per-step stand-in for GPU forward+backward in training_step. Configurable via
# SIMULATED_STEP_COMPUTE_SECONDS (default 1.0) and recorded in the run summary.
SIMULATED_STEP_COMPUTE_SECONDS = float(
    os.getenv("SIMULATED_STEP_COMPUTE_SECONDS", "1.0")
)
# Single grep-able config marker per knob (parity with the model_id: line).
logging.info("[BENCHMARK] simulated_step_compute_seconds: %s", SIMULATED_STEP_COMPUTE_SECONDS)

# ---- Config (env-overridable) ---------------------------------------------
preset_max_steps = int(os.getenv("MAX_STEPS", "1000"))
full_pass = preset_max_steps < 0
# Default 1 (not the launcher-overridden 4 this once carried): the launcher and
# the Helm template both set GRADIENT_ACCUMULATION_STEPS=1, so 1 is the
# effective value -- the standalone smoke test should agree rather than
# silently use a different accumulation.
gradient_accumulation_steps = int(os.getenv("GRADIENT_ACCUMULATION_STEPS", "1"))
per_device_train_batch_size = int(os.getenv("PER_DEVICE_TRAIN_BATCH_SIZE", "8"))
dataloader_num_workers = int(os.getenv("DATALOADER_NUM_WORKERS", "16"))
checkpoint_load_path = os.getenv("CKPT_LOAD_PATH", None)
checkpoint_write_interval = int(os.getenv("CHECKPOINT_WRITE_INTERVAL", "25"))
if not full_pass:
    checkpoint_write_interval = min(checkpoint_write_interval, preset_max_steps)
checkpoints_to_keep = int(os.getenv("CKPT_TO_KEEP", "1"))
model_id = os.getenv("MODEL_ID", "meta-llama/Llama-3.1-8B")

# Parallel training strategy. ``ddp`` (default) replicates the frozen model on
# every rank and rank 0 writes the full checkpoint; ``fsdp_sharded`` shards the
# model and writes a sharded (distributed) checkpoint where every rank writes
# its own shard concurrently; ``fsdp_full`` shards the model but consolidates
# to a single rank-0-written checkpoint at save time, like ``ddp``. Validate
# eagerly -- before the 16 GB model load -- so a typo fails fast instead of
# after a long download.
training_strategy = os.getenv("TRAINING_STRATEGY", "ddp").lower()
if training_strategy not in ("ddp", "fsdp_sharded", "fsdp_full"):
    raise SystemExit(
        "TRAINING_STRATEGY must be 'ddp', 'fsdp_sharded', or 'fsdp_full' "
        f"(got {training_strategy!r})."
    )
# Parity with the model_id: line -- a single grep-able config marker per knob.
logging.info("[BENCHMARK] training_strategy: %s", training_strategy)

# Map model_id to the canonical id and log it as ``model_id: <id>``.
# NOTE: this macrobenchmarks pipeline does NOT consume this line -- it derives
# the summary's model_id from calculate.py's ``--model-id`` flag. The line is
# emitted so an HF metadata generator can scrape it via regex
# (``model_id: ([a-zA-Z0-9-]+)``). Computed from the original model_id
# (before the gs:// -> /tmp remap below, which would otherwise log a path
# the regex can't parse).
if "Llama-3.1-8B" in model_id:
    metadata_model_id = "llama3-1-8b"  # Default
else:
    metadata_model_id = "unknown"
logging.info("[BENCHMARK] model_id: %s", metadata_model_id)

# If ``MODEL_ID`` is a GCS path, the launcher pre-downloads the weights to
# ``/tmp/<basename>`` (gcloud storage cp -r). Remap ``model_id`` to the local
# directory and force ``local_files_only`` so transformers does not phone home.
# Without this, 8 ranks (2 nodes x 4 procs) would concurrently pull the
# 16 GB Llama-3.1-8B weights from HuggingFace.
use_local_files_only = False
if model_id.startswith("gs://"):
    use_local_files_only = True
    dir_name = os.path.basename(model_id.rstrip("/"))
    model_id = os.path.join("/tmp", dir_name)

# Required: dataset path. Fail fast if unset. Strip trailing slash so the
# downstream glob doesn't produce ``gs://bucket/dir//*.parquet``, which
# behaves inconsistently across fsspec versions.
dataset_path = os.environ.get("DATASET_PATH")
if not dataset_path:
    raise SystemExit(
        "DATASET_PATH env var is required (e.g. gs://your-bucket/parquet-dir)."
    )
dataset_path = dataset_path.rstrip("/")

# HF token is only needed when weights are pulled from the HuggingFace gated
# repo. If ``MODEL_ID`` is a GCS path, the launcher has already downloaded the
# weights locally and ``local_files_only=True`` is set below, so no token is
# required.
if not use_local_files_only and not os.environ.get("HF_TOKEN"):
    raise SystemExit(
        "HF_TOKEN env var is required when MODEL_ID is a HuggingFace repo "
        "(Llama-3.1-8B is gated). Set MODEL_ID=gs://... to use a "
        "pre-downloaded copy instead."
    )

# Optional: checkpoint write path. If unset, the checkpoint callback is
# omitted entirely. Strip trailing slash for the same reason as
# ``dataset_path``.
checkpoint_write_path = os.getenv("CKPT_WRITE_PATH")
if checkpoint_write_path:
    checkpoint_write_path = checkpoint_write_path.rstrip("/")

# torchrun-provided env (defaults so the module is importable outside torchrun).
# Note: torchrun sets RANK, LOCAL_RANK, WORLD_SIZE, LOCAL_WORLD_SIZE,
# MASTER_ADDR, MASTER_PORT -- but NOT NNODES (that's a torchrun CLI flag and
# doesn't propagate to env). Derive num_nodes from WORLD_SIZE / LOCAL_WORLD_SIZE
# so multi-node launches are reported correctly.
world_size = int(os.environ.get("WORLD_SIZE", "1"))
local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
num_nodes = max(1, world_size // local_world_size)
global_batch_size = (
    per_device_train_batch_size * gradient_accumulation_steps * world_size
)
logging.info("[BENCHMARK] global_batch_size: %d", global_batch_size)

# ---- Tokenizer ------------------------------------------------------------
# Real Llama tokenizer. Requires HF_TOKEN env var when downloading from the
# HuggingFace gated repo; when ``MODEL_ID=gs://...`` the launcher has already
# placed the tokenizer files alongside the weights, so ``local_files_only``
# avoids any network access from this process.
tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_id, local_files_only=use_local_files_only
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


def collate_fn(examples):
    """Runs in DataLoader worker processes, so tokenization CPU work overlaps
    with the next batch's GCS reads -- this is the IO-overlap behavior we
    care about preserving.
    """
    tokens = tokenizer(
        [ex["text"] for ex in examples],
        return_tensors="pt",
        padding="longest",
        truncation=True,
        max_length=512,
    )
    tokens["labels"] = tokens["input_ids"].clone()
    return tokens


# ---- LightningModule ------------------------------------------------------
class LlamaLitModel(pl.LightningModule):
    """Holds the real Llama 8B model (frozen) for realistic checkpoint size;
    runs a fake forward via a tiny trainable Linear so DDP all-reduce (or FSDP's
    per-shard gradient sync) has something to sync without paying 8B-param
    collective costs.

    ``training_step`` sleeps for ``SIMULATED_STEP_COMPUTE_SECONDS`` to mimic the time
    a GPU step would take. ``self.model``'s parameters end up in the
    Lightning state_dict, and AdamW is configured over ``self.model`` (ddp) or
    all parameters post-FSDP-wrap (fsdp_sharded/fsdp_full) with materialized
    optimizer state.
    When ``ModelCheckpoint`` writes via fsspec to ``gs://...`` the uploaded
    blob is approximately the size of a real bf16 Llama 8B checkpoint with
    optimizer state.
    """

    def __init__(self, model, training_strategy="ddp"):
        super().__init__()
        self.model = model
        self._fsdp = training_strategy in ("fsdp_sharded", "fsdp_full")
        for p in self.model.parameters():
            p.requires_grad = False
        trainable_dtype = model.dtype if self._fsdp else torch.float32
        self.trainable = torch.nn.Linear(8, 8).to(trainable_dtype)

    def training_step(self, batch, batch_idx):
        # Pull the batch out of the dataloader -- this is what drives the
        # GCS read traffic we are benchmarking. The batch contents are then
        # ignored; we sleep to simulate GPU compute.
        del batch
        time.sleep(SIMULATED_STEP_COMPUTE_SECONDS)
        zeros = torch.zeros(1, 8, dtype=self.trainable.weight.dtype)
        # Real loss with a real grad path so backward + DDP all-reduce run.
        # Squared so the loss is always non-negative: the metrics pipeline's
        # step-metrics regex matches "Loss: [0-9.]+" (no leading '-'), so a
        # negative loss would silently drop every step_time/throughput sample.
        # self.trainable is never optimized (configure_optimizers builds AdamW
        # over the frozen self.model), so without the square the loss is a
        # constant whose sign is random per run -- ~50% of runs would emit a
        # negative loss and capture zero step metrics.
        return (self.trainable(zeros) ** 2).sum()

    @staticmethod
    def _materialize_adamw_state(optimizer):
        """Eagerly allocate AdamW moments so checkpoint size is realistic."""
        for group in optimizer.param_groups:
            for p in group["params"]:
                state = optimizer.state[p]
                if state:
                    continue
                # Random, not zero: an all-zero buffer is trivially compressible/
                # dedupable (page merging, a future transport compression layer,
                # etc.), which would let this ~2/3 of the checkpoint transfer
                # faster than the real, non-degenerate floats a trained
                # optimizer actually produces -- skewing the IO benchmark.
                state["step"] = torch.zeros((), dtype=torch.float32)
                state["exp_avg"] = torch.randn_like(
                    p, memory_format=torch.preserve_format
                )
                state["exp_avg_sq"] = torch.rand_like(
                    p, memory_format=torch.preserve_format
                )
                if group["amsgrad"]:
                    state["max_exp_avg_sq"] = torch.rand_like(
                        p, memory_format=torch.preserve_format
                    )

    def configure_optimizers(self):
        params = self.parameters() if self._fsdp else self.model.parameters()
        optimizer = torch.optim.AdamW(
            params,
            lr=float(os.getenv("LEARNING_RATE", "2e-5")),
            weight_decay=float(os.getenv("WEIGHT_DECAY", "1e-6")),
        )
        self._materialize_adamw_state(optimizer)
        return optimizer


# ---- Callbacks ------------------------------------------------------------
class StepTimeCallback(Callback):
    """Logs ``step_time`` and ``throughput`` every optimizer step."""

    def __init__(self):
        super().__init__()
        self.ckpt_time = 0.0

    def on_train_start(self, trainer, pl_module):
        # Start timer at the beginning of the training to capture the first batch's data loading time
        self.start_time = time.perf_counter()
        self.ckpt_time = 0.0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Only emit metrics on micro-batches that complete an optimizer step;
        # otherwise step_time would cover a single micro-batch while
        # global_batch_size counts the whole accumulation window, inflating
        # throughput by gradient_accumulation_steps.
        if (batch_idx + 1) % trainer.accumulate_grad_batches != 0:
            return

        # Calculate step time excluding the checkpointing time
        step_time = time.perf_counter() - self.start_time - self.ckpt_time

        per_rank_batch_size = (
            per_device_train_batch_size * trainer.accumulate_grad_batches
        )
        local_throughput = per_rank_batch_size / step_time
        global_throughput = global_batch_size / step_time

        pl_module.log("step_time", step_time)
        pl_module.log("local_throughput", local_throughput)
        pl_module.log("global_throughput", global_throughput)

        loss = outputs["loss"] if isinstance(outputs, dict) else outputs
        loss_val = loss.item() if isinstance(loss, torch.Tensor) else loss
        logging.info(
            "[BENCHMARK] Global Rank: %d | Step: %d | Loss: %.4f | Step Time: %.4fs | "
            "Throughput: %.2f samples/s | Local Throughput: %.2f samples/s",
            trainer.global_rank,
            trainer.global_step,
            loss_val,
            step_time,
            global_throughput,
            local_throughput,
        )

        # Reset the timer for the next step, capturing its data loading time.
        self.start_time = time.perf_counter()
        self.ckpt_time = 0.0


class LoggedModelCheckpoint(ModelCheckpoint):
    """ModelCheckpoint wrapper that logs save/delete duration.

    Under DDP, only rank 0 actually writes the checkpoint; that single
    upload of the bf16 Llama 8B state_dict (~16 GB) to gs:// is the
    headline IO event we want to time.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.async_checkpoint = os.getenv("ASYNC_CHECKPOINT", "false").lower() == "true"
        self._executor = ThreadPoolExecutor(max_workers=1) if self.async_checkpoint else None
        self._last_future = None

    def teardown(self, trainer, pl_module, stage=None):
        if self._executor is not None:
            self._executor.shutdown(wait=True)
        super().teardown(trainer, pl_module, stage)

    @staticmethod
    def _parallel_copytree(src_dir, dst_dir, max_workers=None):
        if max_workers is None:
            max_workers = int(os.getenv("PARALLEL_COPY_WORKERS", "32"))
        os.makedirs(dst_dir, exist_ok=True)
        file_tasks = []
        for dirpath, _, filenames in os.walk(src_dir):
            rel_dir = os.path.relpath(dirpath, src_dir)
            target_dir = os.path.join(dst_dir, rel_dir) if rel_dir != "." else dst_dir
            os.makedirs(target_dir, exist_ok=True)
            for f in filenames:
                s_file = os.path.join(dirpath, f)
                d_file = os.path.join(target_dir, f)
                file_tasks.append((s_file, d_file))

        def _copy_one(pair):
            s, d = pair
            shutil.copyfile(s, d)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            list(executor.map(_copy_one, file_tasks))

    @classmethod
    def _copy_staged_checkpoint(cls, src_path, dst_path):
        src_ts = src_path.replace(".ckpt", ".ts_zarr")
        dst_ts = dst_path.replace(".ckpt", ".ts_zarr")

        if os.path.exists(src_ts):
            if dst_path.startswith("gs://"):
                pass
            else:
                parent_dir = os.path.dirname(dst_ts)
                if parent_dir:
                    os.makedirs(parent_dir, exist_ok=True)
                tmp_dst_ts = dst_ts + ".part"
                if os.path.exists(tmp_dst_ts):
                    shutil.rmtree(tmp_dst_ts)
                if os.path.isdir(src_ts):
                    cls._parallel_copytree(src_ts, tmp_dst_ts)
                else:
                    shutil.copyfile(src_ts, tmp_dst_ts)
                if os.path.exists(dst_ts):
                    if os.path.isdir(dst_ts):
                        shutil.rmtree(dst_ts)
                    else:
                        os.remove(dst_ts)
                os.replace(tmp_dst_ts, dst_ts)

        if os.path.exists(src_path) and os.path.getsize(src_path) > 0:
            tmp_dst = dst_path + ".part"
            parent_dir = os.path.dirname(dst_path)
            if parent_dir and not dst_path.startswith("gs://"):
                os.makedirs(parent_dir, exist_ok=True)

            if dst_path.startswith("gs://"):
                from gcsfs.extended_gcsfs import ExtendedGcsFileSystem
                fs = ExtendedGcsFileSystem()
                gcs_path = dst_path[5:] if dst_path.startswith("gs://") else dst_path
                with open(src_path, "rb") as f_src, fs.open(gcs_path, "wb", finalize_on_close=True) as f_dst:
                    shutil.copyfileobj(f_src, f_dst, length=64 * 1024 * 1024)
            else:
                shutil.copyfile(src_path, tmp_dst)
                os.replace(tmp_dst, dst_path)

    def _save_to_single_target(self, trainer, target_filepath, is_writer, staged_tmp_file=None, is_last_target=False):
        start_time_wall = time.time()
        start_time_perf = time.perf_counter()

        if target_filepath.startswith("gs://"):
            backend_label = "TensorStore Direct GCS"
        elif "/lustre" in target_filepath:
            backend_label = "Lustre"
        elif "/gcs" in target_filepath:
            backend_label = "GCSFuse"
        else:
            backend_label = "POSIX"

        if is_writer:
            logging.info(
                "[BENCHMARK] [%s] Checkpoint Save (Writer Rank %d) : Rank: %d : Step: %d : Start time: %f seconds: Path: %s",
                backend_label,
                trainer.global_rank,
                trainer.global_rank,
                trainer.global_step,
                start_time_wall,
                target_filepath,
            )
        else:
            logging.info(
                "[BENCHMARK] [%s] Checkpoint Save (Non-Writing Rank %d - skipped data upload) : Rank: %d : Step: %d : Start time: %f seconds: Path: %s",
                backend_label,
                trainer.global_rank,
                trainer.global_rank,
                trainer.global_step,
                start_time_wall,
                target_filepath,
            )

        def _do_save():
            save_start = time.perf_counter()
            stop_progress_event = threading.Event()
            upload_start_t = [None]
            last_sample_t = [save_start]
            last_bytes = [0]

            def _progress_ticker():
                interval = float(os.getenv("CHECKPOINT_PROGRESS_INTERVAL_SECONDS", "5.0"))
                while not stop_progress_event.wait(interval):
                    now_t = time.perf_counter()
                    total_elapsed = now_t - save_start
                    if is_writer:
                        try:
                            curr_bytes = self._measure_checkpoint_bytes(target_filepath)
                            if (curr_bytes == 0 or curr_bytes is None) and os.path.exists(target_filepath + ".part"):
                                curr_bytes = self._measure_checkpoint_bytes(target_filepath + ".part")
                            curr_mb = curr_bytes / (1024 * 1024)
                            curr_gb = curr_bytes / (1024 * 1024 * 1024)

                            if curr_bytes > 0 and upload_start_t[0] is None:
                                upload_start_t[0] = now_t

                            if upload_start_t[0] is not None:
                                upload_elapsed = max(0.001, now_t - upload_start_t[0])
                                pure_rate_mb_s = curr_mb / upload_elapsed
                                status = f"Uploading/Writing (Upload Time: {upload_elapsed:.1f}s)"
                            else:
                                pure_rate_mb_s = 0.0
                                status = "In-Memory Serialization (CPU pickling state_dict)"

                            dt = max(0.001, now_t - last_sample_t[0])
                            d_bytes = max(0, curr_bytes - last_bytes[0])
                            instant_rate_mb_s = (d_bytes / (1024 * 1024)) / dt

                            last_sample_t[0] = now_t
                            last_bytes[0] = curr_bytes

                            logging.info(
                                "[BENCHMARK] [%s] Checkpoint Upload Progress : Rank : %d : Step : %d : Total Elapsed : %.1fs : Status : %s : Size : %d bytes (%.2f MB / %.2f GB) : Instant Rate : %.2f MB/s : Upload Rate : %.2f MB/s : Path : %s",
                                backend_label,
                                trainer.global_rank,
                                trainer.global_step,
                                total_elapsed,
                                status,
                                curr_bytes,
                                curr_mb,
                                curr_gb,
                                instant_rate_mb_s,
                                pure_rate_mb_s,
                                target_filepath,
                            )
                        except Exception:
                            pass

            ticker_thread = None
            if is_writer:
                ticker_thread = threading.Thread(target=_progress_ticker, daemon=True)
                ticker_thread.start()

            try:
                if is_writer:
                    if target_filepath.startswith("gs://"):
                        try:
                            self._write_checkpoint_file(trainer, target_filepath)
                        except Exception as gs_err:
                            logging.warning("[BENCHMARK] Direct REST GCS save failed for %s (expected for Rapid/Zonal buckets): %s", target_filepath, gs_err)
                    elif staged_tmp_file and os.path.exists(staged_tmp_file):
                        self._copy_staged_checkpoint(staged_tmp_file, target_filepath)
                    else:
                        self._write_checkpoint_file(trainer, target_filepath)
            finally:
                stop_progress_event.set()
                if ticker_thread is not None:
                    ticker_thread.join(timeout=1.0)
                if is_last_target and staged_tmp_file and os.path.exists(staged_tmp_file):
                    try:
                        os.remove(staged_tmp_file)
                    except Exception:
                        pass

            total_duration = time.perf_counter() - save_start
            finish_t = time.perf_counter()
            upload_duration = (finish_t - upload_start_t[0]) if upload_start_t[0] is not None else total_duration

            size_bytes = None
            if is_writer:
                try:
                    size_bytes = self._measure_checkpoint_bytes(target_filepath)
                except Exception as e:
                    logging.warning("[BENCHMARK] Could not measure checkpoint size: %s", e)

            if is_writer and size_bytes is not None and total_duration > 0:
                size_mb = size_bytes / (1024 * 1024)
                size_gb = size_bytes / (1024 * 1024 * 1024)

                overall_mb_s = size_mb / total_duration
                overall_gb_s = size_gb / total_duration

                upload_mb_s = size_mb / upload_duration if upload_duration > 0 else overall_mb_s
                upload_gb_s = size_gb / upload_duration if upload_duration > 0 else overall_gb_s

                logging.info(
                    "[BENCHMARK] [%s] Finished saving checkpoint (Writer Rank %d) to %s in %.2f seconds (Upload Time: %.2f seconds) for global_step %d from rank %d "
                    "(Size: %d bytes / %.2f MB / %.2f GB, Network Upload Throughput: %.2f MB/s / %.2f GB/s, Overall Throughput: %.2f MB/s / %.2f GB/s)",
                    backend_label,
                    trainer.global_rank,
                    target_filepath,
                    total_duration,
                    upload_duration,
                    trainer.global_step,
                    trainer.global_rank,
                    size_bytes,
                    size_mb,
                    size_gb,
                    upload_mb_s,
                    upload_gb_s,
                    overall_mb_s,
                    overall_gb_s,
                )
                logging.info(
                    "[BENCHMARK] [%s] Checkpoint Size : Rank : %d : Step : %d : Bytes : %d : Path: %s",
                    backend_label,
                    trainer.global_rank,
                    trainer.global_step,
                    size_bytes,
                    target_filepath,
                )
            else:
                logging.info(
                    "[BENCHMARK] [%s] Finished saving checkpoint (Non-Writing Rank %d - skipped data upload) to %s in %.2f seconds for global_step %d from rank %d",
                    backend_label,
                    trainer.global_rank,
                    target_filepath,
                    total_duration,
                    trainer.global_step,
                    trainer.global_rank,
                )

            return (size_bytes if (is_writer and size_bytes) else 0, start_time_wall, time.time())

        if self.async_checkpoint:
            if self._last_future is not None and not self._last_future.done():
                logging.info(
                    "[BENCHMARK] Waiting for previous async checkpoint to finish before launching step %d...",
                    trainer.global_step,
                )
                self._last_future.result()
            logging.info(
                "[BENCHMARK] Checkpoint Save launched asynchronously in background thread for step %d to %s",
                trainer.global_step,
                target_filepath,
            )
            self._last_future = self._executor.submit(_do_save)
            return (0, start_time_wall, time.time())
        else:
            res = _do_save()
            for callback in trainer.callbacks:
                if isinstance(callback, StepTimeCallback):
                    callback.ckpt_time += (time.perf_counter() - start_time_perf)
            return res

    def _write_tensorstore_checkpoint(self, trainer, target_path):
        """Writes checkpoint state dict as TensorStore Zarr arrays to target_path."""
        import tensorstore as ts
        import numpy as np
        ts_driver = os.getenv("TS_DRIVER", "zarr").lower()
        ext = ".ts_bin" if ts_driver in ("raw", "bin", "npy", "npz") else ".ts_zarr"
        ts_dir = target_path.replace(".ckpt", ext)
        logging.info(
            "[BENCHMARK] [TensorStore] Save Start (Rank %d) : Step: %d : Path: %s",
            trainer.global_rank,
            trainer.global_step,
            ts_dir,
        )
        t0 = time.perf_counter()
        checkpoint_dict = trainer._checkpoint_connector.dump_checkpoint()
        state_dict = checkpoint_dict.get("state_dict", {})
        ts_driver = os.getenv("TS_DRIVER", "zarr").lower()
        if ts_driver in ("zarr3", "zarr3_sharded", "sharded10", "raw10", "bin10"):
            items_to_write = [(k, v) for k, v in state_dict.items() if isinstance(v, torch.Tensor)]
            num_shards = int(os.getenv("NUM_SHARDS", "10"))
            shard_items = [[] for _ in range(num_shards)]
            for idx, item in enumerate(items_to_write):
                shard_items[idx % num_shards].append(item)

            import gc

            def _write_shard(shard_idx):
                items = shard_items[shard_idx]
                flat_list = []
                for name, tensor in items:
                    if tensor.dtype in (torch.bfloat16, torch.float16):
                        flat_list.append(tensor.detach().cpu().view(torch.uint8).numpy().ravel())
                    else:
                        flat_list.append(tensor.detach().cpu().numpy().ravel())
                if not flat_list:
                    return 0
                concat_arr = np.concatenate(flat_list)
                del flat_list
                if ts_driver in ("raw10", "bin10"):
                    subpath_bin = os.path.join(ts_dir, f"shard_{shard_idx:02d}.bin")
                    os.makedirs(os.path.dirname(subpath_bin), exist_ok=True)
                    with open(subpath_bin, "wb") as f:
                        f.write(concat_arr.tobytes())
                    written_nbytes = concat_arr.nbytes
                    del concat_arr
                    gc.collect()
                    return written_nbytes
                else:
                    subpath = os.path.join(ts_dir, f"shard_{shard_idx:02d}.zarr")
                    if subpath.startswith("gs://"):
                        clean_path = subpath[5:]
                        bucket = clean_path.split("/")[0]
                        blob_path = "/".join(clean_path.split("/")[1:])
                        kvstore_spec = {"driver": "gcs", "bucket": bucket, "path": blob_path}
                    else:
                        kvstore_spec = {"driver": "file", "path": subpath}
                    spec = {
                        "driver": "zarr",
                        "kvstore": kvstore_spec,
                        "metadata": {
                            "dtype": concat_arr.dtype.str,
                            "shape": [len(concat_arr)],
                            "chunks": [len(concat_arr)],
                            "compressor": None,
                        },
                        "create": True,
                        "delete_existing": True,
                    }
                    try:
                        dataset = ts.open(spec).result()
                        dataset.write(concat_arr).result()
                        written_nbytes = concat_arr.nbytes
                        del concat_arr
                        gc.collect()
                        return written_nbytes
                    except Exception as e:
                        logging.error("[BENCHMARK] [TensorStore] Exception writing shard %d: %s", shard_idx, e, exc_info=True)
                        raise e

            ts_max_workers = int(os.getenv("TS_MAX_WORKERS", "2"))
            with ThreadPoolExecutor(max_workers=min(ts_max_workers, num_shards)) as executor:
                written_bytes = list(executor.map(_write_shard, range(num_shards)))
            count = len(written_bytes)
        elif os.getenv("TS_SINGLE_ARRAY", "0") == "1":
            flat_list = []
            for name, tensor in state_dict.items():
                if isinstance(tensor, torch.Tensor):
                    if tensor.dtype in (torch.bfloat16, torch.float16):
                        flat_list.append(tensor.detach().cpu().view(torch.uint8).numpy().ravel())
                    else:
                        flat_list.append(tensor.detach().cpu().numpy().ravel())
            if flat_list:
                concat_arr = np.concatenate(flat_list)
                del flat_list
                subpath = os.path.join(ts_dir, "model_state")
                if subpath.startswith("gs://"):
                    clean_path = subpath[5:]
                    bucket = clean_path.split("/")[0]
                    blob_path = "/".join(clean_path.split("/")[1:])
                    kvstore_spec = {"driver": "gcs", "bucket": bucket, "path": blob_path}
                else:
                    kvstore_spec = {"driver": "file", "path": subpath}
                spec = {
                    "driver": "zarr",
                    "kvstore": kvstore_spec,
                    "metadata": {
                        "dtype": concat_arr.dtype.str,
                        "shape": [len(concat_arr)],
                        "chunks": [len(concat_arr)],
                        "compressor": None,
                    },
                    "create": True,
                    "delete_existing": True,
                }
                try:
                    dataset = ts.open(spec).result()
                    dataset.write(concat_arr).result()
                    del concat_arr
                    gc.collect()
                    count = 1
                except Exception as e:
                    logging.error("[BENCHMARK] [TensorStore] Exception writing single array: %s", e, exc_info=True)
                    raise e
        else:
            items_to_write = [(k, v) for k, v in state_dict.items() if isinstance(v, torch.Tensor)]
            max_workers = int(os.getenv("PARALLEL_COPY_WORKERS", "32"))

            def _write_tensor_item(item):
                name, tensor = item
                if tensor.dtype in (torch.bfloat16, torch.float16):
                    arr = tensor.detach().cpu().view(torch.uint8).numpy()
                else:
                    arr = tensor.detach().cpu().numpy()
                subpath = os.path.join(ts_dir, name.replace(".", "/"))
                ts_driver = os.getenv("TS_DRIVER", "zarr").lower()
                if ts_driver in ("raw", "bin", "npy", "npz"):
                    subpath_bin = subpath + ".bin"
                    os.makedirs(os.path.dirname(subpath_bin), exist_ok=True)
                    with open(subpath_bin, "wb") as f:
                        f.write(arr.tobytes())
                    return arr.nbytes
                else:
                    if subpath.startswith("gs://"):
                        clean_path = subpath[5:]
                        bucket = clean_path.split("/")[0]
                        blob_path = "/".join(clean_path.split("/")[1:])
                        kvstore_spec = {"driver": "gcs", "bucket": bucket, "path": blob_path}
                    else:
                        kvstore_spec = {"driver": "file", "path": subpath}
                    ts_chunk_size = int(os.getenv("TS_CHUNK_SIZE", "0"))
                    if ts_chunk_size > 0 and arr.shape:
                        chunks_spec = [min(d, ts_chunk_size) for d in arr.shape]
                    else:
                        chunks_spec = list(arr.shape) if arr.shape else [1]

                    dtype_str = arr.dtype.str
                    spec = {
                        "driver": "zarr",
                        "kvstore": kvstore_spec,
                        "metadata": {
                            "dtype": dtype_str,
                            "shape": list(arr.shape),
                            "chunks": chunks_spec,
                            "compressor": None,
                        },
                        "create": True,
                        "delete_existing": True,
                    }
                    try:
                        dataset = ts.open(spec).result()
                        dataset.write(arr).result()
                        return arr.nbytes
                    except Exception as e:
                        if subpath.startswith("gs://") and ("appendable objects" in str(e) or "400" in str(e)):
                            logging.warning("[BENCHMARK] [TensorStore] Direct REST GCS ('driver: gcs') is unsupported on Rapid/Zonal buckets: %s", e)
                            return 0
                        logging.error("[BENCHMARK] [TensorStore] Exception writing tensor '%s' (kvstore: %s): %s", name, kvstore_spec, e, exc_info=True)
                        raise e

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                written_bytes = list(executor.map(_write_tensor_item, items_to_write))
            count = len(written_bytes)
        dur = time.perf_counter() - t0
        total_files = 0
        total_bytes = 0
        if os.path.exists(ts_dir) and os.path.isdir(ts_dir):
            for dirpath, _, filenames in os.walk(ts_dir):
                total_files += len(filenames)
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    if not os.path.islink(fp):
                        total_bytes += self._get_effective_file_bytes(fp)

        logging.info(
            "[BENCHMARK] [TensorStore] Finished writing %d tensors (%d total files, %.2f MB / %.2f GB) via TensorStore to %s in %.2f seconds for global_step %d from rank %d",
            count,
            total_files,
            total_bytes / (1024 * 1024),
            total_bytes / (1024 * 1024 * 1024),
            ts_dir,
            dur,
            trainer.global_step,
            trainer.global_rank,
        )

    def _write_checkpoint_file(self, trainer, target_path):
        """Writes checkpoint dictionary to target_path directly on writer rank without DDP collective hooks."""
        if os.getenv("USE_TENSORSTORE", "false").lower() == "true" or os.getenv("CHECKPOINT_FORMAT", "").lower() == "tensorstore":
            self._write_tensorstore_checkpoint(trainer, target_path)
            return
        try:
            checkpoint_dict = trainer._checkpoint_connector.dump_checkpoint()
            torch.save(checkpoint_dict, target_path)
        except Exception:
            if hasattr(trainer, "strategy") and hasattr(trainer.strategy, "checkpoint_io"):
                checkpoint_dict = trainer._checkpoint_connector.dump_checkpoint()
                trainer.strategy.checkpoint_io.save_checkpoint(checkpoint_dict, target_path)
            else:
                super()._save_checkpoint(trainer, target_path)

    @staticmethod
    def _log_aggregated_metrics(trainer, local_bytes, start_wall, end_wall, filepath):
        backend_label = "POSIX"
        if filepath.startswith("gs://"):
            backend_label = "GCSFS"
        elif "/lustre" in filepath:
            backend_label = "Lustre"
        elif "/gcs" in filepath:
            backend_label = "GCSFuse"

        if os.getenv("USE_TENSORSTORE", "false").lower() == "true":
            backend_label = f"TensorStore ({backend_label})"

        local_info = {
            "rank": trainer.global_rank,
            "size_bytes": local_bytes if local_bytes else 0,
            "start_time": start_wall,
            "end_time": end_wall,
        }

        import torch.distributed as dist
        gathered_info = [local_info]
        if dist.is_available() and dist.is_initialized():
            try:
                world_size = dist.get_world_size()
                if world_size > 1:
                    gathered = [None] * world_size
                    dist.all_gather_object(gathered, local_info)
                    gathered_info = gathered
            except Exception as e:
                logging.warning("[BENCHMARK] Could not gather distributed checkpoint metrics: %s", e)

        total_bytes = sum(info["size_bytes"] for info in gathered_info if info)
        min_start = min(info["start_time"] for info in gathered_info if info)
        max_end = max(info["end_time"] for info in gathered_info if info)
        duration = max(0.001, max_end - min_start)

        size_mb = total_bytes / (1024 * 1024)
        size_gb = total_bytes / (1024 * 1024 * 1024)
        agg_mb_s = size_mb / duration
        agg_gb_s = size_gb / duration

        if trainer.global_rank == 0:
            logging.info(
                "[BENCHMARK] [%s] Aggregated Checkpoint Save Complete : Step : %d : Total Size : %d bytes (%.2f MB / %.2f GB) : Total Duration : %.2f seconds : Aggregated Throughput : %.2f MB/s / %.2f GB/s",
                backend_label,
                trainer.global_step,
                total_bytes,
                size_mb,
                size_gb,
                duration,
                agg_mb_s,
                agg_gb_s,
            )

    def _save_checkpoint(self, trainer, filepath):
        is_sharded_fsdp = (
            getattr(getattr(trainer, "strategy", None), "name", "") == "fsdp"
            and getattr(getattr(trainer, "strategy", None), "_state_dict_type", "") == "sharded"
        ) or (
            os.getenv("TRAINING_STRATEGY", "").lower() == "fsdp_sharded"
        )

        if is_sharded_fsdp:
            if filepath.endswith(".ckpt"):
                filepath = f"{filepath[:-5]}-rank{trainer.global_rank}.ckpt"
            else:
                filepath = f"{filepath}-rank{trainer.global_rank}"

        is_writer = (trainer.global_rank == 0) or is_sharded_fsdp

        extra_paths_str = os.getenv("ADDITIONAL_CHECKPOINT_PATHS", "")
        all_targets = [filepath]
        if extra_paths_str:
            for raw_p in extra_paths_str.split(","):
                p = raw_p.strip()
                if p and p != filepath:
                    extra_file = os.path.join(p, os.path.basename(filepath))
                    if extra_file not in all_targets:
                        all_targets.append(extra_file)

        staged_tmp_file = None
        if is_writer and len(all_targets) > 1 and os.getenv("SKIP_RAMDISK_STAGING", "0") != "1":
            stage_dir = "/dev/shm" if os.path.exists("/dev/shm") else None
            staged_fd, staged_tmp_file = tempfile.mkstemp(prefix="staged_ckpt_", suffix=".ckpt", dir=stage_dir)
            os.close(staged_fd)
            try:
                stage_start = time.perf_counter()
                self._write_checkpoint_file(trainer, staged_tmp_file)
                stage_dur = time.perf_counter() - stage_start
                logging.info(
                    "[BENCHMARK] Staged checkpoint to local tmp file %s in %.2f seconds for global_step %d from rank %d",
                    staged_tmp_file,
                    stage_dur,
                    trainer.global_step,
                    trainer.global_rank,
                )
            except Exception as e:
                logging.warning("[BENCHMARK] Staging checkpoint failed: %s; falling back to direct save", e)
                if staged_tmp_file and os.path.exists(staged_tmp_file):
                    try:
                        os.remove(staged_tmp_file)
                    except Exception:
                        pass
                staged_tmp_file = None

        try:
            for i, target_path in enumerate(all_targets):
                is_last = (i == len(all_targets) - 1)
                res = self._save_to_single_target(trainer, target_path, is_writer, staged_tmp_file, is_last_target=is_last)
                target_bytes = res[0] if (res and res[0]) else 0
                target_start = res[1] if (res and len(res) > 1) else time.time()
                target_end = res[2] if (res and len(res) > 2) else time.time()
                self._log_aggregated_metrics(trainer, target_bytes, target_start, target_end, target_path)
            
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
                try:
                    dist.barrier()
                except Exception as barrier_err:
                    logging.warning("[BENCHMARK] Barrier synchronization after checkpoint save failed: %s", barrier_err)
        finally:
            if staged_tmp_file and os.path.exists(staged_tmp_file):
                try:
                    os.remove(staged_tmp_file)
                except Exception:
                    pass

    @staticmethod
    def _get_effective_file_bytes(fp):
        try:
            st = os.stat(fp)
            if hasattr(st, "st_blocks") and st.st_blocks > 0:
                blocks_bytes = st.st_blocks * 512
                return min(st.st_size, blocks_bytes)
            return st.st_size
        except Exception:
            return 0

    @classmethod
    def _measure_checkpoint_bytes(cls, filepath):
        if filepath.startswith("gs://"):
            try:
                from gcsfs.extended_gcsfs import ExtendedGcsFileSystem
                fs = ExtendedGcsFileSystem()
                ts_dir = filepath.replace(".ckpt", ".ts_zarr")
                clean_path = ts_dir[5:]
                info_list = fs.find(clean_path, detail=True)
                if isinstance(info_list, dict):
                    total_gcs_bytes = sum(v.get("size", 0) for v in info_list.values() if isinstance(v, dict))
                elif isinstance(info_list, list):
                    total_gcs_bytes = sum(v.get("size", 0) for v in info_list if isinstance(v, dict))
                else:
                    total_gcs_bytes = 0
                if total_gcs_bytes > 0:
                    return total_gcs_bytes
            except Exception:
                pass

        if os.path.exists(filepath):
            if os.path.isfile(filepath):
                sz = cls._get_effective_file_bytes(filepath)
                if sz > 0:
                    return sz
            elif os.path.isdir(filepath):
                total_size = 0
                for dirpath, _, filenames in os.walk(filepath):
                    for f in filenames:
                        fp = os.path.join(dirpath, f)
                        if not os.path.islink(fp):
                            total_size += cls._get_effective_file_bytes(fp)
                if total_size > 0:
                    return total_size

        ts_candidate = filepath.replace(".ckpt", ".ts_zarr")
        if not os.path.exists(ts_candidate):
            ts_candidate = filepath.replace(".ckpt", ".ts_bin")
        if not os.path.exists(ts_candidate):
            ts_candidate = filepath.replace(".ckpt", ".ts_npy")
        if os.path.exists(ts_candidate):
            if os.path.isdir(ts_candidate):
                total_size = 0
                for dirpath, _, filenames in os.walk(ts_candidate):
                    for f in filenames:
                        fp = os.path.join(dirpath, f)
                        if not os.path.islink(fp):
                            total_size += cls._get_effective_file_bytes(fp)
                return total_size

        # Check temporary or partial files (e.g. filepath + ".part", ".tmp")
        for ext in (".part", ".tmp", ".temp", ".partial"):
            candidate = filepath + ext
            if os.path.exists(candidate):
                if os.path.isfile(candidate):
                    return cls._get_effective_file_bytes(candidate)
                elif os.path.isdir(candidate):
                    total_size = 0
                    for dirpath, _, filenames in os.walk(candidate):
                        for f in filenames:
                            fp = os.path.join(dirpath, f)
                            if not os.path.islink(fp):
                                total_size += cls._get_effective_file_bytes(fp)
                    return total_size

        # Fallback for parent directory: measure any partial/tmp files in parent dir
        parent_dir = os.path.dirname(filepath)
        if os.path.exists(parent_dir) and os.path.isdir(parent_dir):
            base_name = os.path.basename(filepath)
            base_prefix = base_name.split(".")[0] if "." in base_name else base_name
            total_size = 0
            try:
                for f in os.listdir(parent_dir):
                    if base_prefix in f or f.startswith(".") or f.endswith((".part", ".tmp", ".temp", ".partial")):
                        fp = os.path.join(parent_dir, f)
                        if os.path.isfile(fp):
                            total_size += os.path.getsize(fp)
                        elif os.path.isdir(fp):
                            for dirpath, _, filenames in os.walk(fp):
                                for fn in filenames:
                                    sub_fp = os.path.join(dirpath, fn)
                                    if not os.path.islink(sub_fp):
                                        total_size += os.path.getsize(sub_fp)
            except Exception:
                pass
            if total_size > 0:
                return total_size

        try:
            fs, path = fsspec.core.url_to_fs(filepath)
            return int(fs.du(path))
        except Exception:
            return 0

    def _remove_checkpoint(self, trainer, filepath):
        is_sharded_fsdp = (
            getattr(getattr(trainer, "strategy", None), "name", "") == "fsdp"
            and getattr(getattr(trainer, "strategy", None), "_state_dict_type", "") == "sharded"
        ) or (
            os.getenv("TRAINING_STRATEGY", "").lower() == "fsdp_sharded"
        )

        if is_sharded_fsdp:
            if filepath.endswith(".ckpt"):
                filepath = f"{filepath[:-5]}-rank{trainer.global_rank}.ckpt"
            else:
                filepath = f"{filepath}-rank{trainer.global_rank}"

        is_deleter = (trainer.global_rank == 0) or is_sharded_fsdp

        if is_deleter:
            logging.info(
                "[BENCHMARK] Checkpoint Delete Start (Deleter Rank %d) : Rank: %d : Step: %d : Path: %s",
                trainer.global_rank,
                trainer.global_rank,
                trainer.global_step,
                filepath,
            )
        else:
            logging.info(
                "[BENCHMARK] Checkpoint Delete Start (Non-Deleting Rank %d - skipped deletion) : Rank: %d : Step: %d : Path: %s",
                trainer.global_rank,
                trainer.global_rank,
                trainer.global_step,
                filepath,
            )
        start_time = time.perf_counter()
        super()._remove_checkpoint(trainer, filepath)
        extra_paths_str = os.getenv("ADDITIONAL_CHECKPOINT_PATHS", "")
        if is_deleter and extra_paths_str:
            for raw_p in extra_paths_str.split(","):
                p = raw_p.strip()
                if p and p != filepath:
                    extra_file = os.path.join(p, os.path.basename(filepath))
                    if extra_file != filepath:
                        try:
                            if extra_file.startswith("gs://"):
                                fs, path = fsspec.core.url_to_fs(extra_file)
                                if fs.exists(path):
                                    fs.rm(path, recursive=True)
                            else:
                                if os.path.isdir(extra_file):
                                    shutil.rmtree(extra_file)
                                elif os.path.exists(extra_file):
                                    os.remove(extra_file)
                        except Exception as e:
                            logging.warning("[BENCHMARK] Failed to remove extra checkpoint %s: %s", extra_file, e)
        duration = time.perf_counter() - start_time

        # Accumulate checkpointing time to be excluded from step time
        for callback in trainer.callbacks:
            if isinstance(callback, StepTimeCallback):
                callback.ckpt_time += duration

        if is_deleter:
            logging.info(
                "[BENCHMARK] Finished deleting checkpoint (Deleter Rank %d) %s in %.2f seconds for global_step %d from rank %d",
                trainer.global_rank,
                filepath,
                duration,
                trainer.global_step,
                trainer.global_rank,
            )
        else:
            logging.info(
                "[BENCHMARK] Finished deleting checkpoint (Non-Deleting Rank %d - skipped deletion) %s in %.2f seconds for global_step %d from rank %d",
                trainer.global_rank,
                filepath,
                duration,
                trainer.global_step,
                trainer.global_rank,
            )


class LoggedDDPStrategy(DDPStrategy):

    def load_checkpoint(self, checkpoint_path, weights_only: bool = False, **kwargs):
        # Under DDP every rank restores, and calc_restore_metrics aggregates the
        # distributed restore as max(end) - min(start) ACROSS ranks. perf_counter
        # is monotonic-from-boot and per-machine, so mixing ranks on different
        # nodes (the default 2-node topology) produces a meaningless span. Log
        # wall-clock time.time() for the absolute Start/End timestamps so the
        # cross-node span is valid (NTP-synced); duration stays on perf_counter,
        # a within-process elapsed measurement.
        logging.info(
            "[BENCHMARK] Checkpoint Restore Start : Rank : %d : Start time: %f seconds : Path: %s",
            self.global_rank,
            time.time(),
            checkpoint_path,
        )
        start_time = time.perf_counter()
        checkpoint = super().load_checkpoint(checkpoint_path, weights_only, **kwargs)
        duration = time.perf_counter() - start_time
        logging.info(
            "[BENCHMARK] Finished restoring checkpoint : Rank : %d : Duration: %.2f seconds : End Time: %.2f seconds : Path: %s",
            self.global_rank,
            duration,
            time.time(),
            checkpoint_path,
        )
        return checkpoint


class LoggedFSDPStrategy(FSDPStrategy):
    """FSDPStrategy with checkpoint restore logging."""

    def load_checkpoint(self, checkpoint_path, *args, **kwargs):
        is_sharded_fsdp = (
            getattr(self, "_state_dict_type", "") == "sharded"
        ) or (
            os.getenv("TRAINING_STRATEGY", "").lower() == "fsdp_sharded"
        )
        if is_sharded_fsdp and checkpoint_path:
            if checkpoint_path.endswith(".ckpt") and not checkpoint_path.endswith(f"-rank{self.global_rank}.ckpt"):
                checkpoint_path = f"{checkpoint_path[:-5]}-rank{self.global_rank}.ckpt"
            elif not checkpoint_path.endswith(f"-rank{self.global_rank}") and not checkpoint_path.endswith(".ckpt"):
                checkpoint_path = f"{checkpoint_path}-rank{self.global_rank}"

        logging.info(
            "[BENCHMARK] Checkpoint Restore Start : Rank : %d : Start time: %f seconds : Path: %s",
            self.global_rank,
            time.time(),
            checkpoint_path,
        )
        start_time = time.perf_counter()
        checkpoint = super().load_checkpoint(checkpoint_path, *args, **kwargs)
        duration = time.perf_counter() - start_time
        logging.info(
            "[BENCHMARK] Finished restoring checkpoint : Rank : %d : Duration: %.2f seconds : End Time: %.2f seconds : Path: %s",
            self.global_rank,
            duration,
            time.time(),
            checkpoint_path,
        )
        return checkpoint


def build_strategy(name):
    """Construct the parallel-training strategy for ``name``
    (ddp|fsdp_sharded|fsdp_full).

    Uses the gloo CPU backend with process-group timeout configured via
    ``DISTRIBUTED_TIMEOUT_SECONDS`` (default 3600s / 1 hour).
    """
    timeout = timedelta(seconds=int(os.getenv("DISTRIBUTED_TIMEOUT_SECONDS", "3600")))
    if name == "ddp":
        # find_unused_parameters=False: the frozen Llama params have
        # requires_grad=False, so only self.trainable participates in DDP
        # autograd, and it is fully used -- no unused parameters.
        return LoggedDDPStrategy(
            process_group_backend="gloo",
            find_unused_parameters=False,
            broadcast_buffers=False,
            timeout=timeout,
        )
    if name in ("fsdp_sharded", "fsdp_full"):
        # fsdp_sharded writes sharded checkpoints; fsdp_full writes consolidated.
        # use_orig_params=True allows mixed requires_grad in the root FSDP unit.
        state_dict_type = "sharded" if name == "fsdp_sharded" else "full"
        return LoggedFSDPStrategy(
            process_group_backend="gloo",
            state_dict_type=state_dict_type,
            auto_wrap_policy={LlamaDecoderLayer},
            use_orig_params=True,
            timeout=timeout,
        )
    raise SystemExit(
        f"Unsupported TRAINING_STRATEGY: {name!r} (use ddp|fsdp_sharded|fsdp_full)."
    )


if __name__ == "__main__":
    # ---- Verify gcsfs is the active fsspec backend for "gs" ----------------
    try:
        fs = fsspec.filesystem("gs")
        logging.info("[BENCHMARK] [SYSTEM CHECK] fsspec 'gs' backend class: %s", type(fs))
        logging.info(
            "[BENCHMARK] [SYSTEM CHECK] If this says 'gcsfs.core.GCSFileSystem', you are using gcsfs."
        )
    except Exception as e:
        logging.info("[BENCHMARK] [SYSTEM CHECK] Failed to load GS filesystem: %s", e)

    # ---- Dataset: HuggingFace streaming parquet -----------------------------
    # This is the GCS read pattern under test.
    logging.info("[BENCHMARK] [INFO] Loading %s dataset", dataset_path)
    logging.info("[BENCHMARK] [INFO] Using HF dataloader")
    load_start = time.perf_counter()
    ds = datasets.load_dataset(
        "parquet",
        data_files=f"{dataset_path}/*.parquet",
        split="train",
        streaming=True,
    )
    logging.info(
        f"[BENCHMARK] [INFO] HF dataloader prepared in {time.perf_counter() - load_start:.4f}s"
    )
    # Shard the streaming dataset across DDP ranks. torchrun sets RANK and
    # WORLD_SIZE before Python starts, so reading from env works at this point
    # (torch.distributed isn't initialized until trainer.fit). Without this,
    # every rank iterates the same parquet files -- 8x the GCS read traffic
    # and duplicate training samples.
    if world_size > 1:
        ds = datasets.distributed.split_dataset_by_node(
            ds,
            rank=int(os.environ["RANK"]),
            world_size=world_size,
        )
    train_loader = DataLoader(
        ds,
        batch_size=per_device_train_batch_size,
        collate_fn=collate_fn,
        num_workers=dataloader_num_workers,
        persistent_workers=dataloader_num_workers > 0,
    )

    # ---- Model: Llama 8B in bf16 --------------------------------------------
    # Uses architecture config for fast instantiation in RAM without disk I/O overhead.
    # Parameter count, tensor shapes, and state_dict checkpoint size (~34.4GB) are 100% identical.
    config = transformers.AutoConfig.from_pretrained(
        model_id, local_files_only=use_local_files_only
    )
    model = transformers.AutoModelForCausalLM.from_config(
        config, torch_dtype=torch.bfloat16
    )

    # ---- Callbacks ----------------------------------------------------------
    callbacks = [DeviceStatsMonitor(cpu_stats=True)]
    if checkpoint_write_path:
        callbacks.append(
            LoggedModelCheckpoint(
                dirpath=f"{checkpoint_write_path}/{run_id}/",
                filename="llama-{epoch:02d}-{step:02d}",
                every_n_train_steps=checkpoint_write_interval,
                save_top_k=checkpoints_to_keep,
                save_last=False,
                monitor="step",
                mode="max",
            )
        )
    callbacks.append(StepTimeCallback())

    # ---- Strategy: DDP or FSDP on CPU via gloo ------------------------------
    strategy = build_strategy(training_strategy)

    # ---- Trainer ------------------------------------------------------------
    # accelerator="cpu" + devices=local_world_size dynamically matches the
    # local rank count (e.g., 4 devices with torchrun --nproc_per_node=4).
    # ``precision="bf16-mixed"`` is the closest CPU equivalent of a GPU "bf16"
    # setting; since training_step doesn't actually forward through
    # the Llama model, CPU bf16 op limitations don't affect correctness.
    trainer = pl.Trainer(
        max_epochs=1,
        num_nodes=num_nodes,
        max_steps=-1 if full_pass else preset_max_steps,
        accumulate_grad_batches=gradient_accumulation_steps,
        precision="bf16-mixed",
        enable_checkpointing=bool(checkpoint_write_path),
        callbacks=callbacks,
        accelerator="cpu",
        devices=local_world_size,
        limit_test_batches=50,
        limit_val_batches=32,
        log_every_n_steps=1,
        strategy=strategy,
        profiler="simple",
        enable_progress_bar=False,
    )

    if checkpoint_load_path:
        logging.info("[BENCHMARK] [INFO] Resuming from checkpoint: %s", checkpoint_load_path)
    else:
        checkpoint_load_path = None

    # Pass the strategy so the module can adapt under FSDP.
    lit_model = LlamaLitModel(model, training_strategy=training_strategy)

    # TODO: These are underscore private APIs, which may break during a Lightning upgrade.
    # We should consider contributing what we need into Lightning.
    # Tracked in: https://github.com/Lightning-AI/pytorch-lightning/pull/21776
    # ==============================================================================
    # PROFILER HOOK 1: Profile the entire setup_data() phase
    # ==============================================================================
    original_setup_data = FitLoop.setup_data

    def profiled_setup_data(self, *args, **kwargs):
        rank = self.trainer.global_rank
        logging.info(f"[BENCHMARK] [RANK {rank}] [PROFILER] FitLoop.setup_data started")
        # We use the PL Profiler so this appears directly in the FIT Profiler Report
        with self.trainer.profiler.profile("FitLoop.setup_data (Data loading)"):
            return original_setup_data(self, *args, **kwargs)

    FitLoop.setup_data = profiled_setup_data

    # ==============================================================================
    # PROFILER HOOK 2: Isolate the iter() call that spawns the workers
    # ==============================================================================
    original_fetcher_iter = _PrefetchDataFetcher.__iter__

    def profiled_fetcher_iter(self):
        start_time = time.perf_counter()

        # This triggers the actual worker forks, gRPC init, and initial file opens
        result = original_fetcher_iter(self)

        duration = time.perf_counter() - start_time
        # We log this to the console immediately for real-time visibility
        rank = os.environ.get("RANK", "0")
        logging.info(
            f"[BENCHMARK] [RANK {rank}] [PROFILER] _PrefetchDataFetcher.__iter__ "
            f"(Worker Spawn and Data Loading) took {duration:.4f} seconds."
        )

        return result

    _PrefetchDataFetcher.__iter__ = profiled_fetcher_iter
    # ==============================================================================

    logging.info("[BENCHMARK] [INFO] Training Started.")

    trainer.fit(lit_model, train_loader, ckpt_path=checkpoint_load_path)
    logging.info("[BENCHMARK] [INFO] Training Completed.")
