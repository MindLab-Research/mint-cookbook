#!/usr/bin/env python3
"""Eval-first chat-quality DPO experiment on MinT."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import random
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# ===== Paths and constants =====

EXPERIMENT_DIR = Path(__file__).resolve().parent
DATA_DIR = EXPERIMENT_DIR / "data"
DEFAULT_LOG_PATH = EXPERIMENT_DIR / "artifacts" / "latest"
DEFAULT_HF_HOME = "~/.cache/huggingface"
DEFAULT_BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_TRAIN_DATA_PATH = DATA_DIR / "train" / "full.jsonl"
LOCAL_ENV_KEYS = {"MINT_API_KEY", "MINT_BASE_URL"}
WHITESPACE_RE = re.compile(r"\s+")
PAIR_TEXT_PREVIEW_CHARS = 240
DEFAULT_BATCH_PAIR_RECORD_LIMIT = 3
PERIODIC_EVAL_SAMPLE_COUNT = 3


# ===== Local env loading =====


def load_local_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export "):
            stripped = stripped[len("export ") :].lstrip()
        if "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        if key not in LOCAL_ENV_KEYS or key in os.environ:
            continue
        os.environ[key] = value.strip().strip('"').strip("'")


load_local_env(EXPERIMENT_DIR / ".env")


# ===== Runtime backend compatibility =====

from transformers import AutoTokenizer

import mint
from mint import types


def create_service_client(*, timeout: float | None = None) -> Any:
    return mint.ServiceClient(
        base_url=os.environ.get("MINT_BASE_URL"),
        timeout=timeout,
    )


def is_sampler_model_path(model_ref: str) -> bool:
    """Recognize sampler-weight references produced by either tinker or mint.

    Experiments that resume from a sampler checkpoint pass the returned
    ``sampler_path`` as ``--base-model``. Both tinker (``tinker://``) and mint
    (``mint://``) emit the same ``/sampler_weights/<id>`` suffix, so the same
    helper catches both shapes regardless of which runtime the experiment chose.
    """
    return model_ref.startswith(("tinker://", "mint://")) and "/sampler_weights/" in model_ref


async def create_sampling_client(service_client: Any, base_model: str) -> Any:
    client_kwargs = {"model_path": base_model} if is_sampler_model_path(base_model) else {"base_model": base_model}
    create_fn = getattr(service_client, "create_sampling_client_async", None)
    if callable(create_fn):
        return await resolve_api_result_async(create_fn(**client_kwargs))
    create_fn = getattr(service_client, "create_sampling_client", None)
    if callable(create_fn):
        return resolve_api_result(create_fn(**client_kwargs))
    raise RuntimeError("Service client must expose create_sampling_client")


def resolve_api_result(maybe_result: Any) -> Any:
    """Normalize sync SDK return shapes into a final result."""
    if hasattr(maybe_result, "result") and callable(maybe_result.result):
        return resolve_api_result(maybe_result.result())
    return maybe_result


async def resolve_api_result_async(maybe_result: Any) -> Any:
    """Normalize the installed SDK's async return shapes into a final result."""
    if hasattr(maybe_result, "result_async") and callable(maybe_result.result_async):
        return await resolve_api_result_async(await maybe_result.result_async())
    if asyncio.isfuture(maybe_result):
        return await resolve_api_result_async(await maybe_result)
    if hasattr(maybe_result, "__await__"):
        return await resolve_api_result_async(await maybe_result)
    if hasattr(maybe_result, "result") and callable(maybe_result.result):
        return await resolve_api_result_async(await asyncio.to_thread(maybe_result.result))
    return maybe_result


# ===== CLI =====


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an eval-first chat-quality DPO experiment on MinT."
    )
    parser.add_argument("--train-data", default=str(DEFAULT_TRAIN_DATA_PATH), help="Path to training JSONL.")
    parser.add_argument(
        "--eval-data",
        default="",
        help=(
            "Path to held-out eval JSONL. Required for --dry-run, --eval-only, "
            "and final eval during training. Use data/eval/smoke.jsonl for local "
            "validation and data/eval/full.jsonl for the held-out benchmark."
        ),
    )
    parser.add_argument("--log-path", default=str(DEFAULT_LOG_PATH))
    parser.add_argument(
        "--base-model",
        default=DEFAULT_BASE_MODEL,
        help=f"Model to evaluate or train. Default: {DEFAULT_BASE_MODEL}.",
    )
    parser.add_argument(
        "--reference-model",
        default="",
        help="Optional reference model for DPO; defaults to --base-model.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--eval-limit", type=int, default=0, help="Cap eval rows; 0 means all.")
    parser.add_argument("--max-concurrent-requests", type=int, default=8)
    parser.add_argument("--mint-timeout", type=float, default=600.0)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=0, help="Cap total steps; 0 means full epochs.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr-schedule", choices=("linear", "cosine", "constant"), default="linear")
    parser.add_argument("--dpo-beta", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--eval-every", type=int, default=0, help="Run held-out eval every N completed steps.")
    parser.add_argument("--save-every", type=int, default=0, help="Save checkpoint every N completed steps.")
    parser.add_argument(
        "--train-metrics-every",
        type=int,
        default=1,
        help=(
            "Append one train/metrics.jsonl row every N completed optim steps (1=every step). "
            "Always records the first train step, the last step, and any step that runs periodic eval "
            "(merged test/* eval scalars)."
        ),
    )
    parser.add_argument(
        "--train-print-every",
        type=int,
        default=1,
        help="Print per-step training lines every N steps (1=every step); 0 disables Step ... stdout (other prints remain).",
    )
    parser.add_argument(
        "--batch-group-key",
        default="group_id",
        help="Dotted key used to spread repeated prompts apart across batches. Empty string disables grouping.",
    )
    parser.add_argument(
        "--allow-partial-batch",
        dest="allow_partial_batch",
        action="store_true",
        default=True,
        help="Keep the last partial batch.",
    )
    parser.add_argument(
        "--drop-partial-batch",
        dest="allow_partial_batch",
        action="store_false",
        help="Drop the last partial batch instead of keeping it.",
    )
    parser.add_argument(
        "--load-checkpoint-path",
        default="",
        help="Saved training state path for a fresh weight-only training start.",
    )
    return parser.parse_args(argv)


def require_eval_data_arg(raw: str, *, dry_run: bool) -> str:
    eval_data_arg = str(raw).strip()
    if eval_data_arg:
        return eval_data_arg
    if dry_run:
        raise RuntimeError("--eval-data is required for --dry-run")
    raise RuntimeError("--eval-data is required")


# ===== Infrastructure helpers =====


class TeeStream:
    """Mirror writes to multiple streams (stdout + console.log file)."""

    def __init__(self, *streams: Any) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()

    def isatty(self) -> bool:
        return any(getattr(s, "isatty", lambda: False)() for s in self.streams)


def prepare_run_dir(log_path: Path) -> Path:
    """Create run directory; symlink artifacts/runs/latest for smoke runs only."""
    log_path.mkdir(parents=True, exist_ok=True)
    if "smoke" in log_path.name:
        latest = log_path.parent / "latest"
        if latest.is_symlink() or latest.is_file():
            latest.unlink()
        elif latest.exists():
            shutil.rmtree(latest)
        latest.parent.mkdir(parents=True, exist_ok=True)
        latest.symlink_to(log_path.resolve(), target_is_directory=True)
    return log_path


def cached_tokenizer_dir(model_name: str) -> Path | None:
    if "/" not in model_name:
        return None
    hf_home = Path(os.environ.get("HF_HOME", DEFAULT_HF_HOME)).expanduser()
    hub_roots = [hf_home if hf_home.name == "hub" else hf_home / "hub"]
    if hf_home.name != "hub" and not (
        hf_home.name == "huggingface" and hf_home.parent.name == ".cache"
    ):
        hub_roots.append(hf_home / ".cache" / "huggingface" / "hub")
    org, repo = model_name.split("/", 1)

    def has_tokenizer_files(path: Path) -> bool:
        if not path.is_dir():
            return False
        files = {f.name for f in path.iterdir() if f.is_file()}
        if "tokenizer_config.json" not in files:
            return False
        return "tokenizer.json" in files or {"vocab.json", "merges.txt"}.issubset(files)

    seen_roots: set[Path] = set()
    for hub_root in hub_roots:
        if hub_root in seen_roots:
            continue
        seen_roots.add(hub_root)
        repo_dir = hub_root / f"models--{org.replace('/', '--')}--{repo.replace('/', '--')}"
        snapshots_dir = repo_dir / "snapshots"
        if not snapshots_dir.exists():
            continue
        candidates = sorted((p for p in snapshots_dir.iterdir() if has_tokenizer_files(p)), reverse=True)
        if candidates:
            return candidates[0]
    return None


def get_tokenizer(client: Any | None, model_name: str) -> Any:
    cache_dir = cached_tokenizer_dir(model_name)
    if cache_dir is not None:
        print(f"@@ tokenizer_cache model={model_name} path={cache_dir}")
        return AutoTokenizer.from_pretrained(str(cache_dir), fast=True, local_files_only=True)
    if client is not None:
        fn = getattr(client, "get_tokenizer", None)
        if callable(fn):
            return fn()
    return AutoTokenizer.from_pretrained(model_name, fast=True)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise RuntimeError(f"Missing JSONL file: {path}")
    rows: list[dict[str, Any]] = []
    # Stream physical file lines so JSON strings containing Unicode line
    # separators such as U+2028 are not split into fake JSONL records.
    with path.open("r", encoding="utf-8") as handle:
        for i, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSONL at {path}:{i}: {exc}") from exc
            if not isinstance(obj, dict):
                raise RuntimeError(f"Expected object at {path}:{i}")
            rows.append(obj)
    return rows


def load_existing_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    return load_jsonl(path)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def scalar_metric_items(metrics: dict[str, Any]) -> dict[str, float]:
    """Filter a metrics dict to only numeric, non-boolean scalars."""
    scalars: dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, bool):
            continue
        try:
            scalars[key] = float(value)
        except (TypeError, ValueError):
            continue
    return scalars


def emit_metric_lines(metrics: dict[str, Any]) -> None:
    for name, value in scalar_metric_items(metrics).items():
        print(f"METRIC {name}={value:.4f}")


def pick_first(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def extract_row_id(row: dict[str, Any], *, fallback: str = "") -> str:
    value = pick_first(row, "pair_id", "id", "example_id")
    if value is None:
        return fallback
    return str(value).strip()


def normalize_text(text: str) -> str:
    return WHITESPACE_RE.sub(" ", str(text)).strip().lower()


def short_text(text: str, *, limit: int = PAIR_TEXT_PREVIEW_CHARS) -> str:
    value = WHITESPACE_RE.sub(" ", str(text)).strip()
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


def preview_messages(messages: list[dict[str, str]]) -> str:
    return " | ".join(f"{msg['role']}: {short_text(msg['content'], limit=80)}" for msg in messages)


def extract_nested_value(data: dict[str, Any], dotted_key: str | None) -> Any:
    if not dotted_key:
        return None
    value: Any = data
    for part in dotted_key.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def normalize_group_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def build_prompt_fingerprint(messages: list[dict[str, str]]) -> str:
    payload = json.dumps(
        [{"role": msg["role"], "content": normalize_text(msg["content"])} for msg in messages],
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


# ===== Training helpers =====


def compute_lr_multiplier(lr_schedule: str, step: int, total_steps: int) -> float:
    if total_steps <= 0:
        return 1.0
    progress = min(max(step / total_steps, 0.0), 1.0)
    if lr_schedule == "linear":
        return max(0.0, 1.0 - progress)
    if lr_schedule == "cosine":
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    if lr_schedule == "constant":
        return 1.0
    raise RuntimeError(f"Unknown lr_schedule: {lr_schedule}")


def compute_total_train_steps(
    num_rows: int,
    *,
    batch_size: int,
    num_epochs: int,
    max_steps: int | None,
    allow_partial_batch: bool,
) -> int:
    if batch_size <= 0:
        raise RuntimeError(f"--batch-size must be positive, got {batch_size}")
    if num_rows <= 0:
        return 0
    full_batches, remainder = divmod(num_rows, batch_size)
    batches_per_epoch = full_batches + (1 if allow_partial_batch and remainder else 0)
    total_steps = batches_per_epoch * max(int(num_epochs), 0)
    if max_steps is None or max_steps <= 0:
        return total_steps
    return min(total_steps, max_steps)


def should_record_train_metrics_row(
    step: int,
    total_steps: int,
    every: int,
    *,
    has_merged_eval: bool,
) -> bool:
    """Decide whether this completed step is flushed to train/metrics.jsonl.

    A step that carries merged periodic-eval metrics (``has_merged_eval=True``)
    is always recorded regardless of cadence so that eval rows are never
    silently dropped from `train/metrics.jsonl`.
    """
    if has_merged_eval:
        return True
    cadence = every if every > 0 else 1
    if step == 1:
        return True
    if total_steps > 0 and step == total_steps:
        return True
    return step % cadence == 0


def should_print_train_step(step: int, total_steps: int, every: int) -> bool:
    if every <= 0:
        return False
    if every == 1:
        return True
    if step == 1 or (total_steps > 0 and step == total_steps):
        return True
    return step % every == 0


def reset_dpo_append_streams(
    log_path: Path,
    *,
    resume_checkpoint: dict[str, Any] | None,
    include_periodic_eval: bool = True,
    include_checkpoints: bool = True,
    include_train_metrics: bool = True,
    include_batch_trace: bool = True,
) -> None:
    """Reset enabled DPO append streams on fresh runs, preserve them on resume.

    Fresh runs truncate/create the enabled JSONL streams and remove any
    disabled stream files left behind by an older run that reused the same
    ``log_path``. This keeps the file set aligned with the active cadence
    flags without leaving stale artifacts in ``run.json``.
    """
    if resume_checkpoint is not None:
        return
    log_path.mkdir(parents=True, exist_ok=True)
    for relative_path, enabled in (
        (Path("eval/periodic.jsonl"), include_periodic_eval),
        (Path("train/checkpoints.jsonl"), include_checkpoints),
        (Path("train/metrics.jsonl"), include_train_metrics),
        (Path("train/batches.jsonl"), include_batch_trace),
    ):
        path = log_path / relative_path
        if enabled:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("", encoding="utf-8")
        elif path.exists():
            path.unlink()


def reset_eval_output_artifacts(log_path: Path) -> None:
    """Remove final-eval files before a fresh run writes new outputs."""
    for path in (
        log_path / "eval" / "examples.jsonl",
        log_path / "eval" / "predictions.jsonl",
        log_path / "eval" / "metrics.json",
    ):
        if path.exists():
            path.unlink()


def extract_api_path(payload: Any) -> str | None:
    if isinstance(payload, str) and payload.strip():
        return payload.strip()
    if isinstance(payload, dict):
        for key in ("path", "state_path", "sampler_path"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None
    for attr_name in ("path", "state_path", "sampler_path"):
        value = getattr(payload, attr_name, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def build_state_save_name(base_model: str, log_path: Path) -> str:
    model_slug = re.sub(r"[^a-z0-9]+", "-", base_model.lower()).strip("-")
    output_slug = re.sub(r"[^a-z0-9]+", "-", log_path.name.lower()).strip("-") or "latest"
    return f"{EXPERIMENT_DIR.name}-{output_slug}-{model_slug}-dpo"


def checkpoint_save_names(base_name: str) -> tuple[str, str]:
    # Runtime checkpoint names share a backing cache namespace, so training and
    # sampler saves must not reuse the same name.
    return f"{base_name}-state", f"{base_name}-sampler"


def step_to_loop_position(step: int, *, n_batches: int, num_epochs: int) -> tuple[int, int]:
    if step < 0:
        raise RuntimeError(f"step must be non-negative, got {step}")
    if n_batches <= 0:
        raise RuntimeError(f"n_batches must be positive, got {n_batches}")
    full_schedule_steps = n_batches * max(num_epochs, 0)
    if step >= full_schedule_steps:
        return num_epochs, 0
    return step // n_batches, step % n_batches


def max_logged_step(path: Path) -> int | None:
    max_step: int | None = None
    for row in load_existing_jsonl(path):
        value = row.get("step")
        if value is None:
            continue
        try:
            parsed = int(value)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"{path} contains non-integer step value `{value}`") from exc
        if parsed < 0:
            raise RuntimeError(f"{path} contains negative step value `{parsed}`")
        max_step = parsed if max_step is None else max(max_step, parsed)
    return max_step


def get_last_resumable_checkpoint(log_path: Path) -> dict[str, Any] | None:
    checkpoints_path = log_path / "train" / "checkpoints.jsonl"
    checkpoint_rows = load_existing_jsonl(checkpoints_path)
    resumable_rows = [row for row in checkpoint_rows if str(row.get("state_path") or "").strip()]
    if not resumable_rows:
        return None
    return resumable_rows[-1]


def validate_resume_contract(
    log_path: Path,
    args: argparse.Namespace,
    *,
    resume_checkpoint: dict[str, Any] | None,
) -> None:
    """Require current args to match the run.json recorded on the resumed run.

    Automatic same-run resume keys off the current ``--log-path``; when a
    resumable ``state_path`` row is present, reject mismatched run-defining
    args before append-only logs continue. Missing ``run.json`` (older
    resume targets) is tolerated.

    Run-scoped append-only streams sliding past the last checkpoint are not
    a resume failure on their own; this contract does not gate resume on
    ``train/metrics.jsonl`` / ``train/batches.jsonl`` / ``eval/periodic.jsonl``
    being exactly aligned to the last checkpoint.
    """
    if resume_checkpoint is None:
        return

    run_json_path = log_path / "run.json"
    if not run_json_path.is_file():
        return
    try:
        run_payload = json.loads(run_json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON in {run_json_path}: {exc}") from exc
    prior_args = run_payload.get("args")
    if not isinstance(prior_args, dict):
        return

    prior_reference_model = str(prior_args.get("reference_model", "") or "").strip() or str(
        prior_args.get("base_model", "") or ""
    ).strip()
    current_reference_model = str(getattr(args, "reference_model", "") or "").strip() or str(
        getattr(args, "base_model", "") or ""
    ).strip()

    current_run_defining_args = {
        "base_model": args.base_model,
        "reference_model": current_reference_model,
        "train_data": args.train_data,
        "eval_data": args.eval_data,
        "seed": args.seed,
        "rank": args.rank,
        "learning_rate": args.learning_rate,
        "num_epochs": args.num_epochs,
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "lr_schedule": args.lr_schedule,
        "dpo_beta": args.dpo_beta,
        "max_length": args.max_length,
        "eval_every": args.eval_every,
        "save_every": args.save_every,
        "batch_group_key": args.batch_group_key,
        "allow_partial_batch": args.allow_partial_batch,
    }
    mismatches: list[str] = []
    for key, current_value in current_run_defining_args.items():
        if key not in prior_args:
            continue
        expected_value = prior_reference_model if key == "reference_model" else prior_args[key]
        if expected_value != current_value:
            mismatches.append(f"{key}: expected {expected_value!r}, got {current_value!r}")
    if mismatches:
        raise RuntimeError(
            "Automatic same-run resume requires the same run-defining args recorded in "
            f"{run_json_path}: " + "; ".join(mismatches)
        )


def resolve_resume_state(
    log_path: Path,
    resume_checkpoint: dict[str, Any] | None,
    *,
    total_steps: int,
    n_batches: int,
    num_epochs: int,
) -> tuple[int, int, int]:
    if resume_checkpoint is None:
        return 0, 0, 0

    parsed_values: dict[str, int] = {}
    for key in ("step", "epoch", "batch"):
        value = resume_checkpoint.get(key)
        try:
            parsed = int(value)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"Resume checkpoint row is missing integer `{key}`") from exc
        if parsed < 0:
            raise RuntimeError(f"Resume checkpoint row contains negative `{key}`={parsed}")
        parsed_values[key] = parsed

    start_step = parsed_values["step"]
    start_epoch_idx = parsed_values["epoch"]
    start_batch_idx = parsed_values["batch"]
    expected_epoch_idx, expected_batch_idx = step_to_loop_position(
        start_step,
        n_batches=n_batches,
        num_epochs=num_epochs,
    )
    if (start_epoch_idx, start_batch_idx) != (expected_epoch_idx, expected_batch_idx):
        raise RuntimeError(
            "Resume checkpoint row has inconsistent loop state: "
            f"step={start_step} implies epoch={expected_epoch_idx}, batch={expected_batch_idx}, "
            f"but row stores epoch={start_epoch_idx}, batch={start_batch_idx}"
        )
    if start_step >= total_steps:
        raise RuntimeError(
            "The latest resumable checkpoint in this log_path already reached this run's configured total_steps"
        )
    return start_step, start_epoch_idx, start_batch_idx


async def load_training_state(training_client: Any, state_path: str) -> None:
    load_fn = getattr(training_client, "load_state_async", None)
    if callable(load_fn):
        await resolve_api_result_async(load_fn(state_path))
        return
    load_fn = getattr(training_client, "load_state", None)
    if not callable(load_fn):
        raise RuntimeError("Training client must expose load_state")
    resolve_api_result(load_fn(state_path))


async def load_training_state_with_optimizer(training_client: Any, state_path: str) -> None:
    load_fn = getattr(training_client, "load_state_with_optimizer_async", None)
    if callable(load_fn):
        await resolve_api_result_async(load_fn(state_path))
        return
    load_fn = getattr(training_client, "load_state_with_optimizer", None)
    if not callable(load_fn):
        raise RuntimeError("Training client must expose load_state_with_optimizer")
    resolve_api_result(load_fn(state_path))


async def create_training_client(
    service_client: Any,
    args: argparse.Namespace,
    *,
    resume_state_path: str | None,
) -> Any:
    load_checkpoint_path = str(getattr(args, "load_checkpoint_path", "") or "").strip()
    client_kwargs = {
        "base_model": args.base_model,
        "rank": int(args.rank),
        "train_mlp": True,
        "train_attn": True,
        "train_unembed": True,
    }
    create_fn = getattr(service_client, "create_lora_training_client_async", None)
    if callable(create_fn):
        training_client = await resolve_api_result_async(create_fn(**client_kwargs))
    else:
        create_fn = getattr(service_client, "create_lora_training_client", None)
        if not callable(create_fn):
            raise RuntimeError("Service client must expose create_lora_training_client")
        training_client = resolve_api_result(create_fn(**client_kwargs))
    if resume_state_path:
        print(f"Resuming from saved state: {resume_state_path}")
        await load_training_state_with_optimizer(training_client, resume_state_path)
    elif load_checkpoint_path:
        print(f"Loading weights from checkpoint: {load_checkpoint_path}")
        await load_training_state(training_client, load_checkpoint_path)
    return training_client


async def save_weights_for_sampling(training_client: Any) -> Any:
    fn = getattr(training_client, "save_weights_and_get_sampling_client_async", None)
    if callable(fn):
        return await resolve_api_result_async(fn(name="eval"))
    fn = getattr(training_client, "save_weights_and_get_sampling_client", None)
    if callable(fn):
        return resolve_api_result(fn(name="eval"))
    raise RuntimeError("Training client must expose save_weights_and_get_sampling_client")


async def save_training_state(training_client: Any, save_name: str) -> str | None:
    save_fn = getattr(training_client, "save_state_async", None)
    if callable(save_fn):
        return extract_api_path(await resolve_api_result_async(save_fn(name=save_name)))
    save_fn = getattr(training_client, "save_state", None)
    if callable(save_fn):
        return extract_api_path(resolve_api_result(save_fn(name=save_name)))
    print("warning: training client does not expose save_state(); skipping checkpoint")
    return None


async def save_sampler_checkpoint(training_client: Any, save_name: str) -> str | None:
    fn = getattr(training_client, "save_weights_for_sampler_async", None)
    if callable(fn):
        return extract_api_path(await resolve_api_result_async(fn(name=save_name)))
    fn = getattr(training_client, "save_weights_for_sampler", None)
    if callable(fn):
        return extract_api_path(resolve_api_result(fn(name=save_name)))
    print("warning: training client does not expose save_weights_for_sampler(); skipping sampler checkpoint")
    return None


def to_float_tensor(value: Any):
    import torch

    if isinstance(value, torch.Tensor):
        return value.flatten().float()
    if hasattr(value, "to_torch"):
        return value.to_torch().flatten().float()
    if hasattr(value, "tolist"):
        return torch.tensor(value.tolist(), dtype=torch.float32).flatten()
    return torch.tensor(value, dtype=torch.float32).flatten()


def weighted_sequence_score(logprobs: Any, weights: Any):
    import torch

    seq = to_float_tensor(logprobs)
    weight_tensor = to_float_tensor(weights)
    if seq.numel() != weight_tensor.numel():
        common = min(seq.numel(), weight_tensor.numel())
        seq = seq[:common]
        weight_tensor = weight_tensor[:common]
    if seq.numel() == 0:
        return torch.tensor(0.0)
    return torch.dot(seq.float(), weight_tensor.float())


def compute_dpo_loss(
    chosen_logprobs: list[Any],
    rejected_logprobs: list[Any],
    chosen_ref_logprobs: list[Any],
    rejected_ref_logprobs: list[Any],
    dpo_beta: float,
) -> tuple[Any, dict[str, float]]:
    import torch
    import torch.nn.functional as F

    chosen_log_ratio = torch.stack([
        chosen - chosen_ref
        for chosen, chosen_ref in zip(chosen_logprobs, chosen_ref_logprobs, strict=True)
    ])
    rejected_log_ratio = torch.stack([
        rejected - rejected_ref
        for rejected, rejected_ref in zip(rejected_logprobs, rejected_ref_logprobs, strict=True)
    ])
    losses = -F.logsigmoid(float(dpo_beta) * (chosen_log_ratio - rejected_log_ratio))
    loss = losses.mean()
    accuracy = (chosen_log_ratio > rejected_log_ratio).float().mean().item()
    chosen_rewards = float(dpo_beta) * chosen_log_ratio
    rejected_rewards = float(dpo_beta) * rejected_log_ratio
    margin = (chosen_rewards - rejected_rewards).mean().item()
    metrics = {
        "dpo_loss": float(loss.item()),
        "accuracy": float(accuracy),
        "margin": float(margin),
        "chosen_reward": float(chosen_rewards.mean().item()),
        "rejected_reward": float(rejected_rewards.mean().item()),
    }
    return loss, metrics


async def compute_logprobs(client: Any, model_input: Any) -> list[float]:
    fn = getattr(client, "compute_logprobs_async", None)
    if callable(fn):
        payload = await resolve_api_result_async(fn(model_input))
    else:
        fn = getattr(client, "compute_logprobs", None)
        if not callable(fn):
            raise RuntimeError("Client must expose compute_logprobs or compute_logprobs_async")
        payload = await asyncio.to_thread(lambda: resolve_api_result(fn(model_input)))
    if hasattr(payload, "tolist"):
        payload = payload.tolist()
    if isinstance(payload, dict) and "logprobs" in payload:
        payload = payload["logprobs"]
    if not isinstance(payload, list):
        raise RuntimeError(f"Unexpected logprob payload type: {type(payload).__name__}")
    values: list[float] = []
    for index, value in enumerate(payload):
        if value is None:
            if index == 0:
                # Some MinT backends return a BOS slot with no logprob.
                continue
            raise RuntimeError(f"Unexpected null logprob at position {index}")
        values.append(float(value))
    return values


async def compute_logprobs_with_semaphore(
    client: Any,
    model_input: Any,
    semaphore: asyncio.Semaphore,
) -> list[float]:
    async with semaphore:
        return await compute_logprobs(client, model_input)


def align_logprob_sequence(logprobs: list[float], *, target_length: int) -> list[float]:
    if target_length <= 0:
        return []
    if len(logprobs) == target_length:
        return logprobs
    if len(logprobs) == target_length + 1:
        return logprobs[1:]
    if len(logprobs) > target_length:
        return logprobs[-target_length:]
    raise RuntimeError(
        f"Logprob sequence shorter than target length: got {len(logprobs)}, need {target_length}"
    )


def tensor_like_length(value: Any) -> int:
    if hasattr(value, "numel") and callable(value.numel):
        return int(value.numel())
    shape = getattr(value, "shape", None)
    if isinstance(shape, (list, tuple)) and shape:
        total = 1
        for dim in shape:
            total *= int(dim)
        return total
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, dict) and "data" in value:
        value = value["data"]
    if not hasattr(value, "__len__"):
        raise RuntimeError(f"Cannot infer tensor-like length from {type(value).__name__}")
    return int(len(value))


# ===== Task-specific helpers =====


def normalize_message_list(value: Any, *, default_role: str | None = None) -> list[dict[str, str]]:
    if isinstance(value, str):
        role = default_role or "user"
        text = value.strip()
        if not text:
            return []
        return [{"role": role, "content": text}]
    if not isinstance(value, list):
        raise RuntimeError(f"Expected string or list of messages, got {type(value).__name__}")
    messages: list[dict[str, str]] = []
    for index, item in enumerate(value, start=1):
        if isinstance(item, str):
            role = default_role or "assistant"
            content = item.strip()
        elif isinstance(item, dict):
            role = str(item.get("role") or default_role or "assistant").strip()
            content = str(item.get("content") or "").strip()
        else:
            raise RuntimeError(f"Unsupported message item at {index}: {type(item).__name__}")
        if not role or not content:
            raise RuntimeError(f"Empty role/content at message index {index}")
        messages.append({"role": role, "content": content})
    return messages


def comparison_to_pair(example: dict[str, Any], *, fallback_id: str) -> dict[str, Any] | None:
    comparison = example.get("comparison") if isinstance(example.get("comparison"), dict) else example
    label = str(example.get("label") or comparison.get("label") or "A").strip().upper()
    if label not in {"A", "B"}:
        return None
    prompt_conversation = normalize_message_list(comparison.get("prompt_conversation") or [], default_role="user")
    completion_a = normalize_message_list(comparison.get("completion_A") or [], default_role="assistant")
    completion_b = normalize_message_list(comparison.get("completion_B") or [], default_role="assistant")
    if not prompt_conversation or not completion_a or not completion_b:
        return None
    chosen = completion_a if label == "A" else completion_b
    rejected = completion_b if label == "A" else completion_a
    return {
        "pair_id": extract_row_id(example, fallback=fallback_id),
        "messages": prompt_conversation,
        "chosen": chosen,
        "rejected": rejected,
        "group_id": pick_first(example, "group_id") or pick_first(example.get("metadata") or {}, "original_idx"),
        "source": pick_first(example, "source") or pick_first(example.get("metadata") or {}, "source") or "legacy_comparison",
        "metadata": dict(example.get("metadata") or {}),
    }


def normalize_preference_row(example: dict[str, Any], *, fallback_id: str) -> dict[str, Any]:
    canonical: dict[str, Any] | None = None
    if "messages" in example and "chosen" in example and "rejected" in example:
        canonical = {
            "pair_id": extract_row_id(example, fallback=fallback_id),
            "messages": normalize_message_list(example["messages"], default_role="user"),
            "chosen": normalize_message_list(example["chosen"], default_role="assistant"),
            "rejected": normalize_message_list(example["rejected"], default_role="assistant"),
            "group_id": example.get("group_id"),
            "source": pick_first(example, "source") or "messages_pair",
            "metadata": dict(example.get("metadata") or {}),
        }
    elif "prompt" in example and "chosen" in example and "rejected" in example:
        canonical = {
            "pair_id": extract_row_id(example, fallback=fallback_id),
            "messages": normalize_message_list(example["prompt"], default_role="user"),
            "chosen": normalize_message_list(example["chosen"], default_role="assistant"),
            "rejected": normalize_message_list(example["rejected"], default_role="assistant"),
            "group_id": example.get("group_id"),
            "source": pick_first(example, "source") or "prompt_pair",
            "metadata": dict(example.get("metadata") or {}),
        }
    elif "prompt_conversation" in example and "completion_A" in example and "completion_B" in example:
        canonical = comparison_to_pair(example, fallback_id=fallback_id)
    elif isinstance(example.get("comparison"), dict):
        canonical = comparison_to_pair(example, fallback_id=fallback_id)
    if canonical is None:
        raise RuntimeError(f"Unsupported preference row keys: {sorted(example.keys())}")
    if not canonical["messages"]:
        raise RuntimeError("Preference row has empty messages")
    if not canonical["chosen"] or not canonical["rejected"]:
        raise RuntimeError("Preference row must have non-empty chosen and rejected")
    prompt_fingerprint = build_prompt_fingerprint(canonical["messages"])
    canonical["prompt_fingerprint"] = prompt_fingerprint
    if not canonical.get("group_id"):
        canonical["group_id"] = prompt_fingerprint
    return canonical


def normalize_preference_rows(path: Path) -> list[dict[str, Any]]:
    rows = load_jsonl(path)
    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        normalized.append(normalize_preference_row(row, fallback_id=f"{path.stem}-{index:06d}"))
    if not normalized:
        raise RuntimeError(f"Empty preference data: {path}")
    return normalized


def audit_overlap(train_ids: list[str] | None, eval_ids: list[str]) -> dict[str, Any]:
    if train_ids is None:
        return {
            "train_status": "missing",
            "train_rows": 0,
            "eval_rows": len(eval_ids),
            "overlap_count": 0,
            "overlap_preview": [],
        }
    overlap = sorted(set(train_ids) & set(eval_ids))
    return {
        "train_status": "ok",
        "train_rows": len(train_ids),
        "eval_rows": len(eval_ids),
        "overlap_count": len(overlap),
        "overlap_preview": overlap[:10],
    }


def build_epoch_batches(
    rows: list[dict[str, Any]],
    *,
    batch_size: int,
    allow_partial_batch: bool,
    batch_group_key: str | None,
    seed: int,
) -> list[list[dict[str, Any]]]:
    if batch_size <= 0:
        raise RuntimeError(f"--batch-size must be positive, got {batch_size}")
    if not rows:
        return []
    dataset_size = len(rows)
    full_batches, remainder = divmod(dataset_size, batch_size)
    capacities = [batch_size] * full_batches
    if allow_partial_batch and remainder:
        capacities.append(remainder)
    if not capacities:
        return []
    ordered_rows = list(rows)
    rng = random.Random(seed)
    if not batch_group_key:
        rng.shuffle(ordered_rows)
        batches: list[list[dict[str, Any]]] = []
        cursor = 0
        for capacity in capacities:
            batches.append(ordered_rows[cursor : cursor + capacity])
            cursor += capacity
        return batches
    grouped: dict[Any, list[dict[str, Any]]] = {}
    for row in ordered_rows:
        group_value = extract_nested_value(row, batch_group_key)
        if group_value is None:
            group_value = row.get("group_id") or row.get("prompt_fingerprint")
        grouped.setdefault(normalize_group_value(group_value), []).append(row)
    groups = list(grouped.values())
    rng.shuffle(groups)
    for group in groups:
        rng.shuffle(group)
    batches = [[] for _ in capacities]
    remaining = capacities[:]
    for group in groups:
        used_batches: set[int] = set()
        for row in group:
            candidates = [
                index
                for index, slots in enumerate(remaining)
                if slots > 0 and index not in used_batches
            ]
            if not candidates:
                candidates = [index for index, slots in enumerate(remaining) if slots > 0]
            if not candidates:
                raise RuntimeError("Ran out of batch capacity while grouping preference pairs")
            max_remaining = max(remaining[index] for index in candidates)
            top_candidates = [index for index in candidates if remaining[index] == max_remaining]
            chosen_index = rng.choice(top_candidates)
            batches[chosen_index].append(row)
            remaining[chosen_index] -= 1
            used_batches.add(chosen_index)
    for batch in batches:
        rng.shuffle(batch)
    return [batch for batch in batches if batch]


def coerce_token_list(tokens: Any) -> list[int]:
    if hasattr(tokens, "tolist"):
        tokens = tokens.tolist()
    if isinstance(tokens, dict) and "input_ids" in tokens:
        tokens = tokens["input_ids"]
    if hasattr(tokens, "input_ids"):
        tokens = tokens.input_ids
    if isinstance(tokens, list) and tokens and isinstance(tokens[0], list):
        tokens = tokens[0]
    if not isinstance(tokens, list):
        raise RuntimeError(f"Unexpected token container: {type(tokens).__name__}")
    return [int(token) for token in tokens]


def render_messages_as_text(messages: list[dict[str, str]]) -> str:
    return "\n\n".join(
        f"{message['role']}:\n{message['content']}" for message in messages
    )


def build_generation_prompt_tokens(tokenizer: Any, messages: list[dict[str, str]]) -> list[int]:
    apply_fn = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply_fn):
        tokens = apply_fn(messages, tokenize=True, add_generation_prompt=True)
        if hasattr(tokens, "input_ids"):
            tokens = tokens.input_ids
        elif isinstance(tokens, dict) and "input_ids" in tokens:
            tokens = tokens["input_ids"]
        if hasattr(tokens, "tolist"):
            tokens = tokens.tolist()
        if isinstance(tokens, list) and tokens and isinstance(tokens[0], list):
            tokens = tokens[0]
        if isinstance(tokens, list):
            return [int(t) for t in tokens]
        raise RuntimeError(f"Unexpected chat template output: {type(tokens).__name__}")
    text = "\n\n".join(f"{m['role']}:\n{m['content']}" for m in messages)
    return [
        int(t)
        for t in tokenizer.encode(text + "\n\nassistant:", add_special_tokens=True)
    ]


def build_full_conversation_tokens(
    tokenizer: Any,
    messages: list[dict[str, str]],
    completion: list[dict[str, str]],
) -> tuple[list[int], int]:
    apply_fn = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply_fn):
        prompt_tokens = build_generation_prompt_tokens(tokenizer, messages)
        full_tokens = coerce_token_list(apply_fn([*messages, *completion], tokenize=True, add_generation_prompt=False))
    else:
        prompt_tokens = build_generation_prompt_tokens(tokenizer, messages)
        full_tokens = coerce_token_list(tokenizer.encode(render_messages_as_text([*messages, *completion]), add_special_tokens=True))
    prefix_len = 0
    for left, right in zip(prompt_tokens, full_tokens, strict=False):
        if left != right:
            break
        prefix_len += 1
    if prefix_len <= 0 or len(full_tokens) <= prefix_len:
        raise RuntimeError("Could not isolate completion tokens for preference datum")
    return full_tokens, prefix_len


def build_preference_datum(
    tokenizer: Any,
    messages: list[dict[str, str]],
    completion: list[dict[str, str]],
    *,
    max_length: int,
) -> tuple[Any, Any, int]:
    full_tokens, prefix_len = build_full_conversation_tokens(tokenizer, messages, completion)
    if max_length > 0 and len(full_tokens) > max_length:
        full_tokens = full_tokens[:max_length]
    weights = [0.0] * min(prefix_len, len(full_tokens)) + [1.0] * max(len(full_tokens) - prefix_len, 0)
    if len(full_tokens) < 2 or sum(weights[1:]) <= 0:
        raise RuntimeError("Preference datum does not contain enough completion tokens")
    datum = types.Datum(
        model_input=types.ModelInput.from_ints(tokens=full_tokens[:-1]),
        loss_fn_inputs={
            "target_tokens": full_tokens[1:],
            "weights": weights[1:],
        },
    )
    return datum, types.ModelInput.from_ints(tokens=full_tokens), len(full_tokens)


def build_pair_payload(tokenizer: Any, row: dict[str, Any], *, max_length: int) -> dict[str, Any]:
    chosen_datum, chosen_input, chosen_tokens = build_preference_datum(
        tokenizer,
        row["messages"],
        row["chosen"],
        max_length=max_length,
    )
    rejected_datum, rejected_input, rejected_tokens = build_preference_datum(
        tokenizer,
        row["messages"],
        row["rejected"],
        max_length=max_length,
    )
    return {
        "row": row,
        "pair_id": row["pair_id"],
        "chosen_datum": chosen_datum,
        "rejected_datum": rejected_datum,
        "chosen_input": chosen_input,
        "rejected_input": rejected_input,
        "chosen_tokens": chosen_tokens,
        "rejected_tokens": rejected_tokens,
    }


def score_reference_pair_payloads(
    ref_logprob_seqs: list[list[float]],
    pair_payloads: list[dict[str, Any]],
) -> tuple[list[Any], list[Any]]:
    chosen_scores: list[Any] = []
    rejected_scores: list[Any] = []
    for payload_index, payload in enumerate(pair_payloads):
        chosen_weights = payload["chosen_datum"].loss_fn_inputs["weights"]
        rejected_weights = payload["rejected_datum"].loss_fn_inputs["weights"]
        chosen_scores.append(
            weighted_sequence_score(
                align_logprob_sequence(
                    ref_logprob_seqs[payload_index * 2],
                    target_length=tensor_like_length(chosen_weights),
                ),
                chosen_weights,
            )
        )
        rejected_scores.append(
            weighted_sequence_score(
                align_logprob_sequence(
                    ref_logprob_seqs[payload_index * 2 + 1],
                    target_length=tensor_like_length(rejected_weights),
                ),
                rejected_weights,
            )
        )
    return chosen_scores, rejected_scores


def build_dpo_batch_trace_record(
    batch_rows: list[dict[str, Any]],
    *,
    limit: int = DEFAULT_BATCH_PAIR_RECORD_LIMIT,
) -> dict[str, Any]:
    return {
        "num_pairs": len(batch_rows),
        "pairs": [
            {
                "pair_id": row["pair_id"],
                "group_id": row.get("group_id"),
                "prompt_preview": preview_messages(row["messages"]),
                "chosen_preview": short_text(row["chosen"][0]["content"]),
                "rejected_preview": short_text(row["rejected"][0]["content"]),
            }
            for row in batch_rows[:limit]
        ],
    }


# ===== Task-specific adapters =====


def normalize_eval_rows(path: Path) -> list[dict[str, Any]]:
    return normalize_preference_rows(path)


def normalize_train_rows(path: Path) -> list[dict[str, Any]]:
    return normalize_preference_rows(path)


def eval_example_artifact_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "pair_id": row["pair_id"],
        "group_id": row.get("group_id"),
        "source": row.get("source", ""),
        "messages": row["messages"],
        "chosen": row["chosen"],
        "rejected": row["rejected"],
        "prompt_fingerprint": row["prompt_fingerprint"],
    }


def prediction_artifact_rows(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{key: value for key, value in prediction.items() if not str(key).startswith("_")} for prediction in predictions]


def compute_eval_metrics(predictions: list[dict[str, Any]]) -> dict[str, float]:
    total = len(predictions)
    if total == 0:
        return {
            "eval_pair_accuracy": 0.0,
            "eval_margin": 0.0,
            "eval_chosen_score": 0.0,
            "eval_rejected_score": 0.0,
            "eval_num_pairs": 0.0,
        }
    accuracy = sum(1 for prediction in predictions if prediction.get("correct")) / total
    margin = sum(float(prediction["margin"]) for prediction in predictions) / total
    chosen_score = sum(float(prediction["chosen_score"]) for prediction in predictions) / total
    rejected_score = sum(float(prediction["rejected_score"]) for prediction in predictions) / total
    return {
        "eval_pair_accuracy": float(accuracy),
        "eval_margin": float(margin),
        "eval_chosen_score": float(chosen_score),
        "eval_rejected_score": float(rejected_score),
        "eval_num_pairs": float(total),
    }


# ===== Runtime entrypoints =====


def run_dry_run(
    args: argparse.Namespace,
    *,
    train_rows: list[dict[str, Any]] | None,
    eval_rows: list[dict[str, Any]],
    overlap: dict[str, Any],
) -> int:
    train_count = len(train_rows) if train_rows else 0
    unique_train_groups = len({str(row.get("group_id")) for row in train_rows}) if train_rows else 0
    unique_eval_groups = len({str(row.get("group_id")) for row in eval_rows})
    train_prompt_chars = [
        len(render_messages_as_text(row["messages"]))
        for row in (train_rows or [])
    ]
    eval_prompt_chars = [len(render_messages_as_text(row["messages"])) for row in eval_rows]
    train_completion_chars = [
        len(render_messages_as_text(row["chosen"])) + len(render_messages_as_text(row["rejected"]))
        for row in (train_rows or [])
    ]
    eval_completion_chars = [
        len(render_messages_as_text(row["chosen"])) + len(render_messages_as_text(row["rejected"]))
        for row in eval_rows
    ]
    print(f"dry_run: train_rows={train_count} eval_rows={len(eval_rows)}")
    print(f"dry_run: unique_train_groups={unique_train_groups} unique_eval_groups={unique_eval_groups}")
    print(f"dry_run: overlap_count={overlap['overlap_count']}")
    if train_prompt_chars:
        print(
            "dry_run: train_prompt_chars_mean="
            f"{sum(train_prompt_chars) / len(train_prompt_chars):.1f} "
            f"train_completion_chars_mean={sum(train_completion_chars) / len(train_completion_chars):.1f}"
        )
    print(
        "dry_run: eval_prompt_chars_mean="
        f"{sum(eval_prompt_chars) / len(eval_prompt_chars):.1f} "
        f"eval_completion_chars_mean={sum(eval_completion_chars) / len(eval_completion_chars):.1f}"
    )
    if train_rows:
        print(f"dry_run: first_train_prompt={preview_messages(train_rows[0]['messages'])}")
    print(f"dry_run: first_eval_prompt={preview_messages(eval_rows[0]['messages'])}")
    print("dry_run: ok")
    return 0


async def score_pair(
    client: Any,
    tokenizer: Any,
    row: dict[str, Any],
    *,
    max_length: int,
    semaphore: asyncio.Semaphore,
) -> tuple[dict[str, Any], dict[str, Any]]:
    async with semaphore:
        payload = build_pair_payload(tokenizer, row, max_length=max_length)
        chosen_logprobs, rejected_logprobs = await asyncio.gather(
            compute_logprobs(client, payload["chosen_input"]),
            compute_logprobs(client, payload["rejected_input"]),
        )
    chosen_aligned = align_logprob_sequence(
        chosen_logprobs,
        target_length=tensor_like_length(payload["chosen_datum"].loss_fn_inputs["weights"]),
    )
    rejected_aligned = align_logprob_sequence(
        rejected_logprobs,
        target_length=tensor_like_length(payload["rejected_datum"].loss_fn_inputs["weights"]),
    )
    chosen_score = weighted_sequence_score(
        chosen_aligned,
        payload["chosen_datum"].loss_fn_inputs["weights"],
    ).item()
    rejected_score = weighted_sequence_score(
        rejected_aligned,
        payload["rejected_datum"].loss_fn_inputs["weights"],
    ).item()
    margin = chosen_score - rejected_score
    return eval_example_artifact_row(row), {
        "pair_id": row["pair_id"],
        "group_id": row.get("group_id"),
        "chosen_score": float(chosen_score),
        "rejected_score": float(rejected_score),
        "margin": float(margin),
        "correct": bool(margin > 0),
        "prompt_preview": preview_messages(row["messages"]),
    }


async def run_eval(
    sampler: Any,
    tokenizer: Any,
    eval_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, float]]:
    semaphore = asyncio.Semaphore(max(int(args.max_concurrent_requests), 1))
    eval_pairs = await asyncio.gather(
        *(
            score_pair(
                sampler,
                tokenizer,
                row,
                max_length=int(args.max_length),
                semaphore=semaphore,
            )
            for row in eval_rows
        )
    )
    eval_examples = [example for example, _ in eval_pairs]
    predictions = [prediction for _, prediction in eval_pairs]
    metrics = compute_eval_metrics(predictions)
    return eval_examples, predictions, metrics


async def run_train(
    training_client: Any,
    reference_client: Any,
    tokenizer: Any,
    reference_tokenizer: Any,
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    args: argparse.Namespace,
    log_path: Path,
    *,
    resume_checkpoint: dict[str, Any] | None,
) -> dict[str, float]:
    num_epochs = int(args.num_epochs)
    batch_size = int(args.batch_size)
    max_steps = int(args.max_steps) if int(args.max_steps) > 0 else None
    total_steps = compute_total_train_steps(
        len(train_rows),
        batch_size=batch_size,
        num_epochs=num_epochs,
        max_steps=max_steps,
        allow_partial_batch=bool(args.allow_partial_batch),
    )
    if total_steps <= 0:
        raise RuntimeError("No train steps available; check train data and batch settings")
    full_batches, remainder = divmod(len(train_rows), batch_size)
    n_batches = full_batches + (1 if bool(args.allow_partial_batch) and remainder else 0)
    start_step, start_epoch_idx, start_batch_idx = resolve_resume_state(
        log_path,
        resume_checkpoint,
        total_steps=total_steps,
        n_batches=n_batches,
        num_epochs=num_epochs,
    )
    if start_step > 0:
        print(
            "Resume bookkeeping: "
            f"start_step={start_step} "
            f"start_epoch={start_epoch_idx} "
            f"start_batch={start_batch_idx} "
            f"total_steps={total_steps}"
        )
    base_lr = float(args.learning_rate)
    eval_every = int(args.eval_every)
    save_every = int(args.save_every)
    state_name_prefix = build_state_save_name(args.base_model, log_path)
    semaphore = asyncio.Semaphore(max(int(args.max_concurrent_requests), 1))
    last_step_metrics: dict[str, Any] = {}
    train_metrics_every = int(args.train_metrics_every)
    train_metrics_enabled = train_metrics_every > 0
    step = start_step
    train_started_at = time.time()

    for epoch_idx in range(start_epoch_idx, num_epochs):
        batches = build_epoch_batches(
            train_rows,
            batch_size=batch_size,
            allow_partial_batch=bool(args.allow_partial_batch),
            batch_group_key=args.batch_group_key.strip() or None,
            seed=args.seed + epoch_idx,
        )
        batch_start_idx = start_batch_idx if epoch_idx == start_epoch_idx else 0
        for batch_idx in range(batch_start_idx, len(batches)):
            if max_steps is not None and step >= max_steps:
                break
            batch_rows = batches[batch_idx]
            step_started_at = time.time()
            learning_rate = base_lr * compute_lr_multiplier(str(args.lr_schedule), step, total_steps)
            pair_payloads = [build_pair_payload(tokenizer, row, max_length=int(args.max_length)) for row in batch_rows]
            reference_pair_payloads = pair_payloads
            if reference_tokenizer is not tokenizer:
                reference_pair_payloads = [
                    build_pair_payload(reference_tokenizer, row, max_length=int(args.max_length))
                    for row in batch_rows
                ]
            datums: list[Any] = []
            reference_inputs: list[Any] = []
            num_tokens = 0
            for payload in pair_payloads:
                datums.extend([payload["chosen_datum"], payload["rejected_datum"]])
                num_tokens += int(payload["chosen_tokens"]) + int(payload["rejected_tokens"])
            for payload in reference_pair_payloads:
                reference_inputs.extend([payload["chosen_input"], payload["rejected_input"]])
            ref_logprob_seqs = await asyncio.gather(
                *(
                    compute_logprobs_with_semaphore(reference_client, model_input, semaphore)
                    for model_input in reference_inputs
                )
            )
            chosen_ref_logprobs, rejected_ref_logprobs = score_reference_pair_payloads(
                ref_logprob_seqs,
                reference_pair_payloads,
            )

            def dpo_loss_fn(batch: list[Any], logprobs_list: list[Any]):
                chosen_logprobs = []
                rejected_logprobs = []
                for pair_index in range(0, len(batch), 2):
                    chosen_batch = batch[pair_index]
                    rejected_batch = batch[pair_index + 1]
                    chosen_logprobs.append(weighted_sequence_score(logprobs_list[pair_index], chosen_batch.loss_fn_inputs["weights"]))
                    rejected_logprobs.append(weighted_sequence_score(logprobs_list[pair_index + 1], rejected_batch.loss_fn_inputs["weights"]))
                return compute_dpo_loss(
                    chosen_logprobs=chosen_logprobs,
                    rejected_logprobs=rejected_logprobs,
                    chosen_ref_logprobs=chosen_ref_logprobs,
                    rejected_ref_logprobs=rejected_ref_logprobs,
                    dpo_beta=float(args.dpo_beta),
                )

            fb_fn = getattr(training_client, "forward_backward_custom_async", None)
            if callable(fb_fn):
                backward_result = await resolve_api_result_async(fb_fn(datums, dpo_loss_fn))
            else:
                fb_fn = getattr(training_client, "forward_backward_custom", None)
                if not callable(fb_fn):
                    raise RuntimeError("Training client must expose forward_backward_custom")
                backward_result = await asyncio.to_thread(lambda: resolve_api_result(fb_fn(datums, dpo_loss_fn)))
            batch_metrics = dict(getattr(backward_result, "metrics", {}) or {})
            optim_fn = getattr(training_client, "optim_step_async", None)
            if callable(optim_fn):
                await resolve_api_result_async(optim_fn(types.AdamParams(learning_rate=learning_rate)))
            else:
                optim_fn = getattr(training_client, "optim_step", None)
                if not callable(optim_fn):
                    raise RuntimeError("Training client must expose optim_step or optim_step_async")
                await asyncio.to_thread(lambda: resolve_api_result(optim_fn(types.AdamParams(learning_rate=learning_rate))))

            step += 1
            step_time = time.time() - step_started_at
            progress = step / total_steps if total_steps > 0 else 1.0
            tokens_per_second = num_tokens / step_time if step_time > 0 else 0.0
            step_eval_metrics: dict[str, Any] | None = None

            if save_every > 0 and step % save_every == 0:
                checkpoint_name = f"{state_name_prefix}-step-{step:06d}"
                state_save_name, sampler_save_name = checkpoint_save_names(checkpoint_name)
                checkpoint_epoch_idx, checkpoint_batch_idx = step_to_loop_position(
                    step,
                    n_batches=n_batches,
                    num_epochs=num_epochs,
                )
                checkpoint_row: dict[str, Any] = {
                    "name": checkpoint_name,
                    "step": step,
                    "epoch": checkpoint_epoch_idx,
                    "batch": checkpoint_batch_idx,
                }
                state_path = await save_training_state(training_client, state_save_name)
                if state_path:
                    checkpoint_row["state_path"] = state_path
                sampler_path = await save_sampler_checkpoint(training_client, sampler_save_name)
                if sampler_path:
                    checkpoint_row["sampler_path"] = sampler_path
                if len(checkpoint_row) > 4:
                    append_jsonl(log_path / "train" / "checkpoints.jsonl", checkpoint_row)

            if eval_every > 0 and step % eval_every == 0:
                sampler = await save_weights_for_sampling(training_client)
                _, periodic_predictions, periodic_metrics = await run_eval(sampler, tokenizer, eval_rows, args)
                step_eval_metrics = periodic_metrics
                sample_predictions = [
                    {
                        key: value
                        for key, value in prediction.items()
                        if key in {"pair_id", "margin", "correct", "chosen_score", "rejected_score", "prompt_preview"}
                    }
                    for prediction in periodic_predictions[:PERIODIC_EVAL_SAMPLE_COUNT]
                ]
                append_jsonl(
                    log_path / "eval" / "periodic.jsonl",
                    {"step": step, **scalar_metric_items(periodic_metrics), "samples": sample_predictions},
                )

            step_record: dict[str, Any] = {
                "step": step,
                "epoch": epoch_idx,
                "dpo_loss": float(batch_metrics.get("dpo_loss", float("nan"))),
                "accuracy": float(batch_metrics.get("accuracy", 0.0)),
                "margin": float(batch_metrics.get("margin", 0.0)),
                "chosen_reward": float(batch_metrics.get("chosen_reward", 0.0)),
                "rejected_reward": float(batch_metrics.get("rejected_reward", 0.0)),
                "learning_rate": learning_rate,
                "num_pairs": len(batch_rows),
                "num_sequences": len(batch_rows),
                "num_tokens": num_tokens,
                "step_time_seconds": round(step_time, 3),
                "tokens_per_second": round(tokens_per_second, 1),
                "progress": round(progress, 4),
                "lora_rank": int(args.rank),
            }
            if step_eval_metrics:
                for name, value in scalar_metric_items(step_eval_metrics).items():
                    step_record[f"test/{name}"] = value
            if train_metrics_enabled and should_record_train_metrics_row(
                step,
                total_steps,
                train_metrics_every,
                has_merged_eval=step_eval_metrics is not None,
            ):
                append_jsonl(log_path / "train" / "metrics.jsonl", step_record)
            if train_metrics_enabled:
                append_jsonl(
                    log_path / "train" / "batches.jsonl",
                    {"step": step, "epoch": epoch_idx, "batch": batch_idx, **build_dpo_batch_trace_record(batch_rows)},
                )
            if should_print_train_step(step, total_steps, int(args.train_print_every)):
                summary = (
                    f"Step {step}/{total_steps}: dpo_loss={step_record['dpo_loss']:.4f} "
                    f"accuracy={step_record['accuracy']:.4f} margin={step_record['margin']:.4f}"
                )
                if step_eval_metrics:
                    summary += f" eval_pair_accuracy={float(step_eval_metrics.get('eval_pair_accuracy', 0.0)):.4f}"
                print(summary)
            last_step_metrics = step_record
        if max_steps is not None and step >= max_steps:
            break

    elapsed = time.time() - train_started_at
    return {
        "train_dpo_loss": float(last_step_metrics.get("dpo_loss", 0.0)),
        "train_accuracy": float(last_step_metrics.get("accuracy", 0.0)),
        "train_margin": float(last_step_metrics.get("margin", 0.0)),
        "train_chosen_reward": float(last_step_metrics.get("chosen_reward", 0.0)),
        "train_rejected_reward": float(last_step_metrics.get("rejected_reward", 0.0)),
        "train_steps": float(step),
        "train_duration_seconds": float(round(elapsed, 1)),
        "train_total_steps": float(total_steps),
    }


# ===== Artifact writing =====


def optional_git_provenance(repo_subdir: Path) -> dict[str, Any]:
    """Best-effort ``git`` HEAD and dirty flag for ``run.json`` (lightweight provenance).

    Mirrors the *intent* of tinker-cookbook ``code_state`` / run metadata without storing a full diff.
    """
    out: dict[str, Any] = {}
    try:
        head = subprocess.run(
            ["git", "-C", str(repo_subdir), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if head.returncode == 0 and head.stdout.strip():
            out["git_commit"] = head.stdout.strip()
        st = subprocess.run(
            ["git", "-C", str(repo_subdir), "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if st.returncode == 0:
            out["git_worktree_dirty"] = bool(st.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    return out


def collect_run_artifacts(
    log_path: Path,
    *,
    include_append_streams: bool,
) -> dict[str, dict[str, str]]:
    """Build the current run.json artifact index from files on disk."""
    artifacts: dict[str, dict[str, str]] = {}
    eval_artifacts: dict[str, str] = {}
    train_artifacts: dict[str, str] = {}
    eval_dir = log_path / "eval"
    train_dir = log_path / "train"

    for artifact_name, path in (
        ("examples", eval_dir / "examples.jsonl"),
        ("predictions", eval_dir / "predictions.jsonl"),
        ("metrics", eval_dir / "metrics.json"),
    ):
        if path.is_file():
            eval_artifacts[artifact_name] = str(path)

    if include_append_streams:
        for artifact_name, path, target in (
            ("periodic", eval_dir / "periodic.jsonl", eval_artifacts),
            ("metrics", train_dir / "metrics.jsonl", train_artifacts),
            ("checkpoints", train_dir / "checkpoints.jsonl", train_artifacts),
            ("batches", train_dir / "batches.jsonl", train_artifacts),
        ):
            if path.is_file() and path.stat().st_size > 0:
                target[artifact_name] = str(path)

    if eval_artifacts:
        artifacts["eval"] = eval_artifacts
    if train_artifacts:
        artifacts["train"] = train_artifacts
    return artifacts


def write_run_metadata(
    log_path: Path,
    args: argparse.Namespace,
    *,
    status: str,
    started_at: float,
    ended_at: float | None = None,
    error: str | None = None,
    include_append_streams: bool = True,
) -> None:
    """Two-phase run.json: write status:'running' at start, update to 'completed'/'failed' at end."""
    run_json_path = log_path / "run.json"
    if run_json_path.exists():
        payload = json.loads(run_json_path.read_text(encoding="utf-8"))
    else:
        payload = {}
    payload.update({
        "experiment": EXPERIMENT_DIR.name,
        "status": status,
        "argv": list(sys.argv),
        "command": shlex.join(sys.argv),
        "started_at_unix": started_at,
        "ended_at_unix": ended_at,
        "duration_seconds": round(ended_at - started_at, 1) if ended_at is not None else None,
        "args": vars(args),
        "env": {
            k: os.environ.get(k)
            for k in ("MINT_BASE_URL",)
            if os.environ.get(k)
        },
        "error": error,
    })
    artifacts = collect_run_artifacts(
        log_path,
        include_append_streams=include_append_streams,
    )
    if artifacts:
        payload["artifacts"] = artifacts
    else:
        payload.pop("artifacts", None)
    payload.update(optional_git_provenance(EXPERIMENT_DIR))
    write_json(run_json_path, payload)


def write_outputs(
    log_path: Path,
    *,
    args: argparse.Namespace,
    eval_examples: list[dict[str, Any]] | None = None,
    predictions: list[dict[str, Any]] | None = None,
    metrics: dict[str, Any] | None = None,
    extra_payload: dict[str, Any] | None = None,
    include_existing_artifacts: bool = True,
) -> None:
    log_path.mkdir(parents=True, exist_ok=True)
    run_json_path = log_path / "run.json"
    if run_json_path.exists():
        run_payload = json.loads(run_json_path.read_text(encoding="utf-8"))
    else:
        run_payload = {
            "experiment": EXPERIMENT_DIR.name,
            "argv": list(sys.argv),
            "command": shlex.join(sys.argv),
            "args": vars(args),
        }
    eval_dir = log_path / "eval"
    if eval_examples is not None:
        examples_path = eval_dir / "examples.jsonl"
        write_jsonl(examples_path, eval_examples)
    if predictions is not None:
        pred_path = eval_dir / "predictions.jsonl"
        write_jsonl(pred_path, prediction_artifact_rows(predictions))
    if metrics is not None:
        metrics_path = eval_dir / "metrics.json"
        write_json(metrics_path, metrics)
    artifacts = collect_run_artifacts(
        log_path,
        include_append_streams=include_existing_artifacts,
    )
    if artifacts:
        run_payload["artifacts"] = artifacts
    else:
        run_payload.pop("artifacts", None)
    if extra_payload:
        run_payload.update(extra_payload)
    run_payload.update(optional_git_provenance(EXPERIMENT_DIR))
    write_json(run_json_path, run_payload)


# ===== Entry point =====


async def main_async() -> int:
    args = parse_args()
    random.seed(args.seed)
    log_path = Path(args.log_path)
    load_checkpoint_path = str(getattr(args, "load_checkpoint_path", "") or "").strip()
    if args.eval_only and load_checkpoint_path:
        raise RuntimeError(
            "--load-checkpoint-path is training-only; "
            "use --base-model with a recorded sampler_path for --eval-only"
        )
    if args.dry_run and load_checkpoint_path:
        raise RuntimeError("--load-checkpoint-path is ignored under --dry-run")
    eval_data_arg = require_eval_data_arg(args.eval_data, dry_run=args.dry_run)
    train_path = Path(args.train_data)
    eval_path = Path(eval_data_arg)

    eval_rows = normalize_eval_rows(eval_path)
    if args.eval_limit > 0:
        eval_rows = eval_rows[: args.eval_limit]

    train_rows: list[dict[str, Any]] | None = None
    if train_path.exists():
        train_rows = normalize_train_rows(train_path)
    elif not args.eval_only and not args.dry_run:
        raise RuntimeError(f"train data not found at {train_path}")

    train_ids = (
        [str(row["prompt_fingerprint"]) for row in train_rows]
        if train_rows is not None
        else None
    )
    eval_ids = [str(row["prompt_fingerprint"]) for row in eval_rows]
    overlap = audit_overlap(train_ids, eval_ids)
    if overlap["overlap_count"] > 0 and not args.dry_run:
        raise RuntimeError(f"Train/eval overlap detected: {overlap['overlap_preview']}")

    if args.dry_run:
        return run_dry_run(args, train_rows=train_rows, eval_rows=eval_rows, overlap=overlap)

    started_at = time.time()
    run_dir = prepare_run_dir(log_path)
    # Automatic same-run resume: scan the current --log-path/train/checkpoints.jsonl
    # for the latest row that carries a resumable state_path. Compute this BEFORE
    # opening console.log so the console can be opened in append mode when a
    # resumable checkpoint is present, and before creating the training client so
    # the restored state_path can be passed in.
    resume_checkpoint = get_last_resumable_checkpoint(run_dir) if not args.eval_only else None
    validate_resume_contract(run_dir, args, resume_checkpoint=resume_checkpoint)
    resume_state_path = (
        str(resume_checkpoint.get("state_path") or "").strip()
        if resume_checkpoint is not None
        else ""
    )
    metadata_includes_append_streams = not (args.eval_only or args.dry_run)
    if resume_checkpoint is None:
        reset_eval_output_artifacts(run_dir)
    if not args.eval_only:
        reset_dpo_append_streams(
            run_dir,
            resume_checkpoint=resume_checkpoint,
            include_periodic_eval=int(args.eval_every) > 0,
            include_checkpoints=bool(
                resume_checkpoint is not None or int(args.save_every) > 0
            ),
            include_train_metrics=int(args.train_metrics_every) > 0,
            include_batch_trace=int(args.train_metrics_every) > 0,
        )
    console_log_mode = "a" if resume_checkpoint is not None else "w"
    console_log_handle = (run_dir / "console.log").open(
        console_log_mode,
        encoding="utf-8",
        buffering=1,
    )
    original_stdout, original_stderr = sys.stdout, sys.stderr
    sys.stdout = TeeStream(original_stdout, console_log_handle)
    sys.stderr = TeeStream(original_stderr, console_log_handle)
    write_run_metadata(
        run_dir,
        args,
        status="running",
        started_at=started_at,
        include_append_streams=metadata_includes_append_streams,
    )

    try:
        api_key = (os.environ.get("MINT_API_KEY") or "").strip()
        if not api_key:
            raise RuntimeError(f"Missing MINT_API_KEY in {EXPERIMENT_DIR / '.env'} or environment")
        service_client = create_service_client(timeout=args.mint_timeout)

        if args.eval_only:
            sampler = await create_sampling_client(service_client, args.base_model)
            tokenizer = get_tokenizer(sampler, args.base_model)
            eval_examples, predictions, metrics = await run_eval(sampler, tokenizer, eval_rows, args)
            write_outputs(
                run_dir,
                args=args,
                eval_examples=eval_examples,
                predictions=predictions,
                metrics=metrics,
                include_existing_artifacts=False,
                extra_payload={
                    "eval_data": str(eval_path),
                    "train_data": str(train_path) if train_path else None,
                    "overlap": overlap,
                },
            )
            emit_metric_lines(metrics)
            write_run_metadata(
                run_dir,
                args,
                status="completed",
                started_at=started_at,
                ended_at=time.time(),
                include_append_streams=metadata_includes_append_streams,
            )
            return 0

        if not train_rows:
            raise RuntimeError("Training requires non-empty train data")

        training_client = await create_training_client(
            service_client,
            args,
            resume_state_path=resume_state_path or None,
        )
        tokenizer = get_tokenizer(training_client, args.base_model)
        reference_model = args.reference_model.strip() or args.base_model
        reference_client = await create_sampling_client(service_client, reference_model)
        reference_tokenizer = tokenizer if reference_model == args.base_model else get_tokenizer(reference_client, reference_model)
        train_metrics = await run_train(
            training_client,
            reference_client,
            tokenizer,
            reference_tokenizer,
            train_rows,
            eval_rows,
            args,
            run_dir,
            resume_checkpoint=resume_checkpoint,
        )
        state_name = build_state_save_name(args.base_model, run_dir)
        final_state_name, final_sampler_name = checkpoint_save_names(state_name)
        final_state_path = await save_training_state(training_client, final_state_name)
        final_sampler_path = await save_sampler_checkpoint(training_client, final_sampler_name)
        final_step = int(train_metrics.get("train_steps", 0))
        final_n_batches_full, final_n_batches_remainder = divmod(len(train_rows), int(args.batch_size))
        final_n_batches = final_n_batches_full + (1 if bool(args.allow_partial_batch) and final_n_batches_remainder else 0)
        final_epoch_idx, final_batch_idx = step_to_loop_position(
            final_step,
            n_batches=final_n_batches,
            num_epochs=int(args.num_epochs),
        )
        checkpoint_row = {
            "name": state_name,
            "step": final_step,
            "epoch": final_epoch_idx,
            "batch": final_batch_idx,
            "final": True,
        }
        if final_state_path:
            checkpoint_row["state_path"] = final_state_path
        if final_sampler_path:
            checkpoint_row["sampler_path"] = final_sampler_path
        if len(checkpoint_row) > 5:
            append_jsonl(run_dir / "train" / "checkpoints.jsonl", checkpoint_row)
        if final_sampler_path:
            sampler = await create_sampling_client(service_client, final_sampler_path)
        else:
            sampler = await save_weights_for_sampling(training_client)
        eval_examples, predictions, eval_metrics = await run_eval(sampler, tokenizer, eval_rows, args)
        write_outputs(
            run_dir,
            args=args,
            eval_examples=eval_examples,
            predictions=predictions,
            metrics=eval_metrics,
            extra_payload={
                "overlap": overlap,
                "train_data": str(train_path),
                "eval_data": str(eval_path),
                "train_metrics": train_metrics,
                "reference_model": reference_model,
                "state_path": final_state_path,
                "sampler_path": final_sampler_path,
            },
        )
        emit_metric_lines({**train_metrics, **eval_metrics})
        write_run_metadata(
            run_dir,
            args,
            status="completed",
            started_at=started_at,
            ended_at=time.time(),
            include_append_streams=metadata_includes_append_streams,
        )
        return 0
    except Exception as exc:
        write_run_metadata(
            run_dir,
            args,
            status="failed",
            started_at=started_at,
            ended_at=time.time(),
            error=str(exc),
            include_append_streams=metadata_includes_append_streams,
        )
        raise
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        console_log_handle.close()


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    raise SystemExit(main())
