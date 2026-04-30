#!/usr/bin/env python3
"""Single-file experiment scaffold for the MinT cookbook."""

from __future__ import annotations

import argparse
import asyncio
import csv
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
DEFAULT_LOG_PATH = EXPERIMENT_DIR / "artifacts" / "latest"
DEFAULT_HF_HOME = "~/.cache/huggingface"
LOCAL_ENV_KEYS = {"MINT_API_KEY", "MINT_BASE_URL"}
SFT_TEXT_PREVIEW_CHARS = 240
DEFAULT_BATCH_PROMPT_RECORD_LIMIT = 5
PERIODIC_EVAL_SAMPLE_COUNT = 5


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

# The canonical scaffold uses mint runtime names by default. The current
# compatibility pin still installs through the tinker package, but new
# scaffold-derived experiments should write mint-facing runtime code.

from transformers import AutoTokenizer

import mint
from mint import types


def create_service_client(*, timeout: float | None = None) -> Any:
    return mint.ServiceClient(
        base_url=os.environ.get("MINT_BASE_URL"),
        timeout=timeout,
    )


WHITESPACE_RE = re.compile(r"\s+")


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
        description="Run an eval-first cookbook experiment."
    )
    parser.add_argument("--train-data", default="", help="Path to training JSONL.")
    parser.add_argument(
        "--eval-data",
        default="",
        help=(
            "Comma-separated name:path pairs for eval, e.g. 'name:path'. "
            "Plain paths get auto-named. Required for --dry-run, --eval-only, "
            "and final eval during training. Standard split-layout experiments "
            "usually pass data/eval/smoke.jsonl for local validation and "
            "data/eval/full.jsonl for the frozen benchmark."
        ),
    )
    parser.add_argument("--log-path", default=str(DEFAULT_LOG_PATH))
    parser.add_argument(
        "--base-model",
        default="Qwen/Qwen3-4B-Instruct-2507",
        help="Model to evaluate or train.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--eval-limit", type=int, default=0, help="Cap eval rows; 0 means all."
    )
    parser.add_argument("--eval-temperature", type=float, default=0.0)
    parser.add_argument("--eval-max-tokens", type=int, default=4096)
    parser.add_argument("--eval-top-p", type=float, default=1.0)
    parser.add_argument("--max-concurrent-requests", type=int, default=8)
    parser.add_argument("--mint-timeout", type=float, default=600.0)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Cap total training steps; 0 = train full epochs.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr-schedule", choices=("linear", "cosine"), default="linear")
    parser.add_argument(
        "--eval-every",
        type=int,
        default=0,
        help="Eval every N completed training steps; 0 = only final eval.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=0,
        help="Save periodic checkpoints every N completed steps; 0 = only final save.",
    )
    parser.add_argument(
        "--train-metrics-every",
        type=int,
        default=0,
        help=(
            "Append train/metrics.jsonl every N steps; 0 disables the streamed "
            "train/metrics.jsonl + train/batches.jsonl path."
        ),
    )
    parser.add_argument(
        "--train-print-every",
        type=int,
        default=1,
        help="Print train-step progress every N steps; 0 disables per-step stdout.",
    )
    parser.add_argument(
        "--load-checkpoint-path",
        default="",
        help=(
            "Saved training state path for a fresh weight-only training start. "
            "Ignored when the current --log-path already contains a resumable "
            "state_path in train/checkpoints.jsonl (that row triggers automatic "
            "same-run resume instead)."
        ),
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
        candidates = sorted(
            (p for p in snapshots_dir.iterdir() if has_tokenizer_files(p)),
            reverse=True,
        )
        if candidates:
            return candidates[0]
    return None


def get_tokenizer(client: Any | None, model_name: str) -> Any:
    cache_dir = cached_tokenizer_dir(model_name)
    if cache_dir is not None:
        print(f"@@ tokenizer_cache model={model_name} path={cache_dir}")
        return AutoTokenizer.from_pretrained(
            str(cache_dir), fast=True, local_files_only=True
        )
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


def load_records(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return load_jsonl(path), None
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as f:
            return [dict(row) for row in csv.DictReader(f)], None
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return [dict(item) for item in payload], None
        if isinstance(payload, dict):
            for key in ("items", "records", "examples", "data", "rows"):
                value = payload.get(key)
                if isinstance(value, list):
                    meta = {k: v for k, v in payload.items() if k != key}
                    return [dict(item) for item in value], meta
        raise TypeError(f"Unsupported JSON shape in {path}")
    raise RuntimeError(
        f"Unsupported format for {path}; expected .csv, .json, or .jsonl"
    )


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
    value = pick_first(row, "id", "example_id", "task_id", "episode_id")
    if value is None:
        return fallback
    return str(value).strip()


def normalize_text(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text).strip().lower()


def audit_overlap(
    train_ids: list[str] | None,
    eval_ids: list[str],
) -> dict[str, Any]:
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


def extract_completion_tokens(sample_result: Any) -> list[int]:
    sequences = getattr(sample_result, "sequences", None)
    if not isinstance(sequences, list) or not sequences:
        raise RuntimeError("Sampling result has no sequences")
    seq = sequences[0]
    tokens = getattr(seq, "tokens", None)
    if tokens is None and isinstance(seq, dict):
        tokens = seq.get("tokens")
    if hasattr(tokens, "tolist"):
        tokens = tokens.tolist()
    if not isinstance(tokens, list):
        raise RuntimeError(f"Unexpected token type: {type(tokens).__name__}")
    return [int(t) for t in tokens]


async def sample_assistant_text(
    sampler: Any,
    tokenizer: Any,
    messages: list[dict[str, str]],
    args: argparse.Namespace,
) -> str:
    prompt_tokens = build_generation_prompt_tokens(tokenizer, messages)
    sampling_params = types.SamplingParams(
        max_tokens=max(int(args.eval_max_tokens), 1),
        temperature=float(args.eval_temperature),
        top_p=float(args.eval_top_p),
    )
    kwargs = {
        "prompt": types.ModelInput.from_ints(tokens=prompt_tokens),
        "num_samples": 1,
        "sampling_params": sampling_params,
    }
    sample_fn = getattr(sampler, "sample_async", None)
    if callable(sample_fn):
        result = await resolve_api_result_async(sample_fn(**kwargs))
    else:
        sample_fn = getattr(sampler, "sample", None)
        if not callable(sample_fn):
            raise RuntimeError("Sampler must expose sample or sample_async")
        result = await asyncio.to_thread(
            lambda: resolve_api_result(sample_fn(**kwargs))
        )
    return tokenizer.decode(extract_completion_tokens(result)).strip()


# ===== SFT / training helpers (extend or replace per scaffold profile) =====
# The canonical template now carries a generic SFT baseline distilled from the
# shared control flow in fingpt / lawbench. Concrete experiments should replace
# or specialize the task adapters and any benchmark-specific periodic-eval or
# batch-trace details, but keep the shared helper family aligned.


def build_supervised_datum(
    tokenizer: Any,
    prompt_tokens: list[int],
    assistant_text: str,
) -> Any:
    completion_tokens = tokenizer.encode(assistant_text, add_special_tokens=False)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None:
        completion_tokens.append(int(eos_token_id))
    all_tokens = list(prompt_tokens) + completion_tokens
    if len(all_tokens) < 2:
        raise RuntimeError("Need at least two tokens to build supervised datum")
    weights = [0.0] * len(prompt_tokens) + [1.0] * len(completion_tokens)
    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens=all_tokens[:-1]),
        loss_fn_inputs={
            "target_tokens": all_tokens[1:],
            "weights": weights[1:],
        },
    )


def compute_lr_multiplier(schedule: str, step: int, total_steps: int) -> float:
    if total_steps <= 0:
        return 1.0
    progress = min(max(step / total_steps, 0.0), 1.0)
    if schedule == "linear":
        return 1.0 - progress
    if schedule == "cosine":
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    raise RuntimeError(f"Unknown lr_schedule: {schedule}")


def compute_total_train_steps(
    train_row_count: int,
    *,
    batch_size: int,
    num_epochs: int,
    max_steps: int | None,
) -> int:
    if batch_size <= 0:
        raise RuntimeError(f"--batch-size must be positive, got {batch_size}")
    n_batches = train_row_count // batch_size
    if n_batches == 0:
        raise RuntimeError(
            f"Not enough train rows ({train_row_count}) for batch_size={batch_size}"
        )
    total_steps = n_batches * num_epochs
    if max_steps is not None and max_steps > 0:
        total_steps = min(total_steps, max_steps)
    return total_steps


def compute_mean_nll(fwd_bwd_result: Any, datums: list[Any]) -> tuple[float, float]:
    """Return (mean_nll, total_weight) from a cross-entropy forward_backward result."""
    loss_fn_outputs = getattr(fwd_bwd_result, "loss_fn_outputs", None)
    if loss_fn_outputs is None and isinstance(fwd_bwd_result, dict):
        loss_fn_outputs = fwd_bwd_result.get("loss_fn_outputs")
    if not isinstance(loss_fn_outputs, list):
        raise RuntimeError("forward_backward result missing loss_fn_outputs")
    total_loss = 0.0
    total_weight = 0.0
    for idx, output in enumerate(loss_fn_outputs):
        logprobs = output["logprobs"]
        if hasattr(logprobs, "tolist"):
            logprobs = logprobs.tolist()
        weights = datums[idx].loss_fn_inputs["weights"]
        if hasattr(weights, "tolist"):
            weights = weights.tolist()
        for logprob, weight in zip(logprobs, weights):
            total_loss += -float(logprob) * float(weight)
            total_weight += float(weight)
    if total_weight == 0:
        return float("nan"), 0.0
    return total_loss / total_weight, total_weight


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


def reset_supervised_append_streams(
    log_path: Path,
    *,
    resume_checkpoint: dict[str, Any] | None = None,
    include_periodic_eval: bool = True,
    include_checkpoints: bool = True,
    include_train_metrics: bool = True,
    include_batch_trace: bool = True,
) -> None:
    """Reset enabled SFT append streams on fresh runs, preserve them on resume.

    Fresh runs truncate/create the enabled JSONL streams and remove any
    disabled stream files left behind by an older run that reused the same
    ``log_path``. This keeps the file set aligned with the active cadence
    flags without leaving stale artifacts in ``run.json``.
    """
    if resume_checkpoint is not None:
        return
    log_path.mkdir(parents=True, exist_ok=True)
    for path, enabled in (
        (log_path / "eval" / "periodic.jsonl", include_periodic_eval),
        (log_path / "train" / "checkpoints.jsonl", include_checkpoints),
        (log_path / "train" / "metrics.jsonl", include_train_metrics),
        (log_path / "train" / "batches.jsonl", include_batch_trace),
    ):
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
    return f"{EXPERIMENT_DIR.name}-{output_slug}-{model_slug}-sft"


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


def shuffled_train_rows_for_epoch(
    train_rows: list[dict[str, Any]],
    *,
    seed: int,
    epoch_idx: int,
) -> list[dict[str, Any]]:
    shuffled = list(train_rows)
    random.Random(f"{seed}:{epoch_idx}").shuffle(shuffled)
    return shuffled


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

    current_run_defining_args = {
        "base_model": args.base_model,
        "train_data": args.train_data,
        "eval_data": args.eval_data,
        "seed": args.seed,
        "rank": args.rank,
        "learning_rate": args.learning_rate,
        "num_epochs": args.num_epochs,
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "lr_schedule": args.lr_schedule,
        "eval_every": args.eval_every,
        "save_every": args.save_every,
    }
    mismatches: list[str] = []
    for key, current_value in current_run_defining_args.items():
        if key not in prior_args:
            continue
        if prior_args[key] != current_value:
            mismatches.append(f"{key}: expected {prior_args[key]!r}, got {current_value!r}")
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


def preview_text(text: str, limit: int = SFT_TEXT_PREVIEW_CHARS) -> str:
    collapsed = WHITESPACE_RE.sub(" ", str(text)).strip()
    if len(collapsed) <= limit:
        return collapsed
    if limit <= 1:
        return collapsed[:limit]
    return collapsed[: limit - 1] + "..."


def count_text_tokens(tokenizer: Any, text: str, *, add_special_tokens: bool) -> int:
    tokens = tokenizer.encode(text, add_special_tokens=add_special_tokens)
    return len(tokens)


def build_supervised_batch_trace_record(
    *,
    step: int,
    total_steps: int,
    epoch_idx: int,
    batch_idx: int,
    batch_rows: list[dict[str, Any]],
    tokenizer: Any,
) -> dict[str, Any]:
    examples: list[dict[str, Any]] = []
    prompt_token_counts: list[int] = []
    assistant_text_token_counts: list[int] = []
    assistant_text_loss_token_counts: list[int] = []
    prompt_char_counts: list[int] = []
    assistant_text_char_counts: list[int] = []
    example_ids: list[str] = []

    for row_index, row in enumerate(batch_rows, start=1):
        messages = build_eval_messages(row)
        prompt_preview = "\n".join(
            f"{str(message.get('role') or '').strip()}: {str(message.get('content') or '').strip()}"
            for message in messages
        ).strip()
        prompt_tokens = build_generation_prompt_tokens(tokenizer, messages)
        assistant_text = pick_first(
            row,
            "assistant_text",
            "output",
            "answer",
            "target",
            "label",
            "expected",
            "completion",
            "response",
        )
        if assistant_text is None:
            raise KeyError(
                "No assistant target field found in train row keys: "
                f"{sorted(row.keys())}"
            )
        assistant_text = str(assistant_text)
        assistant_text_tokens = count_text_tokens(
            tokenizer,
            assistant_text,
            add_special_tokens=False,
        )
        example_id = extract_row_id(row, fallback=f"step-{step}-row-{row_index}")

        prompt_token_counts.append(len(prompt_tokens))
        assistant_text_token_counts.append(assistant_text_tokens)
        assistant_text_loss_token_counts.append(assistant_text_tokens + 1)
        prompt_char_counts.append(len(prompt_preview))
        assistant_text_char_counts.append(len(assistant_text))
        example_ids.append(example_id)

        if len(examples) < DEFAULT_BATCH_PROMPT_RECORD_LIMIT:
            examples.append(
                {
                    "example_id": example_id,
                    "prompt_chars": len(prompt_preview),
                    "assistant_text_chars": len(assistant_text),
                    "prompt_tokens": len(prompt_tokens),
                    "assistant_text_tokens": assistant_text_tokens,
                    "assistant_text_loss_tokens": assistant_text_tokens + 1,
                    "prompt_preview": preview_text(prompt_preview),
                    "assistant_text_preview": preview_text(assistant_text),
                }
            )

    def mean(values: list[int]) -> float:
        return sum(values) / len(values) if values else 0.0

    return {
        "step": step,
        "total_steps": total_steps,
        "epoch": epoch_idx,
        "batch_index": batch_idx + 1,
        "progress": round(step / total_steps, 4) if total_steps > 0 else 1.0,
        "num_sequences": len(batch_rows),
        "example_ids": example_ids,
        "example_preview_limit": DEFAULT_BATCH_PROMPT_RECORD_LIMIT,
        "logged_example_count": len(examples),
        "prompt_chars_mean": round(mean(prompt_char_counts), 1),
        "assistant_text_chars_mean": round(mean(assistant_text_char_counts), 1),
        "prompt_tokens_mean": round(mean(prompt_token_counts), 1),
        "assistant_text_tokens_mean": round(mean(assistant_text_token_counts), 1),
        "assistant_text_loss_tokens_mean": round(mean(assistant_text_loss_token_counts), 1),
        "examples": examples,
    }


def build_supervised_row_datum(tokenizer: Any, row: dict[str, Any]) -> Any:
    messages = build_eval_messages(row)
    prompt_tokens = build_generation_prompt_tokens(tokenizer, messages)
    assistant_text = pick_first(
        row,
        "assistant_text",
        "output",
        "answer",
        "target",
        "label",
        "expected",
        "completion",
        "response",
    )
    if assistant_text is None:
        raise KeyError(
            "No assistant target field found in train row keys: "
            f"{sorted(row.keys())}"
        )
    return build_supervised_datum(tokenizer, prompt_tokens, str(assistant_text))


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


def merge_eval_metrics_into_step_record(
    step_record: dict[str, Any],
    eval_metrics: dict[str, Any],
    *,
    prefix: str = "test/",
) -> None:
    """Merge held-out eval scalars into one train step row."""
    for name, value in eval_metrics.items():
        if isinstance(value, bool):
            continue
        try:
            step_record[f"{prefix}{name}"] = float(value)
        except (TypeError, ValueError):
            continue


def merge_optimizer_metrics_into_step_record(
    step_record: dict[str, Any],
    optim_metrics: dict[str, Any],
    *,
    prefix: str = "optim/",
) -> None:
    """Merge optimizer-side metrics without overwriting canonical loop fields."""
    for name, value in optim_metrics.items():
        target_name = name if name not in step_record else f"{prefix}{name}"
        if target_name in step_record:
            raise RuntimeError(
                f"Optimizer metric key collision after prefixing: {name!r} -> {target_name!r}"
            )
        step_record[target_name] = value


# ===== Task-specific helpers (domain-specific, non-adapter helpers) =====
# Place domain validation, rendering, grading sub-helpers, benchmark-specific
# utilities here. Keep them above the adapters that call them.


def parse_eval_data_arg(raw: str) -> list[tuple[str, Path]]:
    """Parse comma-separated eval specs: 'name:path' or plain 'path'."""
    if not raw.strip():
        return []
    sources: list[tuple[str, Path]] = []
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        if ":" in entry:
            name, path_str = entry.split(":", 1)
            name, path_str = name.strip(), path_str.strip()
        else:
            path_str = entry
            name = Path(path_str).stem
        path = Path(path_str)
        if not path.exists():
            raise RuntimeError(f"eval data not found at {path}")
        sources.append((name, path))
    return sources


def strip_internal_row_fields(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in row.items() if not str(key).startswith("_")
    }


def build_eval_uid(*, eval_name: str, dataset_name: str, example_id: str) -> str:
    return f"{eval_name}:{dataset_name}:{example_id}"


def eval_example_artifact_row(
    row: dict[str, Any],
    *,
    messages: list[dict[str, str]],
    example_id: str,
    eval_name: str = "final",
) -> dict[str, Any]:
    dataset_name = str(row.get("_dataset_name") or "eval")
    return {
        "eval_uid": build_eval_uid(
            eval_name=eval_name,
            dataset_name=dataset_name,
            example_id=example_id,
        ),
        "eval_name": eval_name,
        "dataset_name": dataset_name,
        "example_id": example_id,
        "source_eval_data": str(row.get("_source_eval_data") or ""),
        "source_index": int(row.get("_source_index") or 0),
        "messages": messages,
        "eval_row": strip_internal_row_fields(row),
    }


def prediction_artifact_rows(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            key: value
            for key, value in prediction.items()
            if not str(key).startswith("_")
        }
        for prediction in predictions
    ]


# ===== Task-specific adapters (customize these for your benchmark) =====

def normalize_eval_rows(path: Path) -> list[dict[str, Any]]:
    """Load and validate eval data. Replace for benchmark-specific formats."""
    rows, _ = load_records(path)
    if not rows:
        raise RuntimeError(f"Empty eval data: {path}")
    return rows


def normalize_train_rows(path: Path) -> list[dict[str, Any]]:
    """Load and validate train data. Replace for benchmark-specific formats."""
    rows, _ = load_records(path)
    return rows


def build_eval_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    """Build chat messages from an eval row. Replace for benchmark-specific prompts."""
    prompt = pick_first(row, "prompt", "question", "input", "instruction")
    if not prompt:
        raise KeyError(f"No prompt field found in row keys: {sorted(row.keys())}")
    messages: list[dict[str, str]] = []
    system = pick_first(row, "system_prompt", "system")
    if system:
        messages.append({"role": "system", "content": str(system)})
    messages.append({"role": "user", "content": str(prompt)})
    return messages


def grade_assistant_text(assistant_text: str, row: dict[str, Any]) -> tuple[bool, str]:
    """Grade sampled assistant text against the expected answer. Replace for benchmark-specific grading."""
    expected = pick_first(row, "expected", "answer", "target", "label")
    if expected is None:
        return False, assistant_text.strip()
    prediction = assistant_text.strip()
    return normalize_text(prediction) == normalize_text(str(expected)), prediction


def compute_eval_metrics(predictions: list[dict[str, Any]]) -> dict[str, float]:
    """Aggregate predictions into metrics. Replace for benchmark-specific scoring."""
    total = len(predictions)
    correct = sum(1 for p in predictions if p.get("correct"))
    return {
        "eval_accuracy": (correct / total) if total else 0.0,
        "eval_correct": float(correct),
        "eval_total": float(total),
    }


# ===== Runtime entrypoints =====

def run_dry_run(
    args: argparse.Namespace,
    *,
    eval_rows: list[dict[str, Any]],
    overlap: dict[str, Any],
) -> int:
    preview = build_eval_messages(eval_rows[0])
    print(f"dry_run: eval_rows={len(eval_rows)} train_status={overlap['train_status']}")
    print(f"dry_run: overlap_count={overlap['overlap_count']}")
    print("dry_run: prompt_preview_start")
    print(json.dumps(preview, indent=2, ensure_ascii=False))
    print("dry_run: prompt_preview_end")
    return 0


async def run_eval(
    sampler: Any,
    tokenizer: Any,
    eval_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, float]]:
    """Evaluate all rows and return (eval_examples, predictions, metrics)."""
    semaphore = asyncio.Semaphore(max(int(args.max_concurrent_requests), 1))

    async def eval_one(
        index: int,
        row: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        async with semaphore:
            messages = build_eval_messages(row)
            example_id = extract_row_id(row, fallback=f"example-{index}")
            eval_example = (
                eval_example_artifact_row(
                    row,
                    messages=messages,
                    example_id=example_id,
                )
            )
            assistant_text = await sample_assistant_text(
                sampler, tokenizer, messages, args
            )
            correct, prediction = grade_assistant_text(assistant_text, row)
            expected = pick_first(row, "expected", "answer", "target", "label")
            return eval_example, {
                "eval_uid": build_eval_uid(
                    eval_name="final",
                    dataset_name=str(row.get("_dataset_name") or "eval"),
                    example_id=example_id,
                ),
                "eval_name": "final",
                "dataset_name": str(row.get("_dataset_name") or "eval"),
                "id": example_id,
                "example_id": example_id,
                "prediction": prediction,
                "correct": correct,
                "expected": expected,
                "assistant_text": assistant_text,
            }

    eval_pairs = list(
        await asyncio.gather(
            *(eval_one(i, r) for i, r in enumerate(eval_rows, start=1))
        )
    )
    eval_examples = [example for example, _ in eval_pairs]
    predictions = [prediction for _, prediction in eval_pairs]
    metrics = compute_eval_metrics(predictions)
    return eval_examples, predictions, metrics


async def run_train(
    training_client: Any,
    tokenizer: Any,
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, float]:
    num_epochs = int(args.num_epochs)
    max_steps = int(args.max_steps) if int(args.max_steps) > 0 else None
    batch_size = int(args.batch_size)
    if batch_size <= 0:
        raise RuntimeError(f"--batch-size must be positive, got {batch_size}")
    base_lr = float(args.learning_rate)
    rank = int(args.rank)
    lr_schedule = str(args.lr_schedule)
    eval_every = int(args.eval_every)
    save_every = int(args.save_every)
    train_metrics_every = int(args.train_metrics_every)
    train_print_every = int(args.train_print_every)

    total_steps = compute_total_train_steps(
        len(train_rows),
        batch_size=batch_size,
        num_epochs=num_epochs,
        max_steps=max_steps,
    )
    n_batches = len(train_rows) // batch_size
    train_metrics_enabled = train_metrics_every > 0
    resume_checkpoint = get_last_resumable_checkpoint(output_dir)
    start_step, start_epoch_idx, start_batch_idx = resolve_resume_state(
        output_dir,
        resume_checkpoint,
        total_steps=total_steps,
        n_batches=n_batches,
        num_epochs=num_epochs,
    )
    state_name_prefix = build_state_save_name(args.base_model, output_dir)

    train_metrics_path = output_dir / "train" / "metrics.jsonl"
    checkpoints_path = output_dir / "train" / "checkpoints.jsonl"
    periodic_evals_path = output_dir / "eval" / "periodic.jsonl"
    train_batches_path = output_dir / "train" / "batches.jsonl"

    print(
        f"Training: rows={len(train_rows)} batches={n_batches} epochs={num_epochs} "
        f"steps={total_steps} batch_size={batch_size} lr={base_lr} "
        f"rank={rank} schedule={lr_schedule} "
        f"periodic_eval_rows={len(eval_rows)} "
        f"train_metrics_every={train_metrics_every} "
        f"train_print_every={train_print_every} "
        f"batch_prompt_rows_per_step={DEFAULT_BATCH_PROMPT_RECORD_LIMIT}"
    )
    if start_step > 0:
        print(
            "Resume bookkeeping: "
            f"start_step={start_step} "
            f"start_epoch={start_epoch_idx} "
            f"start_batch={start_batch_idx} "
            f"total_steps={total_steps}"
        )

    final_loss = float("inf")
    step = start_step
    if step >= total_steps:
        return {
            "train_duration_seconds": 0.0,
            "train_steps": float(step),
            "train_total_steps": float(total_steps),
        }

    train_started_at = time.time()
    for epoch_idx in range(start_epoch_idx, num_epochs):
        shuffled = shuffled_train_rows_for_epoch(
            train_rows,
            seed=int(args.seed),
            epoch_idx=epoch_idx,
        )
        batch_start_idx = start_batch_idx if epoch_idx == start_epoch_idx else 0
        for batch_idx in range(batch_start_idx, n_batches):
            if max_steps is not None and step >= max_steps:
                break

            step_eval_metrics: dict[str, Any] | None = None
            step_started_at = time.time()
            learning_rate = base_lr * compute_lr_multiplier(
                lr_schedule,
                step,
                total_steps,
            )

            batch_rows = shuffled[
                batch_idx * batch_size : (batch_idx + 1) * batch_size
            ]
            datums = [build_supervised_row_datum(tokenizer, row) for row in batch_rows]

            num_tokens = 0
            for datum in datums:
                target_tokens = datum.loss_fn_inputs["target_tokens"]
                if hasattr(target_tokens, "tolist"):
                    target_tokens = target_tokens.tolist()
                try:
                    num_tokens += len(target_tokens) + 1
                except TypeError:
                    model_input = getattr(datum, "model_input", None)
                    model_input_length = getattr(model_input, "length", None)
                    if isinstance(model_input_length, int):
                        num_tokens += model_input_length + 1
                    else:
                        raise

            forward_backward_fn = getattr(training_client, "forward_backward_async", None)
            if callable(forward_backward_fn):
                fwd_bwd_result = await resolve_api_result_async(
                    forward_backward_fn(datums, loss_fn="cross_entropy")
                )
            else:
                forward_backward_fn = getattr(training_client, "forward_backward", None)
                if not callable(forward_backward_fn):
                    raise RuntimeError(
                        "Training client must expose forward_backward or forward_backward_async"
                    )
                fwd_bwd_result = resolve_api_result(
                    forward_backward_fn(datums, loss_fn="cross_entropy")
                )

            final_loss, total_weight = compute_mean_nll(fwd_bwd_result, datums)
            num_loss_tokens = int(total_weight)

            optim_step_fn = getattr(training_client, "optim_step_async", None)
            if callable(optim_step_fn):
                optim_step_result = await resolve_api_result_async(
                    optim_step_fn(types.AdamParams(learning_rate=learning_rate))
                )
            else:
                optim_step_fn = getattr(training_client, "optim_step", None)
                if not callable(optim_step_fn):
                    raise RuntimeError(
                        "Training client must expose optim_step or optim_step_async"
                    )
                optim_step_result = resolve_api_result(
                    optim_step_fn(types.AdamParams(learning_rate=learning_rate))
                )

            step += 1
            step_time = time.time() - step_started_at
            tokens_per_second = num_tokens / step_time if step_time > 0 else 0.0
            progress = step / total_steps if total_steps > 0 else 1.0

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
                periodic_state = await save_training_state(
                    training_client,
                    state_save_name,
                )
                if periodic_state:
                    checkpoint_row["state_path"] = periodic_state
                periodic_sampler = await save_sampler_checkpoint(
                    training_client,
                    sampler_save_name,
                )
                if periodic_sampler:
                    checkpoint_row["sampler_path"] = periodic_sampler
                if "state_path" in checkpoint_row or "sampler_path" in checkpoint_row:
                    append_jsonl(checkpoints_path, checkpoint_row)

            if eval_every > 0 and step % eval_every == 0:
                sampler = await save_weights_for_sampling(training_client)
                _, periodic_predictions, periodic_metrics = await run_eval(
                    sampler,
                    tokenizer,
                    eval_rows,
                    args,
                )
                step_eval_metrics = periodic_metrics
                append_jsonl(
                    periodic_evals_path,
                    {
                        "step": step,
                        **scalar_metric_items(periodic_metrics),
                        "samples": [
                            {
                                key: value
                                for key, value in prediction.items()
                                if key in (
                                    "id",
                                    "example_id",
                                    "prediction",
                                    "correct",
                                    "expected",
                                    "assistant_text",
                                )
                            }
                            for prediction in periodic_predictions[:PERIODIC_EVAL_SAMPLE_COUNT]
                        ],
                    },
                )
                emit_metric_lines(periodic_metrics)
                print(
                    f"  Periodic eval step={step}: "
                    f"eval_accuracy={float(periodic_metrics.get('eval_accuracy', 0.0)):.4f}"
                )

            step_record: dict[str, Any] = {
                "step": step,
                "epoch": epoch_idx,
                "train_mean_nll": final_loss,
                "learning_rate": learning_rate,
                "num_sequences": len(datums),
                "num_tokens": num_tokens,
                "num_loss_tokens": num_loss_tokens,
                "step_time_seconds": round(step_time, 3),
                "tokens_per_second": round(tokens_per_second, 1),
                "progress": round(progress, 4),
                "lora_rank": rank,
            }
            optim_metrics = getattr(optim_step_result, "metrics", None)
            if isinstance(optim_metrics, dict):
                merge_optimizer_metrics_into_step_record(step_record, optim_metrics)
            if step_eval_metrics is not None:
                merge_eval_metrics_into_step_record(step_record, step_eval_metrics)
            if train_metrics_enabled and should_record_train_metrics_row(
                step,
                total_steps,
                train_metrics_every,
                has_merged_eval=step_eval_metrics is not None,
            ):
                append_jsonl(train_metrics_path, step_record)
            if train_metrics_every > 0:
                append_jsonl(
                    train_batches_path,
                    build_supervised_batch_trace_record(
                        step=step,
                        total_steps=total_steps,
                        epoch_idx=epoch_idx,
                        batch_idx=batch_idx,
                        batch_rows=batch_rows,
                        tokenizer=tokenizer,
                    ),
                )

            if should_print_train_step(step, total_steps, train_print_every):
                eval_suffix = ""
                if step_eval_metrics is not None:
                    eval_suffix = (
                        f" test/eval_accuracy="
                        f"{float(step_eval_metrics.get('eval_accuracy', 0.0)):.4f}"
                    )
                print(
                    f"  Step {step}/{total_steps} ({progress:.0%}): "
                    f"loss={final_loss:.4f} lr={learning_rate:.6f} "
                    f"tok/s={tokens_per_second:.0f} step_time={step_time:.1f}s{eval_suffix}"
                )
        if max_steps is not None and step >= max_steps:
            break

    train_duration = time.time() - train_started_at
    result: dict[str, float] = {
        "train_duration_seconds": round(train_duration, 1),
        "train_steps": float(step),
        "train_total_steps": float(total_steps),
    }
    if final_loss != float("inf"):
        result["train_mean_nll"] = final_loss
    return result


# ===== Artifact writing =====

# Baseline ($new-experiment): write_run_metadata (two-phase run.json), console.log via TeeStream,
# eval/examples.jsonl, eval/predictions.jsonl, eval/metrics.json, and stdout METRIC lines from
# emit_metric_lines. Research-grade ($new-experiment-plus): stream train/metrics.jsonl
# (append cadence optional), append-only eval/periodic.jsonl (with prediction samples) /
# train/checkpoints.jsonl (with run-scoped truncation on fresh starts), optional
# train/batches.jsonl prompt lineage.
#
# Where to document vs implement: scaffolds/README.md → Artifact And Logging Contract
# (templates vs profiles vs experiment vs skills). Do not grow this default helper until the repo promotes the pattern.


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
    """Write eval artifacts and merge into existing run.json."""
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
    log_path = Path(args.log_path)

    # ---- Data ----
    eval_data_arg = require_eval_data_arg(args.eval_data, dry_run=args.dry_run)
    eval_sources = parse_eval_data_arg(eval_data_arg)
    eval_rows: list[dict[str, Any]] = []
    for eval_name, eval_path in eval_sources:
        rows = normalize_eval_rows(eval_path)
        if args.eval_limit > 0:
            rows = rows[: args.eval_limit]
        for source_index, row in enumerate(rows, start=1):
            row_with_meta = dict(row)
            row_with_meta["_dataset_name"] = eval_name
            row_with_meta["_source_eval_data"] = str(eval_path)
            row_with_meta["_source_index"] = source_index
            eval_rows.append(row_with_meta)

    train_path: Path | None = None
    train_rows: list[dict[str, Any]] | None = None
    if args.train_data.strip():
        train_path = Path(args.train_data)
        if not train_path.exists():
            raise RuntimeError(f"train data not found at {train_path}")
        train_rows = normalize_train_rows(train_path)
    elif not args.eval_only and not args.dry_run:
        raise RuntimeError("--train-data is required for training")

    overlap = audit_overlap(
        [extract_row_id(r) for r in train_rows] if train_rows else None,
        [extract_row_id(r) for r in eval_rows],
    )
    if overlap["overlap_count"] > 0:
        raise RuntimeError(f"Train/eval overlap detected: {overlap['overlap_preview']}")

    # ---- Dry run (no credentials needed) ----
    if args.dry_run:
        return run_dry_run(args, eval_rows=eval_rows, overlap=overlap)

    # ---- Run directory + console.log ----
    started_at = time.time()
    run_dir = prepare_run_dir(log_path)
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
        reset_supervised_append_streams(
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
        # ---- Service client ----
        api_key = (os.environ.get("MINT_API_KEY") or "").strip()
        if not api_key:
            raise RuntimeError(
                f"Missing MINT_API_KEY in {EXPERIMENT_DIR / '.env'} or environment"
            )
        service_client = create_service_client(timeout=args.mint_timeout)

        # ---- Eval-only ----
        if args.eval_only:
            sampler = await create_sampling_client(service_client, args.base_model)
            tokenizer = get_tokenizer(sampler, args.base_model)
            eval_examples, predictions, metrics = await run_eval(
                sampler, tokenizer, eval_rows, args
            )
            write_outputs(
                run_dir,
                args=args,
                eval_examples=eval_examples,
                predictions=predictions,
                metrics=metrics,
                include_existing_artifacts=False,
                extra_payload={"overlap": overlap},
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

        # ---- Training ----
        if not train_rows:
            raise RuntimeError("Train data is required for the SFT path.")

        training_client = await create_training_client(
            service_client,
            args,
            resume_state_path=resume_state_path or None,
        )
        tokenizer = get_tokenizer(training_client, args.base_model)

        train_metrics = await run_train(
            training_client,
            tokenizer,
            train_rows,
            eval_rows,
            args,
            run_dir,
        )

        state_name = build_state_save_name(args.base_model, run_dir)
        final_state_name, final_sampler_name = checkpoint_save_names(state_name)
        state_path = await save_training_state(training_client, final_state_name)
        sampler_path = await save_sampler_checkpoint(training_client, final_sampler_name)
        uses_checkpoint_registry = bool(resume_checkpoint is not None or int(args.save_every) > 0)
        if uses_checkpoint_registry and (state_path or sampler_path):
            final_n_batches = len(train_rows) // int(args.batch_size)
            final_step = int(train_metrics.get("train_steps", 0))
            if final_n_batches > 0:
                final_epoch_idx, final_batch_idx = step_to_loop_position(
                    final_step,
                    n_batches=final_n_batches,
                    num_epochs=int(args.num_epochs),
                )
            else:
                final_epoch_idx, final_batch_idx = 0, 0
            checkpoint_row: dict[str, Any] = {
                "name": state_name,
                "step": final_step,
                "epoch": final_epoch_idx,
                "batch": final_batch_idx,
                "final": True,
            }
            if state_path:
                checkpoint_row["state_path"] = state_path
            if sampler_path:
                checkpoint_row["sampler_path"] = sampler_path
            append_jsonl(run_dir / "train" / "checkpoints.jsonl", checkpoint_row)

        if sampler_path:
            sampler = await create_sampling_client(service_client, sampler_path)
        else:
            sampler = await save_weights_for_sampling(training_client)
        eval_examples, predictions, eval_metrics = await run_eval(
            sampler,
            tokenizer,
            eval_rows,
            args,
        )
        write_outputs(
            run_dir,
            args=args,
            eval_examples=eval_examples,
            predictions=predictions,
            metrics=eval_metrics,
            extra_payload={
                "overlap": overlap,
                "train_metrics": train_metrics,
                "state_path": state_path,
                "sampler_path": sampler_path,
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
    except Exception as e:
        write_run_metadata(
            run_dir,
            args,
            status="failed",
            started_at=started_at,
            ended_at=time.time(),
            error=str(e),
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
