#!/usr/bin/env python3
"""dapo-aime24 - direct GRPO on DAPO-Math-17k with AIME 2024 evaluation."""

from __future__ import annotations

import argparse
import asyncio
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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

# ===== Paths and constants =====

EXPERIMENT_DIR = Path(__file__).resolve().parent
DATA_DIR = EXPERIMENT_DIR / "data"
DEFAULT_HF_HOME = "~/.cache/huggingface"
LOCAL_ENV_KEYS = {"MINT_API_KEY", "MINT_BASE_URL"}

STOP_TOKEN_RE = re.compile(r"<\|[^>]+\|>")
ANSWER_LINE_RE = re.compile(r"(?im)^\s*Answer\s*:\s*([^\n<]+)")
ANSWER_PARSE_TAIL_CHARS = 2048
VERL_DAPO_PROMPT_INTRO = (
    "Solve the following math problem step by step. The last line of your response should be of the form "
    "Answer: <answer>, where <answer> is the final answer to the problem."
)
VERL_DAPO_PROMPT_REMINDER = (
    'Remember to put only the final answer on its own line after "Answer:".'
)
VERL_NORMALIZE_SUBSTITUTIONS = [
    ("an ", ""),
    ("a ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]
VERL_NORMALIZE_REMOVED_EXPRESSIONS = [
    "square",
    "ways",
    "integers",
    "dollars",
    "mph",
    "inches",
    "hours",
    "km",
    "units",
    "\\ldots",
    "sue",
    "points",
    "feet",
    "minutes",
    "digits",
    "cents",
    "degrees",
    "cm",
    "gm",
    "pounds",
    "meters",
    "meals",
    "edges",
    "students",
    "childrentickets",
    "multiples",
    "\\text{s}",
    "\\text{.}",
    "\\text{\ns}",
    "\\text{}^2",
    "\\text{}^3",
    "\\text{\n}",
    "\\text{}",
    r"\mathrm{th}",
    r"^\circ",
    r"^{\circ}",
    r"\;",
    r",\!",
    "{,}",
    '"',
    "\\dots",
]

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


async def resolve_api_result_async(maybe_result: Any) -> Any:
    """Normalize the installed SDK's async return shapes into a final result."""
    if hasattr(maybe_result, "result_async") and callable(maybe_result.result_async):
        return await resolve_api_result_async(await maybe_result.result_async())
    if asyncio.isfuture(maybe_result):
        return await resolve_api_result_async(await maybe_result)
    if hasattr(maybe_result, "__await__"):
        return await resolve_api_result_async(await maybe_result)
    if hasattr(maybe_result, "result") and callable(maybe_result.result):
        return await resolve_api_result_async(
            await asyncio.to_thread(maybe_result.result)
        )
    return maybe_result


def resolve_api_result(maybe_result: Any) -> Any:
    """Normalize sync SDK return shapes into a final result."""
    if hasattr(maybe_result, "result") and callable(maybe_result.result):
        return resolve_api_result(maybe_result.result())
    return maybe_result


def require_async_method(
    obj: Any, attr_name: str, owner_name: str
) -> Callable[..., Any]:
    method = getattr(obj, attr_name, None)
    if callable(method):
        return method
    raise RuntimeError(f"{owner_name} must expose async method `{attr_name}`")


async def preflight_connection_async(service_client: Any) -> Any:
    get_capabilities_async = require_async_method(
        service_client,
        "get_server_capabilities_async",
        "MinT service client",
    )
    return await resolve_api_result_async(await get_capabilities_async())


async def create_lora_training_client_async(
    service_client: Any,
    *,
    base_model: str,
    rank: int,
) -> Any:
    create_client_async = require_async_method(
        service_client,
        "create_lora_training_client_async",
        "MinT service client",
    )
    return await resolve_api_result_async(
        await create_client_async(
            base_model=base_model,
            rank=rank,
            train_mlp=True,
            train_attn=True,
            train_unembed=True,
        )
    )


async def create_sampling_client(service_client: Any, base_model: str) -> Any:
    client_kwargs = (
        {"model_path": base_model}
        if is_sampler_model_path(base_model)
        else {"base_model": base_model}
    )
    create_fn = getattr(service_client, "create_sampling_client_async", None)
    if callable(create_fn):
        return await resolve_api_result_async(create_fn(**client_kwargs))
    create_fn = getattr(service_client, "create_sampling_client", None)
    if callable(create_fn):
        return resolve_api_result(create_fn(**client_kwargs))
    raise RuntimeError("Service client must expose create_sampling_client")


async def create_sampling_client_async_compat(
    service_client: Any,
    model_path: str | None = None,
    *,
    base_model: str | None = None,
) -> Any:
    create_sampling_client_async = require_async_method(
        service_client,
        "create_sampling_client_async",
        "MinT service client",
    )
    kwargs: dict[str, Any] = {}
    if model_path is not None:
        kwargs["model_path"] = model_path
    if base_model is not None:
        kwargs["base_model"] = base_model
    if not kwargs:
        raise ValueError("create_sampling_client requires model_path and/or base_model")
    return await resolve_api_result_async(await create_sampling_client_async(**kwargs))


def extract_trajectory_tokens_and_logprobs(
    sample_result: Any,
) -> tuple[list[list[int]], list[list[float]]]:
    if isinstance(sample_result, list):
        tokens_group: list[list[int]] = []
        logprobs_group: list[list[float]] = []
        for item in sample_result:
            item_tokens, item_logprobs = extract_trajectory_tokens_and_logprobs(item)
            tokens_group.extend(item_tokens)
            logprobs_group.extend(item_logprobs)
        return tokens_group, logprobs_group

    sequences = getattr(sample_result, "sequences", None)
    if sequences is not None:
        tokens_group = []
        logprobs_group = []
        for sequence in sequences:
            tokens_group.append(
                [int(token) for token in (getattr(sequence, "tokens", None) or [])]
            )
            logprobs_group.append(
                [
                    float(logprob)
                    for logprob in (getattr(sequence, "logprobs", None) or [])
                ]
            )
        return tokens_group, logprobs_group

    completion_tokens = getattr(sample_result, "completion_tokens", None)
    if completion_tokens is not None:
        tokens_group = [
            [int(token) for token in tokens] for tokens in (completion_tokens or [])
        ]
        raw_logprobs = getattr(sample_result, "logprobs", None) or []
        logprobs_group = [
            [float(logprob) for logprob in (logprobs or [])]
            for logprobs in raw_logprobs
        ]
        return tokens_group, logprobs_group

    raise TypeError(
        f"Unsupported sample response type: {type(sample_result).__name__}; expected `sequences` or `completion_tokens`"
    )


def copy_sampling_params_with_seed(sampling_params: Any, seed: int | None) -> Any:
    if seed is None:
        return sampling_params

    model_copy = getattr(sampling_params, "model_copy", None)
    if callable(model_copy):
        return model_copy(update={"seed": seed})

    copy_method = getattr(sampling_params, "copy", None)
    if callable(copy_method):
        try:
            return copy_method(update={"seed": seed})
        except TypeError:
            pass

    kwargs: dict[str, Any] = {
        "max_tokens": getattr(sampling_params, "max_tokens"),
        "temperature": getattr(sampling_params, "temperature", 0.0),
        "seed": seed,
    }
    top_p = getattr(sampling_params, "top_p", None)
    if top_p is not None:
        kwargs["top_p"] = top_p
    stop = getattr(sampling_params, "stop", None)
    if stop is not None:
        kwargs["stop"] = stop
    stop_token_ids = getattr(sampling_params, "stop_token_ids", None)
    if stop_token_ids is not None:
        kwargs["stop_token_ids"] = list(stop_token_ids)
    return types.SamplingParams(**kwargs)


async def sample_one_async(
    sampler: Any,
    prompt_tokens: list[int],
    num_samples: int,
    sampling_params: Any,
) -> Any:
    """Sample one prompt through the async MinT sampler API."""
    prompt = types.ModelInput.from_ints(tokens=prompt_tokens)
    sample_async = require_async_method(
        sampler,
        "sample_async",
        "MinT sampling client",
    )
    return await resolve_api_result_async(
        await sample_async(
            prompt=prompt,
            num_samples=num_samples,
            sampling_params=sampling_params,
        )
    )


async def sample_many_async(
    sampler: Any,
    sampling_requests: list[tuple[list[int], int, Any]],
    max_in_flight: int,
    semaphore: asyncio.Semaphore | None = None,
    on_result: Callable[[int, Any], None] | None = None,
) -> list[Any]:
    """Run a sliding in-flight sampling window while preserving request order."""
    if not sampling_requests:
        return []

    max_in_flight = max(max_in_flight, 1)
    results_by_index: list[Any | None] = [None] * len(sampling_requests)
    pending_tasks: dict[asyncio.Task[Any], int] = {}
    next_submit_index = 0

    async def run_request(
        prompt_tokens: list[int], num_samples: int, sampling_params: Any
    ) -> Any:
        if semaphore is None:
            return await sample_one_async(
                sampler=sampler,
                prompt_tokens=prompt_tokens,
                num_samples=num_samples,
                sampling_params=sampling_params,
            )
        async with semaphore:
            return await sample_one_async(
                sampler=sampler,
                prompt_tokens=prompt_tokens,
                num_samples=num_samples,
                sampling_params=sampling_params,
            )

    def submit_next_request() -> None:
        nonlocal next_submit_index
        if next_submit_index >= len(sampling_requests):
            return
        index = next_submit_index
        prompt_tokens, num_samples, sampling_params = sampling_requests[index]
        pending_tasks[
            asyncio.create_task(
                run_request(
                    prompt_tokens=prompt_tokens,
                    num_samples=num_samples,
                    sampling_params=sampling_params,
                ),
                name=f"sample_many_async_{index}",
            )
        ] = index
        next_submit_index += 1

    for _ in range(min(max_in_flight, len(sampling_requests))):
        submit_next_request()

    try:
        while pending_tasks:
            done_tasks, _ = await asyncio.wait(
                pending_tasks, return_when=asyncio.FIRST_COMPLETED
            )
            for task in done_tasks:
                index = pending_tasks.pop(task)
                result = task.result()
                results_by_index[index] = result
                if on_result is not None:
                    on_result(index, result)
                submit_next_request()
    finally:
        for task in pending_tasks:
            task.cancel()
        if pending_tasks:
            await asyncio.gather(*pending_tasks, return_exceptions=True)

    ordered_results: list[Any] = []
    for result in results_by_index:
        if result is None:
            raise RuntimeError("sample_many_async completed without a result")
        ordered_results.append(result)
    return ordered_results


async def sample_many_with_semaphore_async(
    sampler: Any,
    sampling_requests: list[tuple[list[int], int, Any]],
    max_concurrent_requests: int,
    on_result: Callable[[int, Any], None] | None = None,
) -> list[Any]:
    """Run cookbook-style eval concurrency while surfacing each completed result immediately."""
    if not sampling_requests:
        return []

    semaphore = asyncio.Semaphore(max(max_concurrent_requests, 1))
    results_by_index: list[Any | None] = [None] * len(sampling_requests)

    async def _one(
        index: int, prompt_tokens: list[int], num_samples: int, sampling_params: Any
    ) -> tuple[int, Any]:
        async with semaphore:
            result = await sample_one_async(
                sampler=sampler,
                prompt_tokens=prompt_tokens,
                num_samples=num_samples,
                sampling_params=sampling_params,
            )
        return index, result

    tasks = [
        asyncio.create_task(
            _one(index, *sampling_request),
            name=f"sample_many_with_semaphore_async_{index}",
        )
        for index, sampling_request in enumerate(sampling_requests)
    ]
    try:
        for completed_task in asyncio.as_completed(tasks):
            index, result = await completed_task
            results_by_index[index] = result
            if on_result is not None:
                on_result(index, result)
    finally:
        pending = [task for task in tasks if not task.done()]
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    ordered_results: list[Any] = []
    for result in results_by_index:
        if result is None:
            raise RuntimeError(
                "sample_many_with_semaphore_async completed without a result"
            )
        ordered_results.append(result)
    return ordered_results


def make_int_tensor_data(values: list[int]) -> types.TensorData:
    return types.TensorData(
        data=[int(value) for value in values], dtype="int64", shape=[len(values)]
    )


def make_float_tensor_data(values: list[float]) -> types.TensorData:
    return types.TensorData(
        data=[float(value) for value in values], dtype="float32", shape=[len(values)]
    )


# ===== CLI =====


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", default=str(DATA_DIR / "train.jsonl"))
    parser.add_argument("--eval-data", default=str(DATA_DIR / "eval.jsonl"))
    parser.add_argument(
        "--log-path", default=str(EXPERIMENT_DIR / "artifacts" / "latest")
    )
    parser.add_argument("--base-model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--mint-timeout", type=float, default=600.0)
    parser.add_argument("--client-create-timeout", type=float, default=180.0)
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
    parser.add_argument("--save-state-name", default="")
    parser.add_argument(
        "--save-every-steps", dest="save_state_every_steps", type=int, default=5
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--grpo-steps", type=int, default=180)
    parser.add_argument("--eval-every-steps", type=int, default=1)
    parser.add_argument("--groups-per-batch", type=int, default=32)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--stream-minibatches-per-step", type=int, default=4)
    parser.add_argument("--min-accepted-groups", type=int, default=0)
    parser.add_argument("--min-started-minibatches", type=int, default=0)
    parser.add_argument("--tail-grace-seconds", type=float, default=90.0)
    parser.add_argument("--rl-learning-rate", type=float, default=1e-4)
    parser.add_argument("--rl-temperature", type=float, default=0.7)
    parser.add_argument("--rl-max-tokens", type=int, default=8192)
    parser.add_argument("--apply-overlong-filtering", action="store_true")
    parser.add_argument("--overlong-buffer-len", type=int, default=4096)
    parser.add_argument("--overlong-buffer-penalty-factor", type=float, default=1.0)
    parser.add_argument(
        "--dynamic-sampling-type", choices=["none", "filter"], default="none"
    )
    parser.add_argument("--dynamic-sampling-max-rollout-waves", type=int, default=0)
    parser.add_argument("--grpo-max-train-prompts", type=int, default=0)
    parser.add_argument(
        "--rl-loss",
        choices=["importance_sampling", "ppo"],
        default="importance_sampling",
    )

    parser.add_argument("--eval-num-samples", type=int, default=16)
    parser.add_argument("--eval-temperature", type=float, default=1.0)
    parser.add_argument("--eval-top-p", type=float, default=0.7)
    parser.add_argument("--eval-max-tokens", type=int, default=8192)
    parser.add_argument("--max-concurrent-requests", type=int, default=32)
    parser.add_argument("--recreate-client-every-steps", type=int, default=0)
    return parser.parse_args()


# ===== Infrastructure helpers =====


class TeeStream:
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
        return any(
            getattr(stream, "isatty", lambda: False)() for stream in self.streams
        )


def remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
        return
    if path.exists():
        shutil.rmtree(path)


def prepare_run_dir(output_dir: Path) -> Path:
    """Create run directory; symlink artifacts/runs/latest for smoke runs only."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if "smoke" in output_dir.name:
        latest = output_dir.parent / "latest"
        if latest.is_symlink() or latest.is_file():
            latest.unlink()
        elif latest.exists():
            shutil.rmtree(latest)
        latest.parent.mkdir(parents=True, exist_ok=True)
        latest.symlink_to(output_dir.resolve(), target_is_directory=True)
    return output_dir


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


def iter_jsonl_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Invalid JSONL in {path}:{line_number}: {exc}"
                ) from exc
            if isinstance(record, dict):
                records.append(record)
    return records


def load_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() != ".jsonl":
        raise RuntimeError(f"Unsupported data format for {path}; expected .jsonl")
    return load_jsonl(path)


def get_tokenizer(client: Any | None, model_name: str) -> Any:
    if is_sampler_model_path(model_name) and client is not None:
        fn = getattr(client, "get_tokenizer", None)
        if callable(fn):
            return fn()
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


def preview_text(text: str | None, limit: int = 72) -> str:
    if text is None:
        return "<none>"
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= limit:
        return compact
    return compact[: max(limit - 3, 1)] + "..."


def format_elapsed(seconds: float) -> str:
    total_seconds = max(int(seconds), 0)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def preview_messages(messages: list[dict[str, str]]) -> str:
    return "\n\n".join(
        f"{message['role']}:\n{message['content']}" for message in messages
    )


# ===== Training helpers =====


@dataclass(frozen=True)
class ResumeLoopState:
    state_path: str
    completed_steps: int
    next_step: int
    source: str
    record_path: str | None = None


def get_last_resumable_checkpoint(log_path: Path) -> dict[str, Any] | None:
    checkpoints_path = log_path / "train" / "checkpoints.jsonl"
    checkpoint_rows = iter_jsonl_records(checkpoints_path)
    resumable_rows = [
        row for row in checkpoint_rows if str(row.get("state_path") or "").strip()
    ]
    if not resumable_rows:
        return None
    return resumable_rows[-1]


def validate_resume_contract(
    log_path: Path,
    args: argparse.Namespace,
    *,
    resume_checkpoint: dict[str, Any] | None,
) -> None:
    """Require current args to match ``run.json`` when resuming the same run directory.

    Same contract as ``experiments/lawbench`` / ``fingpt`` / ``chat-dpo``: automatic
    same-run resume keys off the current ``--log-path``; missing ``run.json`` is
    tolerated. Append-only streams past the last checkpoint do not block resume.
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

    mismatches: list[str] = []
    for key, current_value in {
        "base_model": args.base_model,
        "train_data": args.train_data,
        "eval_data": args.eval_data,
        "seed": args.seed,
        "rank": args.rank,
        "grpo_steps": args.grpo_steps,
        "groups_per_batch": args.groups_per_batch,
        "group_size": args.group_size,
        "rl_learning_rate": args.rl_learning_rate,
        "rl_loss": args.rl_loss,
        "eval_every_steps": args.eval_every_steps,
        "save_state_every_steps": args.save_state_every_steps,
        "rl_temperature": args.rl_temperature,
        "stream_minibatches_per_step": args.stream_minibatches_per_step,
        "dynamic_sampling_type": args.dynamic_sampling_type,
        "apply_overlong_filtering": bool(args.apply_overlong_filtering),
    }.items():
        if key not in prior_args:
            continue
        if prior_args[key] != current_value:
            mismatches.append(f"{key}: expected {prior_args[key]!r}, got {current_value!r}")
    if mismatches:
        raise RuntimeError(
            "Automatic same-run resume requires the same run-defining args recorded in "
            f"{run_json_path}: " + "; ".join(mismatches)
        )


def coerce_nonnegative_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, float) and value.is_integer():
        int_value = int(value)
        return int_value if int_value >= 0 else None
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def checkpoint_record_completed_steps(record: dict[str, Any]) -> int | None:
    for key in ("completed_steps", "step", "batch"):
        value = coerce_nonnegative_int(record.get(key))
        if value is not None:
            return value
    return None


def parse_step_from_checkpoint_path(state_path: str) -> int | None:
    checkpoint_name = state_path.rstrip("/").rsplit("/", 1)[-1]
    return int(checkpoint_name) if checkpoint_name.isdigit() else None


def candidate_checkpoint_record_paths(output_dir: Path) -> list[Path]:
    seen: set[Path] = set()
    candidates: list[Path] = []

    def add(path: Path) -> None:
        resolved = path.resolve(strict=False)
        if resolved in seen:
            return
        seen.add(resolved)
        candidates.append(path)

    add(output_dir / "train" / "checkpoints.jsonl")
    runs_dir = EXPERIMENT_DIR / "artifacts" / "runs"
    if runs_dir.exists():
        for run_dir in sorted(runs_dir.iterdir(), reverse=True):
            add(run_dir / "train" / "checkpoints.jsonl")
    return candidates


def resume_loop_state_from_checkpoint_row(
    record: dict[str, Any],
    run_dir: Path,
    args: argparse.Namespace,
) -> ResumeLoopState:
    state_path = str(record.get("state_path") or "").strip()
    if not state_path:
        raise RuntimeError("Resumable checkpoint row must carry a non-empty state_path")
    checkpoints_path = run_dir / "train" / "checkpoints.jsonl"
    record_relpath = os.path.relpath(checkpoints_path, Path.cwd())
    completed_steps = checkpoint_record_completed_steps(record)
    if completed_steps is None:
        parsed = parse_step_from_checkpoint_path(state_path)
        if parsed is None:
            raise RuntimeError(
                "Checkpoint row is missing completed_steps/step/batch and state_path has no numeric segment"
            )
        completed_steps = parsed
    cap = max(args.grpo_steps, 0)
    completed_steps = min(completed_steps, cap)
    next_from_row = coerce_nonnegative_int(record.get("next_step"))
    if next_from_row is not None:
        next_step = min(next_from_row, cap + 1)
    else:
        next_step = min(completed_steps + 1, cap + 1)
    return ResumeLoopState(
        state_path=state_path,
        completed_steps=completed_steps,
        next_step=next_step,
        source="checkpoints_jsonl",
        record_path=record_relpath,
    )


def resume_loop_state_from_explicit_state_path(
    state_path: str,
    output_dir: Path,
    args: argparse.Namespace,
) -> ResumeLoopState | None:
    """Resolve loop state when given an explicit training state path (tests, tooling)."""
    checkpoint = state_path.strip()
    if not checkpoint:
        return None

    for checkpoints_path in candidate_checkpoint_record_paths(output_dir):
        for record in reversed(iter_jsonl_records(checkpoints_path)):
            if record.get("state_path") != checkpoint:
                continue
            completed_steps = checkpoint_record_completed_steps(record)
            if completed_steps is None:
                continue
            completed_steps = min(completed_steps, max(args.grpo_steps, 0))
            return ResumeLoopState(
                state_path=checkpoint,
                completed_steps=completed_steps,
                next_step=min(completed_steps + 1, max(args.grpo_steps, 0) + 1),
                source="checkpoints_jsonl",
                record_path=os.path.relpath(checkpoints_path, Path.cwd()),
            )

    parsed_step = parse_step_from_checkpoint_path(checkpoint)
    if parsed_step is not None:
        completed_steps = min(parsed_step, max(args.grpo_steps, 0))
        return ResumeLoopState(
            state_path=checkpoint,
            completed_steps=completed_steps,
            next_step=min(completed_steps + 1, max(args.grpo_steps, 0) + 1),
            source="checkpoint_name",
        )

    raise RuntimeError(
        "Could not infer resume loop state from explicit state path. "
        "Use a checkpoint path recorded in train/checkpoints.jsonl or a periodic checkpoint whose name is the step number."
    )


@dataclass(frozen=True)
class PromptSamplingJob:
    step: int
    row_index: int
    row: dict[str, str]
    prompt_tokens: list[int]
    sampling_params: Any


@dataclass(frozen=True)
class PromptSamplingResult:
    step: int
    row_index: int
    sampler_step: int
    row: dict[str, str]
    prompt_tokens: list[int]
    result: Any
    dynamic_rollout_wave_index: int = 1
    dynamic_accepted_non_flat: bool = True


def make_prompt_group_batch(
    rows: list[dict[str, Any]],
    step: int,
    groups_per_batch: int,
) -> list[dict[str, Any]]:
    if not rows:
        return []
    actual_groups = min(max(groups_per_batch, 1), len(rows))
    start = ((step - 1) * actual_groups) % len(rows)
    return [rows[(start + offset) % len(rows)] for offset in range(actual_groups)]
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


async def load_training_state(training_client: Any, state_path: str) -> None:
    load_fn = getattr(training_client, "load_state_async", None)
    if callable(load_fn):
        await resolve_api_result_async(load_fn(state_path))
        return
    load_fn = getattr(training_client, "load_state", None)
    if not callable(load_fn):
        raise RuntimeError(
            "Training client must expose load_state for --load-checkpoint-path"
        )
    resolve_api_result(load_fn(state_path))


async def load_training_state_with_optimizer(
    training_client: Any, state_path: str
) -> None:
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
    resume_state_path: str | None = None,
    load_checkpoint_weights_path: str | None = None,
) -> Any:
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
    elif load_checkpoint_weights_path:
        print(f"Loading weights from checkpoint: {load_checkpoint_weights_path}")
        await load_training_state(training_client, load_checkpoint_weights_path)

    return training_client


async def save_weights_for_sampling(training_client: Any) -> Any:
    fn = getattr(training_client, "save_weights_and_get_sampling_client_async", None)
    if callable(fn):
        return await resolve_api_result_async(fn(name="eval"))
    fn = getattr(training_client, "save_weights_and_get_sampling_client", None)
    if callable(fn):
        return resolve_api_result(fn(name="eval"))
    raise RuntimeError(
        "Training client must expose save_weights_and_get_sampling_client"
    )


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
    print(
        "warning: training client does not expose save_weights_for_sampler(); skipping sampler checkpoint"
    )
    return None


async def save_sampling_client(training_client: Any, name: str) -> Any:
    save_fn = getattr(
        training_client, "save_weights_and_get_sampling_client_async", None
    )
    if callable(save_fn):
        return await resolve_api_result_async(save_fn(name=name))
    save_fn = getattr(training_client, "save_weights_and_get_sampling_client", None)
    if callable(save_fn):
        return resolve_api_result(save_fn(name=name))
    raise RuntimeError(
        "Training client must expose save_weights_and_get_sampling_client"
    )


# Keep historical helper aliases for local tests and older notes while the
# canonical save_* names become the primary contract across experiments.
async def save_sampling_client_async(training_client: Any, name: str) -> Any:
    return await save_sampling_client(training_client, name=name)


async def save_state_async_compat(training_client: Any, name: str) -> Any:
    save_state_async = require_async_method(
        training_client,
        "save_state_async",
        "LoRA training client",
    )
    return await resolve_api_result_async(await save_state_async(name=name))


async def save_weights_for_sampler_async_compat(training_client: Any, name: str) -> Any:
    save_weights_async = require_async_method(
        training_client,
        "save_weights_for_sampler_async",
        "LoRA training client",
    )
    return await resolve_api_result_async(await save_weights_async(name=name))


async def provision_training_client(
    service_client: Any,
    args: argparse.Namespace,
    *,
    resume_state_path: str | None = None,
    load_checkpoint_weights_path: str | None = None,
) -> Any:
    create_timeout = max(float(args.client_create_timeout), 1.0)
    print(
        f">> training_client_create model={args.base_model} rank={args.rank} timeout={create_timeout:.0f}s",
        flush=True,
    )
    training_client = await asyncio.wait_for(
        create_training_client(
            service_client,
            args,
            resume_state_path=resume_state_path,
            load_checkpoint_weights_path=load_checkpoint_weights_path,
        ),
        timeout=create_timeout,
    )
    print(">> training_client ready", flush=True)
    return training_client


def validate_grpo_datum_inputs(
    *,
    full_tokens: list[int],
    target_tokens: list[int],
    logprobs: list[float],
    advantages: list[float],
    weights: list[float],
) -> str | None:
    expected_len = len(full_tokens) - 1
    if expected_len <= 0:
        return f"expected_len={expected_len}"
    lengths = {
        "target_tokens": len(target_tokens),
        "logprobs": len(logprobs),
        "advantages": len(advantages),
        "weights": len(weights),
    }
    if any(length != expected_len for length in lengths.values()):
        return f"length_mismatch expected={expected_len} got={lengths}"
    for name, values in (
        ("logprobs", logprobs),
        ("advantages", advantages),
        ("weights", weights),
    ):
        for index, value in enumerate(values):
            if not math.isfinite(float(value)):
                return f"non_finite_{name}[{index}]={value}"
    return None


async def enqueue_forward_backward_async(
    training_client: Any, datums: list[types.Datum], loss_fn: str
) -> Any:
    forward_backward_async = require_async_method(
        training_client,
        "forward_backward_async",
        "LoRA training client",
    )
    return await forward_backward_async(datums, loss_fn)


async def enqueue_optim_step_async(training_client: Any, learning_rate: float) -> Any:
    adam_params = types.AdamParams(learning_rate=learning_rate)
    optim_step_async = require_async_method(
        training_client,
        "optim_step_async",
        "LoRA training client",
    )
    return await optim_step_async(adam_params)


def build_grpo_datums_from_rollout_groups(
    rollout_groups: list[PromptSamplingResult],
    tokenizer: Any,
    args: argparse.Namespace,
) -> tuple[list[types.Datum], dict[str, float], list[dict[str, Any]]]:
    datums: list[types.Datum] = []
    rewards_seen: list[float] = []
    trajectory_total = 0
    correct_trajectory_total = 0
    formatted_trajectory_total = 0
    overlong_trajectory_total = 0
    trajectory_token_cap_hits = 0
    filtered_trajectory_total = 0
    overlong_penalty_sum = 0.0
    reward_zeroed_for_overmax_total = 0
    invalid_logprob_trajectory_total = 0
    rollout_records: list[dict[str, Any]] = []

    for rollout in sorted(rollout_groups, key=lambda item: item.row_index):
        token_groups, logprob_groups = extract_trajectory_tokens_and_logprobs(
            rollout.result
        )
        group_record = {
            "step": rollout.step,
            "row_index": rollout.row_index,
            "sampler_step": rollout.sampler_step,
            "id": rollout.row["id"],
            "question": rollout.row["question"],
            "gold_answer": normalize_answer(rollout.row["answer"]),
            "num_trajectories": 0,
            "num_correct": 0,
            "num_formatted": 0,
            "num_datums": 0,
            "num_valid_logprob_trajectories": 0,
            "num_invalid_logprob_trajectories": 0,
            "reward_mean": 0.0,
            "reward_min": 0.0,
            "reward_max": 0.0,
            "status": "empty_group",
            "trajectory_answers": [],
            "trajectory_correct": [],
            "trajectory_format_valid": [],
            "trajectory_base_rewards": [],
            "trajectory_rewards": [],
            "trajectory_advantages": [],
            "trajectory_token_counts": [],
            "trajectory_overlong_penalties": [],
            "trajectory_reward_zeroed_for_overmax": [],
            "trajectory_hit_token_cap": [],
            "trajectory_overlong_filtered": [],
            "trajectory_has_full_logprobs": [],
            "trajectory_used_for_update": [],
            "trajectory_text_preview": [],
            "first_trajectory_text": None,
        }
        if not token_groups:
            rollout_records.append(group_record)
            continue

        group_base_rewards: list[float] = []
        group_rewards: list[float] = []
        group_answers: list[str | None] = []
        group_correct: list[bool] = []
        group_format_valid: list[bool] = []
        group_token_counts: list[int] = []
        group_overlong_penalties: list[float] = []
        group_reward_zeroed_for_overmax: list[bool] = []
        group_hit_token_cap: list[bool] = []
        group_overlong_filtered: list[bool] = []
        group_has_full_logprobs: list[bool] = []
        group_text_preview: list[str] = []
        update_candidate_scores: list[float] = []
        update_tokens: list[list[int]] = []
        update_logprobs: list[list[float]] = []
        update_trajectory_indexes: list[int] = []
        valid_logprob_trajectories = 0
        first_trajectory_text: str | None = None
        group_invalid_logprob_trajectories = 0

        for trajectory_index, response_tokens in enumerate(token_groups):
            if not response_tokens:
                continue
            (
                response,
                reward,
                base_reward,
                extracted,
                format_valid,
                correct,
                overlong_penalty,
                hit_token_cap,
                overlong_filtered,
                reward_zeroed_for_overmax,
            ) = score_response_tokens(
                tokenizer, response_tokens, rollout.row["answer"], args
            )
            raw_logprobs = (
                logprob_groups[trajectory_index]
                if trajectory_index < len(logprob_groups)
                else []
            )
            has_full_logprobs = len(raw_logprobs) >= len(response_tokens)
            logprobs = (
                list(raw_logprobs[: len(response_tokens)]) if has_full_logprobs else []
            )
            group_base_rewards.append(base_reward)
            group_rewards.append(reward)
            group_answers.append(extracted)
            group_correct.append(correct)
            group_format_valid.append(format_valid)
            group_token_counts.append(len(response_tokens))
            group_overlong_penalties.append(overlong_penalty)
            group_reward_zeroed_for_overmax.append(reward_zeroed_for_overmax)
            group_hit_token_cap.append(hit_token_cap)
            group_overlong_filtered.append(overlong_filtered)
            group_has_full_logprobs.append(has_full_logprobs)
            group_text_preview.append(preview_text(response, 240))
            if first_trajectory_text is None:
                first_trajectory_text = response
            rewards_seen.append(reward)
            overlong_penalty_sum += overlong_penalty
            if overlong_penalty > 0.0:
                overlong_trajectory_total += 1
            if reward_zeroed_for_overmax:
                reward_zeroed_for_overmax_total += 1
            if hit_token_cap:
                trajectory_token_cap_hits += 1
            if overlong_filtered:
                filtered_trajectory_total += 1
            trajectory_total += 1
            if format_valid:
                formatted_trajectory_total += 1
            if correct:
                correct_trajectory_total += 1
            if not has_full_logprobs:
                invalid_logprob_trajectory_total += 1
                group_invalid_logprob_trajectories += 1
                print(
                    "!! skipping_trajectory_missing_logprobs "
                    f"step={rollout.step} row_index={rollout.row_index} trajectory_index={trajectory_index + 1} "
                    f"response_tokens={len(response_tokens)} logprobs={len(raw_logprobs)}",
                    flush=True,
                )
                continue
            valid_logprob_trajectories += 1
            if overlong_filtered:
                continue
            update_candidate_scores.append(reward)
            update_tokens.append(response_tokens)
            update_logprobs.append(logprobs)
            update_trajectory_indexes.append(len(group_rewards) - 1)

        if not group_rewards:
            rollout_records.append(group_record)
            continue

        trajectory_advantages = [0.0] * len(group_rewards)
        update_advantages: list[float] = []
        if update_candidate_scores:
            update_mean_score = sum(update_candidate_scores) / len(
                update_candidate_scores
            )
            score_sq_diffs = [
                (score - update_mean_score) ** 2 for score in update_candidate_scores
            ]
            if len(update_candidate_scores) > 1:
                update_score_std = math.sqrt(
                    sum(score_sq_diffs) / (len(update_candidate_scores) - 1)
                )
            else:
                update_score_std = 0.0
            update_advantages = [
                (score - update_mean_score) / (update_score_std + 1e-6)
                for score in update_candidate_scores
            ]
            for trajectory_index, advantage in zip(
                update_trajectory_indexes, update_advantages, strict=True
            ):
                trajectory_advantages[trajectory_index] = advantage
        eligible_for_update = [
            has_full_logprobs and not filtered
            for has_full_logprobs, filtered in zip(
                group_has_full_logprobs,
                group_overlong_filtered,
                strict=True,
            )
        ]
        has_any_eligible = any(eligible_for_update)
        if valid_logprob_trajectories == 0:
            status = "missing_logprobs"
        elif not has_any_eligible:
            status = "all_overlong_filtered"
        elif not update_advantages:
            status = "flat_zero_reward_std"
        else:
            status = "used_for_update"
        flat_advantages = status != "used_for_update"
        group_record.update(
            {
                "num_trajectories": len(group_rewards),
                "num_correct": sum(1 for item in group_correct if item),
                "num_formatted": sum(1 for item in group_format_valid if item),
                "num_datums": (
                    0
                    if flat_advantages
                    else sum(1 for item in eligible_for_update if item)
                ),
                "num_valid_logprob_trajectories": valid_logprob_trajectories,
                "num_invalid_logprob_trajectories": group_invalid_logprob_trajectories,
                "reward_mean": sum(group_rewards) / len(group_rewards),
                "reward_min": min(group_rewards),
                "reward_max": max(group_rewards),
                "status": status,
                "trajectory_answers": group_answers,
                "trajectory_correct": group_correct,
                "trajectory_format_valid": group_format_valid,
                "trajectory_base_rewards": group_base_rewards,
                "trajectory_rewards": group_rewards,
                "trajectory_advantages": trajectory_advantages,
                "trajectory_token_counts": group_token_counts,
                "trajectory_overlong_penalties": group_overlong_penalties,
                "trajectory_reward_zeroed_for_overmax": group_reward_zeroed_for_overmax,
                "trajectory_hit_token_cap": group_hit_token_cap,
                "trajectory_overlong_filtered": group_overlong_filtered,
                "trajectory_has_full_logprobs": group_has_full_logprobs,
                "trajectory_used_for_update": (
                    [False] * len(group_rewards)
                    if flat_advantages
                    else eligible_for_update
                ),
                "trajectory_text_preview": group_text_preview,
                "first_trajectory_text": first_trajectory_text,
            }
        )
        rollout_records.append(group_record)
        if flat_advantages:
            continue

        prefix_len = len(rollout.prompt_tokens) - 1
        for response_tokens, logprobs, advantage in zip(
            update_tokens,
            update_logprobs,
            update_advantages,
            strict=True,
        ):
            full_tokens = rollout.prompt_tokens + response_tokens
            target_tokens = [0] * prefix_len + response_tokens
            padded_logprobs = [0.0] * prefix_len + logprobs
            padded_advantages = [0.0] * prefix_len + [advantage] * len(response_tokens)
            # Current tinker training expects an explicit loss mask/weights tensor.
            padded_weights = [0.0] * prefix_len + [1.0] * len(response_tokens)
            datum_error = validate_grpo_datum_inputs(
                full_tokens=full_tokens,
                target_tokens=target_tokens,
                logprobs=padded_logprobs,
                advantages=padded_advantages,
                weights=padded_weights,
            )
            if datum_error is not None:
                print(
                    "!! skipping_invalid_grpo_datum "
                    f"step={rollout.step} row_index={rollout.row_index} reason={datum_error}",
                    flush=True,
                )
                continue
            datums.append(
                types.Datum(
                    model_input=types.ModelInput.from_ints(tokens=full_tokens[:-1]),
                    loss_fn_inputs={
                        "target_tokens": make_int_tensor_data(target_tokens),
                        "logprobs": make_float_tensor_data(padded_logprobs),
                        "advantages": make_float_tensor_data(padded_advantages),
                        "weights": make_float_tensor_data(padded_weights),
                    },
                )
            )

    sampler_steps = [rollout.sampler_step for rollout in rollout_groups]
    dynamic_rollout_wave_indices = [
        rollout.dynamic_rollout_wave_index for rollout in rollout_groups
    ]
    nonflat_accepts = [rollout.dynamic_accepted_non_flat for rollout in rollout_groups]
    return (
        datums,
        {
            "rl_reward_mean": (
                sum(rewards_seen) / len(rewards_seen) if rewards_seen else 0.0
            ),
            "rl_group_accuracy": (
                correct_trajectory_total / trajectory_total if trajectory_total else 0.0
            ),
            "rl_final_answer_format_success": (
                formatted_trajectory_total / trajectory_total
                if trajectory_total
                else 0.0
            ),
            "rl_overlong_penalty_rate": (
                overlong_trajectory_total / trajectory_total
                if trajectory_total
                else 0.0
            ),
            "rl_overlong_penalty_mean": (
                overlong_penalty_sum / trajectory_total if trajectory_total else 0.0
            ),
            "rl_overmax_zeroed_rate": (
                reward_zeroed_for_overmax_total / trajectory_total
                if trajectory_total
                else 0.0
            ),
            "rl_hit_token_cap_rate": (
                trajectory_token_cap_hits / trajectory_total
                if trajectory_total
                else 0.0
            ),
            "rl_overlong_filtered_rate": (
                filtered_trajectory_total / trajectory_total
                if trajectory_total
                else 0.0
            ),
            "rl_invalid_logprob_rate": (
                invalid_logprob_trajectory_total / trajectory_total
                if trajectory_total
                else 0.0
            ),
            "rl_invalid_logprob_trajectories": float(invalid_logprob_trajectory_total),
            "rl_dynamic_rollout_wave_mean": (
                sum(dynamic_rollout_wave_indices) / len(dynamic_rollout_wave_indices)
                if dynamic_rollout_wave_indices
                else 0.0
            ),
            "rl_dynamic_rollout_wave_max": (
                float(max(dynamic_rollout_wave_indices))
                if dynamic_rollout_wave_indices
                else 0.0
            ),
            "rl_dynamic_nonflat_accept_rate": (
                sum(1.0 for item in nonflat_accepts if item) / len(nonflat_accepts)
                if nonflat_accepts
                else 0.0
            ),
            "rl_datums_per_step": float(len(datums)),
            "rl_samples_per_step": float(trajectory_total),
            "rl_sampler_step_min": float(min(sampler_steps)) if sampler_steps else 0.0,
            "rl_sampler_step_max": float(max(sampler_steps)) if sampler_steps else 0.0,
        },
        rollout_records,
    )


async def grpo_train_loop(
    service_client: Any,
    training_client: Any,
    tokenizer: Any,
    train_rows: list[dict[str, str]],
    eval_rows: list[dict[str, str]],
    args: argparse.Namespace,
    output_dir: Path,
    resume_loop_state: ResumeLoopState | None = None,
) -> tuple[dict[str, float], list[dict[str, Any]], dict[str, float] | None, str | None]:
    start_step = (
        resume_loop_state.completed_steps if resume_loop_state is not None else 0
    )

    if args.grpo_steps <= 0:
        return (
            {
                "rl_reward_mean": 0.0,
                "rl_group_accuracy": 0.0,
                "rl_final_answer_format_success": 0.0,
                "rl_overlong_penalty_rate": 0.0,
                "rl_overlong_penalty_mean": 0.0,
                "rl_overmax_zeroed_rate": 0.0,
                "rl_hit_token_cap_rate": 0.0,
                "rl_overlong_filtered_rate": 0.0,
                "rl_invalid_logprob_rate": 0.0,
                "rl_invalid_logprob_trajectories": 0.0,
                "rl_dynamic_rollout_wave_mean": 0.0,
                "rl_dynamic_rollout_wave_max": 0.0,
                "rl_dynamic_nonflat_accept_rate": 0.0,
                "rl_datums_per_step": 0.0,
            },
            [],
            None,
            None,
        )
    if not train_rows:
        raise RuntimeError("Training data is empty")

    effective_rows = (
        train_rows[: args.grpo_max_train_prompts]
        if args.grpo_max_train_prompts > 0
        else train_rows
    )
    random.Random(args.seed + 1).shuffle(effective_rows)
    groups_per_batch = max(args.groups_per_batch, 1)
    remaining_steps = max(args.grpo_steps - start_step, 0)
    if resume_loop_state is not None:
        record_suffix = (
            f" record={resume_loop_state.record_path}"
            if resume_loop_state.record_path
            else ""
        )
        print(
            f">> train_resume completed_steps={resume_loop_state.completed_steps} next_step={resume_loop_state.next_step} "
            f"remaining_steps={remaining_steps} source={resume_loop_state.source}{record_suffix}",
            flush=True,
        )
    configured_stream_minibatches = max(args.stream_minibatches_per_step, 1)
    train_max_concurrent_requests = max(args.max_concurrent_requests, 1)
    train_sampling_semaphore = asyncio.Semaphore(train_max_concurrent_requests)
    eval_every_steps = max(args.eval_every_steps, 0)

    scheduled_prompt_group_jobs_by_step: dict[int, list[PromptSamplingJob]] = {}
    for step in range(start_step + 1, args.grpo_steps + 1):
        step_prompt_group_rows = make_prompt_group_batch(
            effective_rows, step, args.groups_per_batch
        )
        step_prompt_group_jobs: list[PromptSamplingJob] = []
        for row_index, row in enumerate(step_prompt_group_rows, start=1):
            step_prompt_group_jobs.append(
                PromptSamplingJob(
                    step=step,
                    row_index=row_index,
                    row=row,
                    prompt_tokens=build_generation_prompt_tokens(
                        tokenizer, row["messages"]
                    ),
                    sampling_params=types.SamplingParams(
                        max_tokens=args.rl_max_tokens,
                        temperature=args.rl_temperature,
                        seed=args.seed + step * 100 + row_index,
                        stop_token_ids=[tokenizer.eos_token_id],
                    ),
                )
            )
        scheduled_prompt_group_jobs_by_step[step] = step_prompt_group_jobs

    total_prompt_groups = sum(
        len(step_prompt_group_jobs)
        for step_prompt_group_jobs in scheduled_prompt_group_jobs_by_step.values()
    )
    print(
        ">> train_config "
        f"grpo_steps={args.grpo_steps} eval_every_steps={eval_every_steps} save_state_every_steps={args.save_state_every_steps} "
        f"groups_per_batch={groups_per_batch} group_size={args.group_size} "
        f"stream_minibatches_per_step={configured_stream_minibatches} "
        f"min_accepted_groups={args.min_accepted_groups} min_started_minibatches={args.min_started_minibatches} "
        f"tail_grace_seconds={args.tail_grace_seconds:.1f} "
        f"prompt_groups={total_prompt_groups} "
        f"train_max_concurrent_requests={train_max_concurrent_requests} rl_temperature={args.rl_temperature} rl_loss={args.rl_loss} "
        f"apply_overlong_filtering={args.apply_overlong_filtering} dynamic_sampling_type={args.dynamic_sampling_type} "
        f"dynamic_sampling_max_rollout_waves={args.dynamic_sampling_max_rollout_waves}",
        flush=True,
    )

    sampler = await save_sampling_client(
        training_client, name=f"grpo-step-{time.time_ns()}-{start_step}"
    )
    sampler_step = start_step
    total_completed_samples = 0
    train_started_at = time.time()
    last_metrics = {
        "rl_reward_mean": 0.0,
        "rl_group_accuracy": 0.0,
        "rl_final_answer_format_success": 0.0,
        "rl_overlong_penalty_rate": 0.0,
        "rl_overlong_penalty_mean": 0.0,
        "rl_overmax_zeroed_rate": 0.0,
        "rl_hit_token_cap_rate": 0.0,
        "rl_overlong_filtered_rate": 0.0,
        "rl_invalid_logprob_rate": 0.0,
        "rl_invalid_logprob_trajectories": 0.0,
        "rl_dynamic_rollout_wave_mean": 0.0,
        "rl_dynamic_rollout_wave_max": 0.0,
        "rl_dynamic_nonflat_accept_rate": 0.0,
        "rl_datums_per_step": 0.0,
    }
    periodic_eval_history: list[dict[str, Any]] = []
    cached_final_eval: dict[str, float] | None = None
    background_eval_task: asyncio.Task[dict[str, float]] | None = None
    background_eval_step: int | None = None
    background_eval_output_dir: Path | None = None
    background_eval_sampler_path: str | None = None
    pending_eval_specs: list[tuple[int, Path, str]] = []
    train_dir = output_dir / "train"
    eval_dir = output_dir / "eval"
    metrics_jsonl_path = train_dir / "metrics.jsonl"
    checkpoints_jsonl_path = train_dir / "checkpoints.jsonl"
    train_rollouts_jsonl_path = train_dir / "rollouts.jsonl"
    train_failures_log_path = train_dir / "failures.log"
    train_failures_jsonl_path = train_dir / "failures.jsonl"
    periodic_eval_metrics_jsonl_path = eval_dir / "metrics.jsonl"
    print(f"@@ train_rollouts={os.path.relpath(train_rollouts_jsonl_path, Path.cwd())}")
    print(
        f"@@ train_failures_log={os.path.relpath(train_failures_log_path, Path.cwd())}"
    )
    print(
        f"@@ train_failures_jsonl={os.path.relpath(train_failures_jsonl_path, Path.cwd())}"
    )
    print(
        f"@@ periodic_eval_metrics_jsonl={os.path.relpath(periodic_eval_metrics_jsonl_path, Path.cwd())}"
    )
    if resume_loop_state is None:
        train_failures_log_path.write_text("", encoding="utf-8")
        train_failures_jsonl_path.write_text("", encoding="utf-8")
        periodic_eval_metrics_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        periodic_eval_metrics_jsonl_path.write_text("", encoding="utf-8")
    else:
        periodic_eval_metrics_jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    async def save_recovery_checkpoint(step: int, reason: str) -> tuple[str, str, str]:
        checkpoint_name = f"{step:06d}"
        state_path = await save_training_state(training_client, checkpoint_name)
        sampler_path = await save_sampler_checkpoint(training_client, checkpoint_name)
        if not state_path or not sampler_path:
            raise RuntimeError(
                "GRPO periodic checkpoints require both save_state and save_weights_for_sampler support"
            )
        checkpoint_record = {
            "name": checkpoint_name,
            "step": step,
            "completed_steps": step,
            "next_step": min(step + 1, args.grpo_steps + 1),
            "batch": step,
            "epoch": 0,
            "state_path": state_path,
            "sampler_path": sampler_path,
        }
        append_jsonl(checkpoints_jsonl_path, checkpoint_record)
        print(
            f"@@ checkpoint step={step}/{args.grpo_steps} reason={reason} name={checkpoint_name} state_path={state_path} sampler_path={sampler_path}",
            flush=True,
        )
        return checkpoint_name, state_path, sampler_path

    async def start_next_background_eval_if_idle() -> None:
        nonlocal background_eval_task, background_eval_step, background_eval_output_dir, background_eval_sampler_path
        if background_eval_task is not None or not pending_eval_specs:
            return
        next_step, next_output_dir, next_sampler_path = pending_eval_specs.pop(0)
        background_eval_step = next_step
        background_eval_output_dir = next_output_dir
        background_eval_sampler_path = next_sampler_path
        print(
            f">> eval_start step={next_step}/{args.grpo_steps} queued_remaining={len(pending_eval_specs)} "
            f"output_dir={os.path.relpath(next_output_dir, Path.cwd())} sampler_path={next_sampler_path}",
            flush=True,
        )

        async def run_background_eval(
            model_path: str, eval_output_dir: Path, step: int
        ) -> dict[str, float]:
            eval_sampler = await create_sampling_client(service_client, model_path)
            return await evaluate_with_sampler(
                eval_sampler,
                tokenizer,
                eval_rows,
                args,
                eval_output_dir,
                log_label=f"bg_eval_step_{step:04d}",
                verbose_item_logs=True,
            )

        background_eval_task = asyncio.create_task(
            run_background_eval(next_sampler_path, next_output_dir, next_step),
            name=f"background_eval_step_{next_step}",
        )

    async def collect_background_eval_if_done() -> None:
        nonlocal background_eval_task, background_eval_step, background_eval_output_dir, background_eval_sampler_path, cached_final_eval
        if background_eval_task is None:
            await start_next_background_eval_if_idle()
            return
        if not background_eval_task.done():
            return

        finished_step = background_eval_step
        finished_output_dir = background_eval_output_dir
        finished_sampler_path = background_eval_sampler_path
        try:
            eval_metrics = await background_eval_task
        finally:
            background_eval_task = None
            background_eval_step = None
            background_eval_output_dir = None
            background_eval_sampler_path = None

        if finished_step is not None and finished_output_dir is not None:
            periodic_eval_history.append(
                {
                    "step": finished_step,
                    "output_dir": os.path.relpath(finished_output_dir, Path.cwd()),
                    "sampler_path": finished_sampler_path,
                    "metrics": eval_metrics,
                }
            )
            append_periodic_eval_record(
                periodic_eval_metrics_jsonl_path,
                step=finished_step,
                total_steps=args.grpo_steps,
                eval_output_dir=finished_output_dir,
                status="ok",
                sampler_path=finished_sampler_path,
                eval_metrics=eval_metrics,
            )
            print(
                "== eval_done "
                f"step={finished_step}/{args.grpo_steps} {format_eval_metric_summary(eval_metrics)}",
                flush=True,
            )
            if finished_step == args.grpo_steps:
                cached_final_eval = eval_metrics
        await start_next_background_eval_if_idle()

    for step in range(start_step + 1, args.grpo_steps + 1):
        step_started_at = time.time()
        step_prompt_group_jobs = scheduled_prompt_group_jobs_by_step[step]
        expected_prompt_groups = len(step_prompt_group_jobs)
        requested_step_samples = expected_prompt_groups * max(args.group_size, 1)
        step_completed_samples = 0
        step_progress_stride = max(args.group_size * 8, 64)
        next_step_progress_threshold = min(step_progress_stride, requested_step_samples)
        step_minibatches = min(
            configured_stream_minibatches, max(expected_prompt_groups, 1)
        )
        while step_minibatches > 1 and expected_prompt_groups % step_minibatches != 0:
            step_minibatches -= 1
        groups_per_minibatch = max(expected_prompt_groups // step_minibatches, 1)
        effective_min_started_minibatches = (
            min(max(args.min_started_minibatches, 0), step_minibatches)
            if args.min_started_minibatches > 0
            else max(step_minibatches - 1, 1)
        )
        effective_min_accepted_groups = (
            min(max(args.min_accepted_groups, 0), expected_prompt_groups)
            if args.min_accepted_groups > 0
            else groups_per_minibatch * effective_min_started_minibatches
        )
        tail_grace_seconds = max(args.tail_grace_seconds, 0.0)

        def handle_group_result(sample_result: Any) -> None:
            nonlocal total_completed_samples, step_completed_samples, next_step_progress_threshold
            token_groups, _ = extract_trajectory_tokens_and_logprobs(sample_result)
            completed_now = len(token_groups)
            total_completed_samples += completed_now
            step_completed_samples += completed_now
            while (
                next_step_progress_threshold
                and step_completed_samples >= next_step_progress_threshold
            ):
                print(
                    "++ train "
                    f"step={step}/{args.grpo_steps} raw_sample_attempts={next_step_progress_threshold} "
                    f"target_samples={requested_step_samples} "
                    f"total_raw_sample_attempts={total_completed_samples} "
                    f"step_elapsed={format_elapsed(time.time() - step_started_at)}",
                    flush=True,
                )
                next_step_progress_threshold += step_progress_stride
                if next_step_progress_threshold > requested_step_samples:
                    next_step_progress_threshold = 0

        max_rollout_waves = 1
        if args.dynamic_sampling_type == "filter":
            max_rollout_waves = max(args.dynamic_sampling_max_rollout_waves, 1)
            if args.dynamic_sampling_max_rollout_waves <= 0:
                max_rollout_waves = 0

        prompt_group_start_offset = ((step - 1) * expected_prompt_groups) % len(
            effective_rows
        )
        next_sampling_offset = expected_prompt_groups
        next_row_index = expected_prompt_groups + 1

        def build_prompt_job(
            row_index: int, row: dict[str, str], seed: int
        ) -> PromptSamplingJob:
            return PromptSamplingJob(
                step=step,
                row_index=row_index,
                row=row,
                prompt_tokens=build_generation_prompt_tokens(
                    tokenizer, row["messages"]
                ),
                sampling_params=types.SamplingParams(
                    max_tokens=args.rl_max_tokens,
                    temperature=args.rl_temperature,
                    seed=seed,
                    stop_token_ids=[tokenizer.eos_token_id],
                ),
            )

        def make_accumulation_job() -> PromptSamplingJob:
            nonlocal next_sampling_offset, next_row_index
            row = effective_rows[
                (prompt_group_start_offset + next_sampling_offset) % len(effective_rows)
            ]
            seed = args.seed + step * 100000 + next_sampling_offset
            row_index = next_row_index
            next_sampling_offset += 1
            next_row_index += 1
            return build_prompt_job(row_index, row, seed)

        async def sample_prompt_group(job: PromptSamplingJob) -> PromptSamplingResult:
            base_seed = getattr(job.sampling_params, "seed", None)
            sampling_requests = [
                (
                    job.prompt_tokens,
                    1,
                    copy_sampling_params_with_seed(
                        job.sampling_params,
                        None if base_seed is None else base_seed + trajectory_index,
                    ),
                )
                for trajectory_index in range(max(args.group_size, 1))
            ]
            result = await sample_many_async(
                sampler=sampler,
                sampling_requests=sampling_requests,
                max_in_flight=max(args.group_size, 1),
                semaphore=train_sampling_semaphore,
                on_result=None,
            )
            handle_group_result(result)
            return PromptSamplingResult(
                step=job.step,
                row_index=job.row_index,
                sampler_step=sampler_step,
                row=job.row,
                prompt_tokens=job.prompt_tokens,
                result=result,
            )

        prompt_group_tasks: dict[asyncio.Task[PromptSamplingResult], int] = {}
        accepted_rollouts: list[PromptSamplingResult] = []
        ready_rollouts: list[PromptSamplingResult] = []
        trained_rollouts: list[PromptSamplingResult] = []
        forward_backward_futures: list[Any] = []
        started_minibatches = 0
        tail_eligible_at: float | None = None
        early_stop_triggered = False
        early_stop_reason = ""
        rollout_wave_count = 0

        def launch_prompt_group(
            job: PromptSamplingJob, rollout_wave_index: int
        ) -> None:
            task = asyncio.create_task(
                sample_prompt_group(job),
                name=f"grpo_prompt_group_{step}_{rollout_wave_index}_{job.row_index}",
            )
            prompt_group_tasks[task] = rollout_wave_index

        def launch_sampling_batch(num_groups: int) -> None:
            nonlocal rollout_wave_count
            if num_groups <= 0:
                return
            if max_rollout_waves > 0 and rollout_wave_count >= max_rollout_waves:
                raise RuntimeError(
                    f"dynamic sampling reached max_rollout_waves={max_rollout_waves} before collecting "
                    f"{expected_prompt_groups} non-flat groups"
                )
            rollout_wave_index = rollout_wave_count
            rollout_wave_count += 1
            if rollout_wave_index > 0:
                max_rollout_waves_label = (
                    str(max_rollout_waves) if max_rollout_waves > 0 else "inf"
                )
                print(
                    ">> dynamic_sampling_filter "
                    f"step={step}/{args.grpo_steps} rollout_wave={rollout_wave_index + 1}/{max_rollout_waves_label} "
                    f"launch_groups={num_groups} collected_groups={len(accepted_rollouts)}/{expected_prompt_groups} "
                    f"raw_sample_attempts={step_completed_samples}",
                    flush=True,
                )
            if rollout_wave_index == 0:
                wave_jobs = step_prompt_group_jobs
            else:
                wave_jobs = [make_accumulation_job() for _ in range(num_groups)]
            for job in wave_jobs:
                launch_prompt_group(job, rollout_wave_index)

        async def enqueue_ready_minibatches() -> None:
            nonlocal started_minibatches
            while len(ready_rollouts) >= groups_per_minibatch:
                minibatch_rollouts = ready_rollouts[:groups_per_minibatch]
                del ready_rollouts[:groups_per_minibatch]
                started_minibatches += 1
                trained_rollouts.extend(minibatch_rollouts)
                minibatch_datums, _minibatch_metrics, _ = (
                    build_grpo_datums_from_rollout_groups(
                        minibatch_rollouts,
                        tokenizer,
                        args,
                    )
                )
                print(
                    " >> train_minibatch"[1:] + " "
                    f"step={step}/{args.grpo_steps} minibatch={started_minibatches}/{step_minibatches} "
                    f"groups={len(minibatch_rollouts)}/{groups_per_minibatch} datums={len(minibatch_datums)} "
                    f"accepted_groups={len(accepted_rollouts)}/{expected_prompt_groups} "
                    f"raw_sample_attempts={step_completed_samples} sampled_with={sampler_step}",
                    flush=True,
                )
                if minibatch_datums:
                    forward_backward_futures.append(
                        await enqueue_forward_backward_async(
                            training_client, minibatch_datums, loss_fn=args.rl_loss
                        )
                    )

        launch_sampling_batch(expected_prompt_groups)
        print(
            " >> train"[1:] + " "
            f"step={step}/{args.grpo_steps} target_groups={expected_prompt_groups} target_samples={requested_step_samples} "
            f"max_concurrent_requests={train_max_concurrent_requests} group_size={args.group_size} "
            f"stream_minibatches={step_minibatches} groups_per_minibatch={groups_per_minibatch} "
            f"min_accepted_groups={effective_min_accepted_groups} min_started_minibatches={effective_min_started_minibatches} "
            f"tail_grace_seconds={tail_grace_seconds:.1f} "
            f"total_elapsed={format_elapsed(time.time() - train_started_at)} sampled_with={sampler_step}",
            flush=True,
        )

        async def log_step_heartbeat() -> None:
            while True:
                await asyncio.sleep(360)
                pending_groups = len(prompt_group_tasks)
                if pending_groups == 0:
                    return
                done_groups = len(accepted_rollouts)
                nominal_accepted_samples = done_groups * max(args.group_size, 1)
                print(
                    ".. train "
                    f"step={step}/{args.grpo_steps} accepted_groups={done_groups}/{expected_prompt_groups} "
                    f"nominal_accepted_samples={nominal_accepted_samples}/{requested_step_samples} "
                    f"raw_sample_attempts={step_completed_samples} "
                    f"total_raw_sample_attempts={total_completed_samples} "
                    f"ready_groups={len(ready_rollouts)} minibatches_started={started_minibatches}/{step_minibatches} "
                    f"step_elapsed={format_elapsed(time.time() - step_started_at)} "
                    f"total_elapsed={format_elapsed(time.time() - train_started_at)}",
                    flush=True,
                )

        def tail_cutoff_reached(now: float) -> bool:
            if tail_grace_seconds <= 0.0 or tail_eligible_at is None:
                return False
            return now - tail_eligible_at >= tail_grace_seconds

        heartbeat_task = asyncio.create_task(
            log_step_heartbeat(), name=f"grpo_step_heartbeat_{step}"
        )
        try:
            while prompt_group_tasks:
                done_tasks, _pending = await asyncio.wait(
                    prompt_group_tasks.keys(),
                    timeout=1.0,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if not done_tasks:
                    if tail_cutoff_reached(time.time()):
                        early_stop_triggered = True
                        early_stop_reason = (
                            f"tail_timeout accepted_groups={len(accepted_rollouts)} "
                            f"trained_groups={len(trained_rollouts)} ready_groups={len(ready_rollouts)} "
                            f"min_accepted_groups={effective_min_accepted_groups} "
                            f"min_started_minibatches={effective_min_started_minibatches}"
                        )
                        break
                    continue
                for done_task in done_tasks:
                    rollout_wave_index = prompt_group_tasks.pop(done_task)
                    rollout = done_task.result()
                    lacks_signal = False
                    if args.dynamic_sampling_type == "filter":
                        token_groups, _ = extract_trajectory_tokens_and_logprobs(
                            rollout.result
                        )
                        lacks_signal = group_lacks_reward_signal(
                            tokenizer, token_groups, rollout.row["answer"], args
                        )
                        if lacks_signal:
                            continue
                    accepted_rollout = PromptSamplingResult(
                        step=rollout.step,
                        row_index=rollout.row_index,
                        sampler_step=rollout.sampler_step,
                        row=rollout.row,
                        prompt_tokens=rollout.prompt_tokens,
                        result=rollout.result,
                        dynamic_rollout_wave_index=rollout_wave_index + 1,
                        dynamic_accepted_non_flat=not lacks_signal,
                    )
                    accepted_rollouts.append(accepted_rollout)
                    ready_rollouts.append(accepted_rollout)
                    await enqueue_ready_minibatches()
                    if (
                        started_minibatches >= effective_min_started_minibatches
                        and len(accepted_rollouts) >= effective_min_accepted_groups
                    ):
                        tail_eligible_at = time.time()
                if tail_cutoff_reached(time.time()):
                    early_stop_triggered = True
                    early_stop_reason = (
                        f"tail_timeout accepted_groups={len(accepted_rollouts)} "
                        f"trained_groups={len(trained_rollouts)} ready_groups={len(ready_rollouts)} "
                        f"min_accepted_groups={effective_min_accepted_groups} "
                        f"min_started_minibatches={effective_min_started_minibatches}"
                    )
                    break
                if (
                    not prompt_group_tasks
                    and len(accepted_rollouts) < expected_prompt_groups
                ):
                    if (
                        started_minibatches >= effective_min_started_minibatches
                        and len(accepted_rollouts) >= effective_min_accepted_groups
                    ):
                        early_stop_triggered = True
                        early_stop_reason = (
                            f"filter_shortfall accepted_groups={len(accepted_rollouts)} "
                            f"trained_groups={len(trained_rollouts)} rollout_waves={rollout_wave_count} "
                            f"min_accepted_groups={effective_min_accepted_groups} "
                            f"min_started_minibatches={effective_min_started_minibatches}"
                        )
                        break
                    if args.dynamic_sampling_type == "filter":
                        missing_groups = expected_prompt_groups - len(accepted_rollouts)
                        launch_sampling_batch(missing_groups)
            if early_stop_triggered:
                pending_tasks = list(prompt_group_tasks)
                for pending_task in pending_tasks:
                    pending_task.cancel()
                if pending_tasks:
                    await asyncio.gather(*pending_tasks, return_exceptions=True)
                prompt_group_tasks.clear()
                dropped_ready_groups = len(ready_rollouts)
                ready_rollouts.clear()
                print(
                    ">> train_early_stop "
                    f"step={step}/{args.grpo_steps} reason={early_stop_reason} "
                    f"dropped_ready_groups={dropped_ready_groups} pending_groups={len(pending_tasks)}",
                    flush=True,
                )
            elif ready_rollouts:
                raise RuntimeError(
                    f"step {step} finished with {len(ready_rollouts)} unconsumed accepted groups; "
                    f"groups_per_minibatch={groups_per_minibatch}"
                )
            batch_rollouts = list(trained_rollouts)
        finally:
            heartbeat_task.cancel()
            await asyncio.gather(heartbeat_task, return_exceptions=True)

        accepted_prompt_groups = len(accepted_rollouts)
        trained_prompt_groups = len(batch_rollouts)
        dropped_prompt_groups = max(accepted_prompt_groups - trained_prompt_groups, 0)

        if forward_backward_futures:
            optim_future = await enqueue_optim_step_async(
                training_client, learning_rate=args.rl_learning_rate
            )
            for future in forward_backward_futures:
                await resolve_api_result_async(future)
            await resolve_api_result_async(optim_future)
            sampler = await save_sampling_client(
                training_client,
                name=f"grpo-step-{time.time_ns()}-{step}",
            )
        sampler_step = step

        _unused_datums, step_metrics, rollout_records = (
            build_grpo_datums_from_rollout_groups(batch_rollouts, tokenizer, args)
        )
        last_metrics = {
            "rl_reward_mean": step_metrics["rl_reward_mean"],
            "rl_group_accuracy": step_metrics["rl_group_accuracy"],
            "rl_final_answer_format_success": step_metrics[
                "rl_final_answer_format_success"
            ],
            "rl_overlong_penalty_rate": step_metrics["rl_overlong_penalty_rate"],
            "rl_overlong_penalty_mean": step_metrics["rl_overlong_penalty_mean"],
            "rl_overmax_zeroed_rate": step_metrics["rl_overmax_zeroed_rate"],
            "rl_hit_token_cap_rate": step_metrics["rl_hit_token_cap_rate"],
            "rl_overlong_filtered_rate": step_metrics["rl_overlong_filtered_rate"],
            "rl_invalid_logprob_rate": step_metrics["rl_invalid_logprob_rate"],
            "rl_invalid_logprob_trajectories": step_metrics[
                "rl_invalid_logprob_trajectories"
            ],
            "rl_dynamic_rollout_wave_mean": step_metrics[
                "rl_dynamic_rollout_wave_mean"
            ],
            "rl_dynamic_rollout_wave_max": step_metrics["rl_dynamic_rollout_wave_max"],
            "rl_dynamic_nonflat_accept_rate": step_metrics[
                "rl_dynamic_nonflat_accept_rate"
            ],
            "rl_datums_per_step": step_metrics["rl_datums_per_step"],
        }
        step_elapsed = time.time() - step_started_at
        train_elapsed = time.time() - train_started_at
        steps_completed_this_run = max(step - start_step, 1)
        average_seconds_per_step = train_elapsed / steps_completed_this_run
        eta_seconds = average_seconds_per_step * max(args.grpo_steps - step, 0)
        samples_per_second = step_metrics["rl_samples_per_step"] / max(
            step_elapsed, 1e-9
        )
        print(
            "== train "
            f"step={step}/{args.grpo_steps} accepted_groups={accepted_prompt_groups}/{expected_prompt_groups} "
            f"trained_groups={trained_prompt_groups}/{expected_prompt_groups} dropped_groups={dropped_prompt_groups} "
            f"trained_samples={int(step_metrics['rl_samples_per_step'])}/{requested_step_samples} "
            f"raw_sample_attempts={step_completed_samples} "
            f"total_raw_sample_attempts={total_completed_samples} "
            f"reward_mean={step_metrics['rl_reward_mean']:.4f} accuracy={step_metrics['rl_group_accuracy']:.4f} "
            f"format={step_metrics['rl_final_answer_format_success']:.4f} "
            f"overlong_rate={step_metrics['rl_overlong_penalty_rate']:.4f} overlong_mean={step_metrics['rl_overlong_penalty_mean']:.4f} "
            f"overmax_zeroed_rate={step_metrics['rl_overmax_zeroed_rate']:.4f} "
            f"cap_hit_rate={step_metrics['rl_hit_token_cap_rate']:.4f} filtered_rate={step_metrics['rl_overlong_filtered_rate']:.4f} "
            f"invalid_logprob_rate={step_metrics['rl_invalid_logprob_rate']:.4f} invalid_logprob_trajectories={int(step_metrics['rl_invalid_logprob_trajectories'])} "
            f"dynamic_rollout_wave_mean={step_metrics['rl_dynamic_rollout_wave_mean']:.2f} dynamic_rollout_wave_max={int(step_metrics['rl_dynamic_rollout_wave_max'])} "
            f"dynamic_accept={step_metrics['rl_dynamic_nonflat_accept_rate']:.4f} datums={int(step_metrics['rl_datums_per_step'])} early_stop={int(early_stop_triggered)} step_time={step_elapsed:.1f}s samples_per_sec={samples_per_second:.2f} "
            f"total_elapsed={format_elapsed(train_elapsed)} eta={format_elapsed(eta_seconds)} loss_fn={args.rl_loss} "
            f"sampled_with={int(step_metrics['rl_sampler_step_min'])}-{int(step_metrics['rl_sampler_step_max'])}",
            flush=True,
        )

        await collect_background_eval_if_done()

        checkpoint_reason_parts: list[str] = []
        if args.save_state_every_steps > 0 and step % args.save_state_every_steps == 0:
            checkpoint_reason_parts.append("periodic")
        if (
            args.recreate_client_every_steps > 0
            and step < args.grpo_steps
            and step % args.recreate_client_every_steps == 0
        ):
            checkpoint_reason_parts.append("recreate")

        checkpoint_name: str | None = None
        checkpoint_state_path: str | None = None
        checkpoint_sampler_path: str | None = None
        if checkpoint_reason_parts:
            checkpoint_name, checkpoint_state_path, checkpoint_sampler_path = (
                await save_recovery_checkpoint(
                    step,
                    "+".join(checkpoint_reason_parts),
                )
            )

        if eval_every_steps > 0 and step % eval_every_steps == 0:
            is_final_step = step == args.grpo_steps
            eval_output_dir = (
                output_dir / "eval"
                if is_final_step
                else output_dir / "eval" / "steps" / f"step-{step:04d}"
            )
            # Keep eval snapshots independent from periodic checkpoints so benchmark replays
            # do not depend on checkpoint metadata or checkpoint-save timing.
            eval_sampler_source = "eval_snapshot"
            eval_sampler_path = await save_sampler_checkpoint(
                training_client, f"eval-{step:06d}"
            )
            if not eval_sampler_path:
                raise RuntimeError(
                    "Periodic eval snapshots require save_weights_for_sampler support"
                )
            pending_eval_specs.append((step, eval_output_dir, eval_sampler_path))
            print(
                f">> eval_trigger step={step}/{args.grpo_steps} queued={len(pending_eval_specs)} "
                f"source={eval_sampler_source} output_dir={os.path.relpath(eval_output_dir, Path.cwd())} "
                f"sampler_path={eval_sampler_path}",
                flush=True,
            )
            await start_next_background_eval_if_idle()

        metrics_record = {
            "step": step,
            "total_steps": args.grpo_steps,
            "learning_rate": args.rl_learning_rate,
            "progress": step / max(args.grpo_steps, 1),
            "reward_mean": step_metrics["rl_reward_mean"],
            "accuracy": step_metrics["rl_group_accuracy"],
            "format": step_metrics["rl_final_answer_format_success"],
            "overlong_penalty_rate": step_metrics["rl_overlong_penalty_rate"],
            "overlong_penalty_mean": step_metrics["rl_overlong_penalty_mean"],
            "overmax_zeroed_rate": step_metrics["rl_overmax_zeroed_rate"],
            "hit_token_cap_rate": step_metrics["rl_hit_token_cap_rate"],
            "overlong_filtered_rate": step_metrics["rl_overlong_filtered_rate"],
            "invalid_logprob_rate": step_metrics["rl_invalid_logprob_rate"],
            "invalid_logprob_trajectories": int(
                step_metrics["rl_invalid_logprob_trajectories"]
            ),
            "dynamic_rollout_wave_mean": step_metrics["rl_dynamic_rollout_wave_mean"],
            "dynamic_rollout_wave_max": step_metrics["rl_dynamic_rollout_wave_max"],
            "dynamic_nonflat_accept_rate": step_metrics[
                "rl_dynamic_nonflat_accept_rate"
            ],
            "datums": int(step_metrics["rl_datums_per_step"]),
            "num_trajectories": int(step_metrics["rl_samples_per_step"]),
            "prompt_groups": expected_prompt_groups,
            "accepted_groups": accepted_prompt_groups,
            "trained_groups": trained_prompt_groups,
            "dropped_groups": dropped_prompt_groups,
            "early_stop": int(early_stop_triggered),
            "early_stop_reason": early_stop_reason,
            "group_size": args.group_size,
            "groups_per_batch": args.groups_per_batch,
            "samples_per_sec": samples_per_second,
            "time/step": step_elapsed,
            "time/total": train_elapsed,
            "time/eta": eta_seconds,
            "checkpoint_name": checkpoint_name,
            "checkpoint_state_path": checkpoint_state_path,
            "checkpoint_sampler_path": checkpoint_sampler_path,
        }
        append_jsonl(metrics_jsonl_path, metrics_record)
        for rollout_record in rollout_records:
            append_jsonl(train_rollouts_jsonl_path, rollout_record)
            if is_train_rollout_failure(rollout_record):
                append_jsonl(train_failures_jsonl_path, rollout_record)
                with train_failures_log_path.open(
                    "a", encoding="utf-8"
                ) as train_failures_handle:
                    write_train_failure_record(train_failures_handle, rollout_record)

        if checkpoint_state_path is not None and "recreate" in checkpoint_reason_parts:
            print(
                f">> training_client_recreate step={step}/{args.grpo_steps} checkpoint={checkpoint_state_path}",
                flush=True,
            )
            training_client = await provision_training_client(
                service_client,
                args,
                resume_state_path=checkpoint_state_path,
            )
            sampler = await save_sampling_client(
                training_client,
                name=f"grpo-step-{time.time_ns()}-{step}-resume",
            )
            print(
                f">> training_client_recreated step={step}/{args.grpo_steps} checkpoint={checkpoint_state_path}",
                flush=True,
            )

    await collect_background_eval_if_done()
    eval_wait_started_at = time.time()
    while background_eval_task is not None or pending_eval_specs:
        if background_eval_task is not None:
            pending_step = background_eval_step
            print(
                f">> eval_wait pending_step={pending_step}/{args.grpo_steps} queued_remaining={len(pending_eval_specs)} "
                f"elapsed={format_elapsed(time.time() - eval_wait_started_at)}",
                flush=True,
            )
            await asyncio.wait([background_eval_task], timeout=60)
        await collect_background_eval_if_done()

    final_sampler_path = None
    for attr_name in ("path", "sampler_path", "state_path"):
        value = getattr(sampler, attr_name, None)
        if value:
            final_sampler_path = str(value)
            break
    return last_metrics, periodic_eval_history, cached_final_eval, final_sampler_path


# ===== Task-specific helpers =====


def restore_latex_escapes(text: str) -> str:
    return (
        text.replace("\f", r"\f")
        .replace("\t", r"\t")
        .replace("\b", r"\b")
        .replace("\r", r"\r")
        .replace("\v", r"\v")
    )


def strip_generation_tokens(text: str) -> str:
    return STOP_TOKEN_RE.sub("", text).strip()


def normalize_answer(answer: str | None) -> str | None:
    if answer is None:
        return None
    answer = str(answer).strip()
    if not answer:
        return answer
    final_answer = answer.split("=")[-1]
    for before, after in VERL_NORMALIZE_SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in VERL_NORMALIZE_REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")
    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")
    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")
    return final_answer.strip()


@dataclass(frozen=True)
class ParsedFinalAnswer:
    extracted: str | None
    format_valid: bool


def extract_answer_line(text: str) -> str:
    search_text = text[-ANSWER_PARSE_TAIL_CHARS:]
    matches = ANSWER_LINE_RE.findall(search_text)
    if not matches:
        raise ValueError("No Answer: line found")
    return matches[-1].strip()


def parse_final_answer(text: str) -> ParsedFinalAnswer:
    cleaned = strip_generation_tokens(text)
    if not cleaned:
        return ParsedFinalAnswer(extracted=None, format_valid=False)
    try:
        return ParsedFinalAnswer(
            extracted=extract_answer_line(cleaned), format_valid=True
        )
    except ValueError:
        return ParsedFinalAnswer(extracted=None, format_valid=False)


def summarize_extracted_answers(items: list[str | None], limit: int = 6) -> str:
    counts: dict[str | None, int] = {}
    ordered: list[str | None] = []
    for item in items:
        if item not in counts:
            counts[item] = 0
            ordered.append(item)
        counts[item] += 1

    parts: list[str] = []
    for item in ordered[:limit]:
        parts.append(f"{preview_text(item, 36)} x{counts[item]}")
    if len(ordered) > limit:
        parts.append(f"... +{len(ordered) - limit} more")
    return " | ".join(parts) if parts else "<none>"


def _split_prompt_blocks(text: str) -> list[str]:
    return [block.strip() for block in text.split("\n\n") if block.strip()]


def _strip_known_math_answer_directives(text: str) -> str:
    text = restore_latex_escapes(text).strip()
    blocks = _split_prompt_blocks(text)
    if blocks and blocks[0] == VERL_DAPO_PROMPT_INTRO:
        blocks = blocks[1:]
    blocks = [
        block
        for block in blocks
        if block
        not in {
            VERL_DAPO_PROMPT_REMINDER,
            'Remember to put only the final answer on the last line after "Answer:"',
        }
    ]
    cleaned = "\n\n".join(blocks).strip()
    answer_line_marker = "and repeat it on its own last line as Answer:"
    if answer_line_marker in cleaned:
        cleaned = cleaned.split(answer_line_marker, 1)[0].rstrip()
    return cleaned.rstrip()


def build_math_rl_prompt(question: str) -> str:
    question = _strip_known_math_answer_directives(question)
    if not question:
        raise ValueError("Question is empty after prompt normalization")
    return f"{VERL_DAPO_PROMPT_INTRO}\n\n{question}\n\n{VERL_DAPO_PROMPT_REMINDER}"


def build_math_rl_messages(prompt: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": prompt}]


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


def make_math_record(
    *,
    row_id: str,
    question: str,
    answer: str,
    source: str,
) -> dict[str, Any]:
    clean_question = _strip_known_math_answer_directives(
        restore_latex_escapes(question).strip()
    )
    prompt = build_math_rl_prompt(clean_question)
    return {
        "id": row_id,
        "question": clean_question,
        "prompt": prompt,
        "messages": build_math_rl_messages(prompt),
        "answer": answer,
        "source": source,
    }


def grade_math_answer(given_answer: str | None, ground_truth: str) -> bool:
    if given_answer is None:
        return False
    gold_normalized = normalize_answer(ground_truth)
    given_normalized = normalize_answer(given_answer)
    if gold_normalized is None or given_normalized is None:
        return False
    return given_normalized == gold_normalized


def choose_majority_answer(extracted_answers: list[str | None]) -> str | None:
    vote_counts: dict[str, int] = {}
    first_raw_answer_by_normalized: dict[str, str] = {}
    for raw_answer in extracted_answers:
        normalized_answer = normalize_answer(raw_answer)
        if normalized_answer is None:
            continue
        vote_counts[normalized_answer] = vote_counts.get(normalized_answer, 0) + 1
        if (
            normalized_answer not in first_raw_answer_by_normalized
            and raw_answer is not None
        ):
            first_raw_answer_by_normalized[normalized_answer] = raw_answer
    if not vote_counts:
        return None
    winning_normalized_answer = max(vote_counts, key=vote_counts.get)
    return first_raw_answer_by_normalized[winning_normalized_answer]


def summarize_eval_answers(
    extracted_answers: list[str | None],
    gold_answer: str,
) -> dict[str, Any]:
    trajectory_correct = [
        grade_math_answer(item, gold_answer) for item in extracted_answers
    ]
    pass_at_k = any(trajectory_correct)
    greedy_answer = extracted_answers[0] if extracted_answers else None
    greedy_correct = grade_math_answer(greedy_answer, gold_answer)
    majority_answer = choose_majority_answer(extracted_answers)
    majority_correct = grade_math_answer(majority_answer, gold_answer)
    return {
        "trajectory_correct": trajectory_correct,
        "pass_at_k": pass_at_k,
        "greedy_answer": greedy_answer,
        "greedy_correct": greedy_correct,
        "majority_answer": majority_answer,
        "majority_correct": majority_correct,
    }


def reward_from_response(
    response: str, gold_answer: str
) -> tuple[float, str | None, bool, bool]:
    parsed_final = parse_final_answer(response)
    correct = grade_math_answer(parsed_final.extracted, gold_answer)
    base_reward = 1.0 if correct else 0.0
    return base_reward, parsed_final.extracted, parsed_final.format_valid, correct


def compute_soft_overlong_penalty(
    response_length: int,
    *,
    max_tokens: int,
    buffer_len: int,
    penalty_factor: float,
) -> float:
    if (
        max_tokens <= 0
        or response_length <= 0
        or buffer_len <= 0
        or penalty_factor <= 0.0
    ):
        return 0.0
    effective_buffer_len = min(buffer_len, max_tokens)
    max_exceed_length = max_tokens - effective_buffer_len
    if response_length > max_tokens:
        return float(penalty_factor)
    if response_length <= max_exceed_length:
        return 0.0
    exceed_length = response_length - max_exceed_length
    return float(exceed_length / effective_buffer_len * penalty_factor)


def shape_sequence_reward(
    base_reward: float,
    response_length: int,
    *,
    max_tokens: int,
    buffer_len: int,
    penalty_factor: float,
) -> tuple[float, float, bool]:
    if max_tokens > 0 and response_length > max_tokens:
        # Match the SkyRL DAPO reference path: once a response goes past the
        # configured max length, treat it as zero-reward instead of continuing to
        # push it negative with additional shaping.
        return 0.0, 0.0, True

    overlong_penalty = compute_soft_overlong_penalty(
        response_length,
        max_tokens=max_tokens,
        buffer_len=buffer_len,
        penalty_factor=penalty_factor,
    )
    return base_reward - overlong_penalty, overlong_penalty, False


def classify_overlong_response(
    response_tokens: list[int],
    *,
    max_tokens: int,
    eos_token_id: int | None,
    apply_filtering: bool,
    buffer_len: int,
    penalty_factor: float,
) -> tuple[bool, bool, float, bool]:
    response_length = len(response_tokens)
    hit_token_cap = max_tokens > 0 and response_length >= max_tokens
    finished_with_stop = (
        bool(response_tokens)
        and eos_token_id is not None
        and response_tokens[-1] == eos_token_id
    )
    overlong_filtered = apply_filtering and hit_token_cap and not finished_with_stop
    _shaped_reward, overlong_penalty, reward_zeroed_for_overmax = shape_sequence_reward(
        0.0,
        response_length,
        max_tokens=max_tokens,
        buffer_len=buffer_len,
        penalty_factor=penalty_factor,
    )
    return hit_token_cap, overlong_filtered, overlong_penalty, reward_zeroed_for_overmax


def score_response_tokens(
    tokenizer: Any,
    response_tokens: list[int],
    gold_answer: str,
    args: argparse.Namespace,
) -> tuple[str, float, float, str | None, bool, bool, float, bool, bool, bool]:
    response = tokenizer.decode(response_tokens).strip()
    base_reward, extracted, format_valid, correct = reward_from_response(
        response, gold_answer
    )
    hit_token_cap, overlong_filtered, overlong_penalty, reward_zeroed_for_overmax = (
        classify_overlong_response(
            response_tokens,
            max_tokens=args.rl_max_tokens,
            eos_token_id=tokenizer.eos_token_id,
            apply_filtering=args.apply_overlong_filtering,
            buffer_len=args.overlong_buffer_len,
            penalty_factor=args.overlong_buffer_penalty_factor,
        )
    )
    reward, _effective_penalty, _reward_zeroed = shape_sequence_reward(
        base_reward,
        len(response_tokens),
        max_tokens=args.rl_max_tokens,
        buffer_len=args.overlong_buffer_len,
        penalty_factor=args.overlong_buffer_penalty_factor,
    )
    return (
        response,
        reward,
        base_reward,
        extracted,
        format_valid,
        correct,
        overlong_penalty,
        hit_token_cap,
        overlong_filtered,
        reward_zeroed_for_overmax,
    )


def group_lacks_reward_signal(
    tokenizer: Any,
    token_groups: list[list[int]],
    gold_answer: str,
    args: argparse.Namespace,
) -> bool:
    rewards: list[float] = []
    for response_tokens in token_groups:
        if not response_tokens:
            continue
        (
            _,
            reward,
            _base_reward,
            _extracted,
            _format_valid,
            _correct,
            _overlong_penalty,
            _hit_token_cap,
            _overlong_filtered,
            _reward_zeroed_for_overmax,
        ) = score_response_tokens(
            tokenizer,
            response_tokens,
            gold_answer,
            args,
        )
        rewards.append(reward)
    if len(rewards) < 2:
        return True
    mean_reward = sum(rewards) / len(rewards)
    return all(abs(reward - mean_reward) < 1e-9 for reward in rewards)


# ===== Task-specific adapters =====


def normalize_train_row(row: dict[str, Any], index: int) -> dict[str, Any]:
    if "question" in row and "answer" in row:
        return make_math_record(
            row_id=extract_row_id(row, fallback=f"row-{index}"),
            question=str(row["question"]),
            answer=str(row["answer"]),
            source=str(row.get("source", "unknown")),
        )

    raise KeyError(
        f"row {index} has unsupported training schema; expected question/answer JSONL"
    )


def normalize_eval_row(row: dict[str, Any], index: int) -> dict[str, Any]:
    if "Problem" in row and "Answer" in row:
        return make_math_record(
            row_id=extract_row_id(row, fallback=f"row-{index}"),
            question=str(row["Problem"]),
            answer=str(row["Answer"]),
            source=str(row.get("source", "Maxwell-Jia/AIME_2024")),
        )

    raise KeyError(
        f"row {index} has unsupported evaluation schema; expected Problem/Answer JSONL"
    )


def normalize_train_rows(path: Path) -> list[dict[str, Any]]:
    rows = [
        normalize_train_row(row, index)
        for index, row in enumerate(load_rows(path), start=1)
    ]
    if not rows:
        raise RuntimeError("Training split is empty")
    return rows


def normalize_eval_rows(path: Path) -> list[dict[str, Any]]:
    rows = [
        normalize_eval_row(row, index)
        for index, row in enumerate(load_rows(path), start=1)
    ]
    if not rows:
        raise RuntimeError("Evaluation split is empty")
    return rows


# ===== Runtime entrypoints =====


async def evaluate_with_sampler(
    sampler: Any,
    tokenizer: Any,
    rows: list[dict[str, str]],
    args: argparse.Namespace,
    output_dir: Path,
    *,
    log_label: str = "eval",
    verbose_item_logs: bool = True,
) -> dict[str, float]:
    output_dir.mkdir(parents=True, exist_ok=True)
    examples_path = output_dir / "examples.jsonl"
    predictions_path = output_dir / "predictions.jsonl"
    failures_log_path = output_dir / "failures.log"
    failures_jsonl_path = output_dir / "failures.jsonl"
    eval_metrics_path = output_dir / "metrics.json"
    write_jsonl(
        examples_path,
        [eval_example_artifact_row(row, index=index) for index, row in enumerate(rows, start=1)],
    )
    print(f"@@ {log_label}_examples={os.path.relpath(examples_path, Path.cwd())}")
    print(f"@@ {log_label}_predictions={os.path.relpath(predictions_path, Path.cwd())}")
    print(
        f"@@ {log_label}_failures_log={os.path.relpath(failures_log_path, Path.cwd())}"
    )
    print(
        f"@@ {log_label}_failures_jsonl={os.path.relpath(failures_jsonl_path, Path.cwd())}"
    )
    print(
        f">> {log_label}_config max_concurrent_requests={args.max_concurrent_requests} "
        f"num_samples={args.eval_num_samples} temperature={args.eval_temperature} top_p={args.eval_top_p}",
        flush=True,
    )
    if not rows:
        write_jsonl(examples_path, [])
        predictions_path.write_text("", encoding="utf-8")
        failures_log_path.write_text("", encoding="utf-8")
        failures_jsonl_path.write_text("", encoding="utf-8")
        empty_metrics = {
            "eval_accuracy": 0.0,
            "eval_majority_accuracy": 0.0,
            "eval_greedy_accuracy": 0.0,
            "eval_pass_at_k": 0.0,
            "final_answer_format_success": 0.0,
            "avg_completion_tokens": 0.0,
        }
        eval_metrics_path.write_text(
            json.dumps(empty_metrics, indent=2) + "\n", encoding="utf-8"
        )
        return empty_metrics

    exact = 0
    majority_exact = 0
    greedy_exact = 0
    pass_at_k = 0
    formatted = 0
    completion_tokens = 0
    completed_count = 0
    eval_started_at = time.time()
    total_rows = len(rows)

    with predictions_path.open(
        "w", encoding="utf-8", buffering=1
    ) as predictions_handle, failures_log_path.open(
        "w", encoding="utf-8", buffering=1
    ) as failures_handle, failures_jsonl_path.open(
        "w", encoding="utf-8", buffering=1
    ) as failures_jsonl_handle:
        max_concurrent_requests = max(args.max_concurrent_requests, 1)
        eval_requests: list[tuple[int, dict[str, str], list[int], Any]] = []
        records_by_request_index: list[dict[str, Any] | None] = [None] * total_rows
        for index, row in enumerate(rows, start=1):
            eval_requests.append(
                (
                    index,
                    row,
                    build_generation_prompt_tokens(tokenizer, row["messages"]),
                    types.SamplingParams(
                        max_tokens=args.eval_max_tokens,
                        temperature=args.eval_temperature,
                        top_p=args.eval_top_p,
                        seed=args.seed + index,
                        stop_token_ids=[tokenizer.eos_token_id],
                    ),
                )
            )

        print(
            f">> {log_label}_pool submit=01-{total_rows:02d} size={len(eval_requests)} done=0/{total_rows} "
            f"elapsed={format_elapsed(time.time() - eval_started_at)}",
            flush=True,
        )

        def process_eval_result(request_index: int, result: Any) -> None:
            nonlocal completed_count, completion_tokens, exact, majority_exact, greedy_exact, formatted, pass_at_k
            index, row, _prompt_tokens, _sampling_params = eval_requests[request_index]
            token_groups, _ = extract_trajectory_tokens_and_logprobs(result)
            responses = [
                tokenizer.decode(response_tokens).strip()
                for response_tokens in token_groups
            ]
            parsed_finals = [parse_final_answer(response) for response in responses]
            extracted = [item.extracted for item in parsed_finals]
            structured = [item.format_valid for item in parsed_finals]
            gold = normalize_answer(row["answer"])

            completion_tokens += sum(
                len(response_tokens) for response_tokens in token_groups
            )
            format_count = sum(1 for item in structured if item)
            formatted += format_count
            eval_summary = summarize_eval_answers(extracted, row["answer"])
            trajectory_correct = eval_summary["trajectory_correct"]
            if eval_summary["pass_at_k"]:
                pass_at_k += 1
            if eval_summary["greedy_correct"]:
                greedy_exact += 1
            chosen = (
                eval_summary["greedy_answer"]
                if args.eval_num_samples == 1
                else eval_summary["majority_answer"]
            )
            chosen_correct = (
                eval_summary["greedy_correct"]
                if args.eval_num_samples == 1
                else eval_summary["majority_correct"]
            )
            if chosen_correct:
                exact += 1
            if eval_summary["majority_correct"]:
                majority_exact += 1

            status = "PASS" if chosen_correct else "FAIL"
            line_prefix = "++" if chosen_correct else "!!"
            record = {
                "index": index,
                "id": row["id"],
                "question": row["question"],
                "gold_answer": gold,
                "chosen_answer": chosen,
                "chosen_correct": chosen_correct,
                "greedy_answer": eval_summary["greedy_answer"],
                "greedy_correct": eval_summary["greedy_correct"],
                "majority_answer": eval_summary["majority_answer"],
                "majority_correct": eval_summary["majority_correct"],
                "status": status,
                "format_count": format_count,
                "trajectory_answers": extracted,
                "trajectory_correct": trajectory_correct,
                "trajectory_format_valid": structured,
                "trajectory_text": responses,
            }
            records_by_request_index[request_index] = record
            completed_count += 1

            elapsed_now = time.time() - eval_started_at
            average_seconds_per_problem = elapsed_now / max(completed_count, 1)
            eta_seconds = average_seconds_per_problem * max(
                total_rows - completed_count, 0
            )
            running_accuracy = exact / max(completed_count, 1)
            running_greedy_accuracy = greedy_exact / max(completed_count, 1)
            running_pass_at_k = pass_at_k / max(completed_count, 1)
            if verbose_item_logs:
                print(
                    f"{line_prefix} {log_label}[{index:02d}] {status} done={completed_count}/{total_rows} elapsed={format_elapsed(elapsed_now)} "
                    f"eta={format_elapsed(eta_seconds)} running_acc={running_accuracy:.4f} "
                    f"running_greedy_acc={running_greedy_accuracy:.4f} running_pass_at_k={running_pass_at_k:.4f} "
                    f"gold={preview_text(gold, 24)} chosen={preview_text(chosen, 24)} "
                    f"format={format_count}/{len(structured)}",
                    flush=True,
                )
                if not chosen_correct:
                    print(f"!!   q: {preview_text(row['question'], 96)}", flush=True)
                    print(
                        f"!!   trajectories: {summarize_extracted_answers(extracted)}",
                        flush=True,
                    )

        pool_started_at = time.time()
        await sample_many_with_semaphore_async(
            sampler=sampler,
            sampling_requests=[
                (prompt_tokens, args.eval_num_samples, sampling_params)
                for _, _, prompt_tokens, sampling_params in eval_requests
            ],
            max_concurrent_requests=max_concurrent_requests,
            on_result=process_eval_result,
        )
        pool_wait_seconds = time.time() - pool_started_at

        for record in records_by_request_index:
            if record is None:
                raise RuntimeError("evaluate completed without a record")
            append_jsonl_record(predictions_handle, record)
            if not record["chosen_correct"]:
                append_jsonl_record(failures_jsonl_handle, record)
                write_eval_log_record(failures_handle, record, record["chosen_correct"])

        print(
            f"== {log_label}_pool complete=01-{total_rows:02d} wait={pool_wait_seconds:.1f}s "
            f"elapsed={format_elapsed(time.time() - eval_started_at)}",
            flush=True,
        )

    metrics = {
        "eval_accuracy": exact / len(rows),
        "eval_majority_accuracy": majority_exact / len(rows),
        "eval_greedy_accuracy": greedy_exact / len(rows),
        "eval_pass_at_k": pass_at_k / len(rows),
        "final_answer_format_success": formatted
        / max(len(rows) * args.eval_num_samples, 1),
        "avg_completion_tokens": completion_tokens
        / max(len(rows) * args.eval_num_samples, 1),
    }
    eval_metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics


def run_dry_run(
    train_data_path: Path, eval_rows: list[dict[str, str]], args: argparse.Namespace
) -> None:
    train_status = "missing"
    train_rows = 0
    example_prompt = "<train data missing>"
    if train_data_path.exists():
        loaded_rows = normalize_train_rows(train_data_path)
        train_status = "present"
        train_rows = len(loaded_rows)
        example_prompt = preview_messages(loaded_rows[0]["messages"])
    print(
        "dry_run: "
        f"train_data={os.path.relpath(train_data_path, Path.cwd())} "
        f"train_status={train_status} train_rows={train_rows} "
        f"eval_rows={len(eval_rows)} grpo_steps={args.grpo_steps} group_size={args.group_size}"
    )
    print("dry_run: prompt_preview_start")
    print(example_prompt)
    print("dry_run: prompt_preview_end")
    print("dry_run: eval_prompt_preview_start")
    print(preview_messages(eval_rows[0]["messages"]) if eval_rows else "<no eval rows>")
    print("dry_run: eval_prompt_preview_end")


# ===== Artifact writing =====


def eval_example_artifact_row(row: dict[str, Any], *, index: int) -> dict[str, Any]:
    return {
        "index": index,
        "id": row["id"],
        "question": row["question"],
        "gold_answer": normalize_answer(row["answer"]),
        "prompt": row["prompt"],
        "messages": row["messages"],
        "source": row["source"],
    }


def reset_rl_append_streams(
    run_dir: Path,
    *,
    resume_checkpoint: dict[str, Any] | None = None,
) -> None:
    if resume_checkpoint is not None:
        return
    for rel_path in (
        Path("train/metrics.jsonl"),
        Path("train/checkpoints.jsonl"),
        Path("train/rollouts.jsonl"),
        Path("train/failures.log"),
        Path("train/failures.jsonl"),
        Path("eval/metrics.jsonl"),
    ):
        path = run_dir / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
    remove_path(run_dir / "train" / "state_path.txt")
    remove_path(run_dir / "eval" / "steps")


def reset_eval_output_artifacts(run_dir: Path) -> None:
    """Remove final-eval files before a fresh run writes new outputs."""
    for rel_path in (
        Path("eval/examples.jsonl"),
        Path("eval/predictions.jsonl"),
        Path("eval/metrics.json"),
        Path("eval/failures.log"),
        Path("eval/failures.jsonl"),
    ):
        remove_path(run_dir / rel_path)


def append_periodic_eval_record(
    path: Path,
    *,
    step: int,
    total_steps: int,
    eval_output_dir: Path,
    status: str,
    sampler_path: str | None = None,
    eval_metrics: dict[str, float] | None = None,
    error: str | None = None,
) -> None:
    record: dict[str, Any] = {
        "step": step,
        "total_steps": total_steps,
        "status": status,
        "output_dir": os.path.relpath(eval_output_dir, Path.cwd()),
        "completed_at_unix": time.time(),
    }
    if sampler_path is not None:
        record["sampler_path"] = sampler_path
    if eval_metrics is not None:
        record.update(eval_metrics)
    if error is not None:
        record["error"] = error
    append_jsonl(path, record)


def format_eval_metric_summary(eval_metrics: dict[str, float]) -> str:
    accuracy = float(eval_metrics.get("eval_accuracy", 0.0))
    majority_accuracy = float(eval_metrics.get("eval_majority_accuracy", accuracy))
    greedy_accuracy = float(eval_metrics.get("eval_greedy_accuracy", 0.0))
    pass_at_k = float(eval_metrics.get("eval_pass_at_k", 0.0))
    format_success = float(eval_metrics.get("final_answer_format_success", 0.0))
    avg_completion_tokens = float(eval_metrics.get("avg_completion_tokens", 0.0))
    return (
        f"eval_accuracy={accuracy:.4f} "
        f"eval_majority_accuracy={majority_accuracy:.4f} "
        f"eval_greedy_accuracy={greedy_accuracy:.4f} "
        f"eval_pass_at_k={pass_at_k:.4f} "
        f"format_success={format_success:.4f} "
        f"avg_completion_tokens={avg_completion_tokens:.1f}"
    )


def append_jsonl_record(handle: Any, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=True) + "\n")
    handle.flush()


def format_log_block(text: str | None, indent: str = "    ") -> str:
    if text is None:
        return indent + "<none>"
    stripped = text.rstrip()
    if not stripped:
        return indent + "<empty>"
    return "\n".join(
        f"{indent}{line}" if line else indent for line in stripped.splitlines()
    )


def write_eval_log_record(
    handle: Any, record: dict[str, Any], chosen_correct: bool
) -> None:
    format_count = sum(1 for item in record["trajectory_format_valid"] if item)
    status = "PASS" if chosen_correct else "FAIL"
    handle.write(
        f"eval[{record['index']:02d}] {status} gold={preview_text(record['gold_answer'], 24)} "
        f"chosen={preview_text(record['chosen_answer'], 24)} "
        f"format={format_count}/{len(record['trajectory_format_valid'])}\n"
    )
    handle.write(f"  q: {record['question']}\n")
    handle.write(
        f"  trajectories: {summarize_extracted_answers(record['trajectory_answers'])}\n"
    )
    for trajectory_index, (answer, correct, format_valid, raw_text) in enumerate(
        zip(
            record["trajectory_answers"],
            record["trajectory_correct"],
            record["trajectory_format_valid"],
            record["trajectory_text"],
            strict=True,
        ),
        start=1,
    ):
        handle.write(
            f"  trajectory[{trajectory_index:02d}] extracted={preview_text(answer, 36)} "
            f"correct={int(correct)} format={int(format_valid)} chars={len(raw_text)}\n"
        )
        handle.write(format_log_block(raw_text) + "\n")
    handle.write("\n")
    handle.flush()


def is_train_rollout_failure(record: dict[str, Any]) -> bool:
    trajectory_correct = record.get("trajectory_correct") or []
    return not trajectory_correct or not bool(trajectory_correct[0])


def write_train_failure_record(handle: Any, record: dict[str, Any]) -> None:
    trajectory_correct = record.get("trajectory_correct") or []
    trajectory_used_for_update = record.get("trajectory_used_for_update") or []
    trajectory_reward_zeroed_for_overmax = (
        record.get("trajectory_reward_zeroed_for_overmax") or []
    )
    first_correct = bool(trajectory_correct[0]) if trajectory_correct else False
    any_correct = int(record.get("num_correct", 0)) > 0
    first_used_for_update = (
        bool(trajectory_used_for_update[0]) if trajectory_used_for_update else False
    )
    overmax_zeroed_count = sum(
        1 for item in trajectory_reward_zeroed_for_overmax if item
    )
    handle.write(
        f"train[step={int(record['step']):04d} group={int(record['row_index']):02d}] FAIL "
        f"first_correct={int(first_correct)} any_correct={int(any_correct)} "
        f"first_used={int(first_used_for_update)} used={int(record['num_datums'])}/{int(record['num_trajectories'])} "
        f"status={record['status']} gold={preview_text(record['gold_answer'], 24)} "
        f"correct={int(record['num_correct'])}/{int(record['num_trajectories'])} "
        f"format={int(record['num_formatted'])}/{int(record['num_trajectories'])} "
        f"full_logprobs={int(record.get('num_valid_logprob_trajectories', 0))}/{int(record['num_trajectories'])} "
        f"overmax_zeroed={overmax_zeroed_count}/{int(record['num_trajectories'])} "
        f"reward_mean={float(record['reward_mean']):.4f}\n"
    )
    handle.write(f"  q: {record['question']}\n")
    handle.write(
        f"  trajectories: {summarize_extracted_answers(record['trajectory_answers'])}\n"
    )

    if record["trajectory_answers"]:
        answer = record["trajectory_answers"][0]
        correct = record["trajectory_correct"][0]
        format_valid = record["trajectory_format_valid"][0]
        base_reward = record.get("trajectory_base_rewards", [0.0])[0]
        reward = record["trajectory_rewards"][0]
        advantage = record["trajectory_advantages"][0]
        token_count = record["trajectory_token_counts"][0]
        overlong_penalty = record.get("trajectory_overlong_penalties", [0.0])[0]
        reward_zeroed_for_overmax = bool(
            record.get("trajectory_reward_zeroed_for_overmax", [False])[0]
        )
        hit_token_cap = bool(record.get("trajectory_hit_token_cap", [False])[0])
        overlong_filtered = bool(record.get("trajectory_overlong_filtered", [False])[0])
        first_trajectory_text = record.get("first_trajectory_text")
        handle.write(
            f"  trajectory[01] extracted={preview_text(answer, 36)} "
            f"correct={int(correct)} format={int(format_valid)} base_reward={float(base_reward):.4f} "
            f"reward={float(reward):.4f} penalty={float(overlong_penalty):.4f} "
            f"zeroed_overmax={int(reward_zeroed_for_overmax)} hit_cap={int(hit_token_cap)} filtered={int(overlong_filtered)} "
            f"adv={float(advantage):.4f} tokens={int(token_count)}\n"
        )
        handle.write(format_log_block(first_trajectory_text) + "\n")
    else:
        handle.write("  trajectory[01] <none>\n")

    handle.write("\n")
    handle.flush()


def collect_run_artifacts(
    output_dir: Path,
    *,
    include_append_streams: bool,
) -> dict[str, dict[str, str]]:
    """Build the current run.json artifact index from files on disk."""
    artifacts: dict[str, dict[str, str]] = {}
    eval_artifacts: dict[str, str] = {}
    train_artifacts: dict[str, str] = {}
    eval_dir = output_dir / "eval"
    train_dir = output_dir / "train"

    for artifact_name, artifact_path in (
        ("examples", eval_dir / "examples.jsonl"),
        ("predictions", eval_dir / "predictions.jsonl"),
        ("metrics", eval_dir / "metrics.json"),
        ("failures_log", eval_dir / "failures.log"),
        ("failures", eval_dir / "failures.jsonl"),
    ):
        if artifact_path.is_file() and artifact_path.stat().st_size > 0:
            eval_artifacts[artifact_name] = str(artifact_path)

    if include_append_streams:
        metrics_history_path = eval_dir / "metrics.jsonl"
        if metrics_history_path.is_file() and metrics_history_path.stat().st_size > 0:
            eval_artifacts["metrics_history"] = str(metrics_history_path)
        for artifact_name, artifact_path in (
            ("metrics", train_dir / "metrics.jsonl"),
            ("checkpoints", train_dir / "checkpoints.jsonl"),
            ("rollouts", train_dir / "rollouts.jsonl"),
            ("failures_log", train_dir / "failures.log"),
            ("failures", train_dir / "failures.jsonl"),
            ("state_path", train_dir / "state_path.txt"),
        ):
            if artifact_path.is_file() and artifact_path.stat().st_size > 0:
                train_artifacts[artifact_name] = str(artifact_path)

    if eval_artifacts:
        artifacts["eval"] = eval_artifacts
    if train_artifacts:
        artifacts["train"] = train_artifacts
    return artifacts


def write_run_metadata(
    output_dir: Path,
    args: argparse.Namespace,
    *,
    status: str,
    started_at: float,
    ended_at: float | None = None,
    error: str | None = None,
    include_append_streams: bool = True,
) -> None:
    run_json_path = output_dir / "run.json"
    if run_json_path.exists():
        payload = json.loads(run_json_path.read_text(encoding="utf-8"))
    else:
        payload = {}
    payload.update({
        "experiment": EXPERIMENT_DIR.name,
        "argv": list(sys.argv),
        "status": status,
        "command": shlex.join(sys.argv),
        "started_at_unix": started_at,
        "ended_at_unix": ended_at,
        "duration_seconds": round(ended_at - started_at, 1) if ended_at is not None else None,
        "args": vars(args),
        "env": {k: os.environ.get(k) for k in ("MINT_BASE_URL",) if os.environ.get(k)},
        "error": error,
    })
    artifacts = collect_run_artifacts(
        output_dir,
        include_append_streams=include_append_streams,
    )
    if artifacts:
        payload["artifacts"] = artifacts
    else:
        payload.pop("artifacts", None)
    payload.update(optional_git_provenance(EXPERIMENT_DIR))
    write_json(run_json_path, payload)


def write_outputs(
    output_dir: Path,
    *,
    args: argparse.Namespace,
    rl_metrics: dict[str, float],
    eval_metrics: dict[str, float],
    state_path: str | None,
    periodic_eval_history: list[dict[str, Any]],
    include_existing_artifacts: bool = True,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    train_dir = output_dir / "train"
    state_path_file = train_dir / "state_path.txt"
    if state_path:
        state_path_file.parent.mkdir(parents=True, exist_ok=True)
        state_path_file.write_text(state_path + "\n", encoding="utf-8")
    elif state_path_file.exists():
        state_path_file.unlink()

    run_json_path = output_dir / "run.json"
    if run_json_path.exists():
        run_payload = json.loads(run_json_path.read_text(encoding="utf-8"))
    else:
        run_payload = {
            "experiment": EXPERIMENT_DIR.name,
            "argv": list(sys.argv),
            "command": shlex.join(sys.argv),
            "args": vars(args),
        }
    artifacts = collect_run_artifacts(
        output_dir,
        include_append_streams=include_existing_artifacts,
    )
    if artifacts:
        run_payload["artifacts"] = artifacts
    else:
        run_payload.pop("artifacts", None)
    run_payload["rl_metrics"] = rl_metrics
    run_payload["eval_metrics"] = eval_metrics
    run_payload["state_path"] = state_path
    run_payload["periodic_eval_history"] = periodic_eval_history
    run_payload.update(optional_git_provenance(EXPERIMENT_DIR))
    write_json(run_json_path, run_payload)


# ===== Entry point =====


async def main_async() -> int:
    args = parse_args()
    train_path = Path(args.train_data)
    eval_path = Path(args.eval_data)
    load_checkpoint_path = str(getattr(args, "load_checkpoint_path", "") or "").strip()

    if args.dry_run:
        if load_checkpoint_path:
            raise RuntimeError("--load-checkpoint-path is ignored under --dry-run")
        eval_rows = normalize_eval_rows(eval_path)
        run_dry_run(train_path, eval_rows, args)
        return 0

    if args.eval_only and load_checkpoint_path:
        raise RuntimeError(
            "--load-checkpoint-path is training-only; "
            "use --base-model with a recorded sampler_path for --eval-only"
        )

    log_path = Path(args.log_path)
    run_dir = prepare_run_dir(log_path)

    resume_checkpoint_row: dict[str, Any] | None = None
    if not args.eval_only:
        resume_checkpoint_row = get_last_resumable_checkpoint(run_dir)
        validate_resume_contract(run_dir, args, resume_checkpoint=resume_checkpoint_row)

    resume_state_path = (
        str(resume_checkpoint_row.get("state_path") or "").strip()
        if resume_checkpoint_row
        else ""
    )
    load_weights_path = (
        ""
        if resume_state_path
        else load_checkpoint_path
    )

    resume_loop_state: ResumeLoopState | None = None
    if resume_checkpoint_row is not None:
        resume_loop_state = resume_loop_state_from_checkpoint_row(
            resume_checkpoint_row, run_dir, args
        )

    args.resume_completed_steps = (
        resume_loop_state.completed_steps if resume_loop_state is not None else 0
    )

    metadata_includes_append_streams = not (args.eval_only or args.dry_run)
    if resume_checkpoint_row is None:
        reset_eval_output_artifacts(run_dir)
    if not args.eval_only:
        reset_rl_append_streams(run_dir, resume_checkpoint=resume_checkpoint_row)

    started_at = time.time()
    console_log_mode = "a" if resume_checkpoint_row is not None else "w"
    console_log_handle = (run_dir / "console.log").open(
        console_log_mode,
        encoding="utf-8",
        buffering=1,
    )
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = TeeStream(original_stdout, console_log_handle)
    sys.stderr = TeeStream(original_stderr, console_log_handle)
    write_run_metadata(
        run_dir,
        args,
        status="running",
        started_at=started_at,
        include_append_streams=metadata_includes_append_streams,
    )
    print(f"@@ artifacts_run={os.path.relpath(run_dir, Path.cwd())}")

    try:
        eval_rows = normalize_eval_rows(eval_path)

        api_key = (os.environ.get("MINT_API_KEY") or "").strip()
        if not api_key:
            raise RuntimeError(
                f"missing MINT_API_KEY in {EXPERIMENT_DIR / '.env'} or shell environment"
            )

        train_rows: list[dict[str, str]] = []
        if not args.eval_only:
            if not train_path.exists():
                raise RuntimeError(f"Training data not found at {train_path}")
            train_rows = normalize_train_rows(train_path)

        service_client = create_service_client(timeout=args.mint_timeout)
        capabilities = await preflight_connection_async(service_client)
        supported_models = getattr(capabilities, "supported_models", None)
        if isinstance(supported_models, list):
            print(f"OK mint_server supported_models={len(supported_models)}")
        else:
            print("OK mint_server")

        training_client: Any | None = None
        eval_sampler: Any | None = None
        if args.eval_only:
            eval_sampler = await create_sampling_client(service_client, args.base_model)
            tokenizer = get_tokenizer(eval_sampler, args.base_model)
            training_client = None
        else:
            training_client = await provision_training_client(
                service_client,
                args,
                resume_state_path=resume_state_path or None,
                load_checkpoint_weights_path=load_weights_path or None,
            )
            tokenizer = get_tokenizer(training_client, args.base_model)
        print(f"@@ model={args.base_model} vocab={tokenizer.vocab_size:,}")

        rl_metrics: dict[str, float] = {}
        periodic_eval_history: list[dict[str, Any]] = []
        cached_final_eval: dict[str, float] | None = None
        final_sampler_path: str | None = None
        if not args.eval_only:
            rl_metrics, periodic_eval_history, cached_final_eval, final_sampler_path = (
                await grpo_train_loop(
                    service_client,
                    training_client,
                    tokenizer,
                    train_rows,
                    eval_rows,
                    args,
                    run_dir,
                    resume_loop_state=resume_loop_state,
                )
            )
            emit_metric_lines(rl_metrics)

        state_path: str | None = None
        if args.save_state_name:
            if training_client is None:
                raise RuntimeError(
                    "--save-state-name requires a training client; it is not supported with base-model-only --eval-only"
                )
            state_path = await save_training_state(
                training_client, args.save_state_name
            )
            final_sampler_path = await save_sampler_checkpoint(
                training_client, args.save_state_name
            )
            if not state_path or not final_sampler_path:
                raise RuntimeError(
                    "--save-state-name requires both save_state and save_weights_for_sampler support"
                )
            append_jsonl(
                run_dir / "train" / "checkpoints.jsonl",
                {
                    "name": args.save_state_name,
                    "step": args.grpo_steps,
                    "completed_steps": args.grpo_steps,
                    "next_step": args.grpo_steps + 1,
                    "batch": args.grpo_steps,
                    "epoch": 0,
                    "state_path": state_path,
                    "sampler_path": final_sampler_path,
                },
            )
            print(f"@@ saved_state={state_path} sampler_path={final_sampler_path}")

        if cached_final_eval is not None:
            eval_metrics = cached_final_eval
        else:
            if eval_sampler is None:
                if training_client is None:
                    raise RuntimeError("No evaluation sampler available")
                eval_sampler = await save_weights_for_sampling(training_client)
            eval_metrics = await evaluate_with_sampler(
                eval_sampler,
                tokenizer,
                eval_rows,
                args,
                run_dir / "eval",
                log_label="eval",
                verbose_item_logs=True,
            )
        write_outputs(
            run_dir,
            args=args,
            rl_metrics=rl_metrics,
            eval_metrics=eval_metrics,
            state_path=state_path,
            periodic_eval_history=periodic_eval_history,
            include_existing_artifacts=metadata_includes_append_streams,
        )
        emit_metric_lines(eval_metrics)

        ended_at = time.time()
        write_run_metadata(
            run_dir,
            args,
            status="completed",
            started_at=started_at,
            ended_at=ended_at,
            include_append_streams=metadata_includes_append_streams,
        )
        console_log_handle.flush()
        print(
            f"== run_complete artifacts_run={os.path.relpath(run_dir, Path.cwd())}",
            flush=True,
        )
        return 0
    except Exception as exc:
        ended_at = time.time()
        write_run_metadata(
            run_dir,
            args,
            status="failed",
            started_at=started_at,
            ended_at=ended_at,
            error=str(exc),
            include_append_streams=metadata_includes_append_streams,
        )
        console_log_handle.flush()
        raise
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        console_log_handle.close()


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    raise SystemExit(main())
