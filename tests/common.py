from __future__ import annotations

"""Shared MinT smoke helpers.

This file is a copy source for new experiment scripts: copy the pieces you need,
but do not import this file from experiment baselines.
"""

import argparse
import asyncio
import os
import signal
import time
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mint
from transformers import AutoTokenizer

TESTS_DIR = Path(__file__).resolve().parent
ENV_PATH = TESTS_DIR / ".env"
DEFAULT_HF_HOME = str(Path.home() / ".cache" / "huggingface")
DEFAULT_BASE_MODEL = "Qwen/Qwen3-0.6B"
DEFAULT_LORA_RANK = 4
DEFAULT_TIMEOUT_SECONDS = 600.0
DEFAULT_MODELS = [
    DEFAULT_BASE_MODEL,
    "Qwen/Qwen3-235B-A22B-Instruct-2507",
    "Qwen/Qwen3-4B-Instruct-2507",
    "Qwen/Qwen3-4B-Thinking-2507",
]

SMOKE_TIMEOUT_SECONDS = DEFAULT_TIMEOUT_SECONDS


@dataclass(frozen=True)
class SmokeConfig:
    base_model: str = DEFAULT_BASE_MODEL
    lora_rank: int = DEFAULT_LORA_RANK
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS


@dataclass
class SmokeContext:
    service_client: Any
    capabilities: Any
    model: str
    training_client: Any
    tokenizer: Any
    config: SmokeConfig


def load_env(path: Path = ENV_PATH) -> None:
    allowed_keys = {"MINT_BASE_URL", "MINT_API_KEY"}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.startswith("export "):
                stripped = stripped[len("export "):].lstrip()
            if "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key in allowed_keys and key not in os.environ:
                os.environ[key] = value

    os.environ.setdefault("HF_HOME", DEFAULT_HF_HOME)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def add_smoke_cli_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--base-model",
        default=DEFAULT_BASE_MODEL,
        help=f"Base model for smoke tests (default: {DEFAULT_BASE_MODEL}).",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=DEFAULT_LORA_RANK,
        help=f"LoRA rank for smoke tests (default: {DEFAULT_LORA_RANK}).",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"Per-step timeout in seconds (default: {DEFAULT_TIMEOUT_SECONDS:.0f}).",
    )


def smoke_config_from_args(args: argparse.Namespace) -> SmokeConfig:
    base_model = (getattr(args, "base_model", DEFAULT_BASE_MODEL) or DEFAULT_BASE_MODEL).strip()
    lora_rank = max(1, int(getattr(args, "lora_rank", DEFAULT_LORA_RANK)))
    timeout_seconds = normalize_timeout_seconds(getattr(args, "timeout_seconds", DEFAULT_TIMEOUT_SECONDS))
    return SmokeConfig(
        base_model=base_model,
        lora_rank=lora_rank,
        timeout_seconds=timeout_seconds,
    )


def normalize_timeout_seconds(timeout: float | int | str | None) -> float:
    try:
        seconds = float(timeout if timeout is not None else DEFAULT_TIMEOUT_SECONDS)
    except (TypeError, ValueError):
        seconds = DEFAULT_TIMEOUT_SECONDS
    return max(1.0, seconds)


def set_default_smoke_timeout(timeout_seconds: float) -> None:
    global SMOKE_TIMEOUT_SECONDS
    SMOKE_TIMEOUT_SECONDS = normalize_timeout_seconds(timeout_seconds)


def preflight_connection(service_client: Any) -> Any:
    base_url = os.environ.get("MINT_BASE_URL")
    try:
        return service_client.get_server_capabilities()
    except mint.APITimeoutError as exc:
        raise RuntimeError(f"Auth preflight timed out while contacting {base_url}") from exc
    except mint.APIConnectionError as exc:
        raise RuntimeError(f"Auth preflight could not reach {base_url}") from exc
    except mint.APIStatusError as exc:
        status_code = getattr(exc, "status_code", None)
        if not isinstance(status_code, int):
            response = getattr(exc, "response", None)
            response_status = getattr(response, "status_code", None)
            status_code = response_status if isinstance(response_status, int) else None
        if status_code in {401, 403}:
            raise RuntimeError(f"Auth preflight rejected the API key with HTTP {status_code}") from exc
        raise RuntimeError(f"Auth preflight failed with HTTP {status_code or 'unknown'} from {base_url}") from exc


def cached_tokenizer_dir(model_name: str) -> Path | None:
    hf_home = Path(os.environ.get("HF_HOME", DEFAULT_HF_HOME)).expanduser()
    hub_root = hf_home / "hub" if (hf_home / "hub").exists() else hf_home / ".cache" / "huggingface" / "hub"
    org, repo = model_name.split("/", 1)
    repo_dir = hub_root / f"models--{org.replace('/', '--')}--{repo.replace('/', '--')}"
    snapshots_dir = repo_dir / "snapshots"
    if not snapshots_dir.exists():
        return None

    def is_tokenizer_snapshot(snapshot: Path) -> bool:
        if not snapshot.is_dir():
            return False
        files = {path.name for path in snapshot.iterdir() if path.is_file()}
        if "tokenizer_config.json" not in files:
            return False
        if "tokenizer.json" in files:
            return True
        return {"vocab.json", "merges.txt"}.issubset(files)

    candidates = sorted(
        (snapshot for snapshot in snapshots_dir.iterdir() if is_tokenizer_snapshot(snapshot)),
        reverse=True,
    )
    return candidates[0] if candidates else None


def choose_model(capabilities: Any, override: str | None = None) -> str:
    if override:
        return override

    supported = [
        getattr(item, "model_name", str(item))
        for item in getattr(capabilities, "supported_models", [])
    ]
    cached_supported = [model for model in supported if cached_tokenizer_dir(model) is not None]

    for model in DEFAULT_MODELS:
        if model in cached_supported:
            return model
    for model in DEFAULT_MODELS:
        if model in supported:
            return model
    if cached_supported:
        return cached_supported[0]
    if supported:
        return supported[0]
    return DEFAULT_MODELS[0]


def get_tokenizer(training_client: Any, model_name: str) -> Any:
    cache_dir = cached_tokenizer_dir(model_name)
    if cache_dir is not None:
        print(f"tokenizer: using local Hugging Face cache for {model_name} from {cache_dir}")
        return AutoTokenizer.from_pretrained(str(cache_dir), fast=True, local_files_only=True)

    try:
        return training_client.get_tokenizer()
    except Exception as exc:
        raise RuntimeError(
            f"training_client.get_tokenizer() failed and no cached Hugging Face tokenizer was found for {model_name}: {exc}"
        ) from exc


def build_sft_datum(prompt: str, completion: str, tokenizer: Any, types_module: Any) -> Any:
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    completion_tokens = tokenizer.encode(completion, add_special_tokens=False)
    completion_tokens.append(tokenizer.eos_token_id)

    all_tokens = prompt_tokens + completion_tokens
    weights = [0.0] * len(prompt_tokens) + [1.0] * len(completion_tokens)

    return types_module.Datum(
        model_input=types_module.ModelInput.from_ints(tokens=all_tokens[:-1]),
        loss_fn_inputs={
            "target_tokens": all_tokens[1:],
            "weights": weights[1:],
        },
    )


@contextmanager
def _alarm_timeout(seconds: float, step: str):
    if os.name != "posix":
        yield
        return

    def raise_timeout(_signum, _frame):
        raise TimeoutError(step)

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


@contextmanager
def smoke_step(prefix: str, step: str, *, timeout: float | None = None):
    seconds = normalize_timeout_seconds(timeout if timeout is not None else SMOKE_TIMEOUT_SECONDS)
    start = time.time()
    print(f"{prefix}: [{time.strftime('%H:%M:%S')}] {step}: start", flush=True)
    try:
        with _alarm_timeout(seconds, step):
            yield
    except TimeoutError as exc:
        raise TimeoutError(
            f"{prefix}: {step} timed out after {seconds:.0f}s; this usually means the remote MinT/Tinker backend is queued or unreachable"
        ) from exc
    else:
        elapsed = time.time() - start
        print(f"{prefix}: [{time.strftime('%H:%M:%S')}] {step}: done in {elapsed:.1f}s", flush=True)


@asynccontextmanager
async def smoke_step_async(prefix: str, step: str, *, timeout: float | None = None):
    seconds = normalize_timeout_seconds(timeout if timeout is not None else SMOKE_TIMEOUT_SECONDS)
    start = time.time()
    print(f"{prefix}: [{time.strftime('%H:%M:%S')}] {step}: start", flush=True)
    try:
        async with asyncio.timeout(seconds):
            yield
    except TimeoutError as exc:
        raise TimeoutError(
            f"{prefix}: {step} timed out after {seconds:.0f}s; this usually means the remote MinT/Tinker backend is queued or unreachable"
        ) from exc
    else:
        elapsed = time.time() - start
        print(f"{prefix}: [{time.strftime('%H:%M:%S')}] {step}: done in {elapsed:.1f}s", flush=True)


def bootstrap_training_test(prefix: str, config: SmokeConfig | None = None) -> SmokeContext:
    load_env()
    config = config or SmokeConfig()
    set_default_smoke_timeout(config.timeout_seconds)
    api_key = (os.environ.get("MINT_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("Missing MINT_API_KEY in tests/.env or shell environment")
    os.environ.setdefault("MINT_API_KEY", api_key)

    with smoke_step(prefix, "create service client"):
        service_client = mint.ServiceClient(base_url=os.environ.get("MINT_BASE_URL"))
    with smoke_step(prefix, "fetch server capabilities"):
        capabilities = preflight_connection(service_client)

    model = choose_model(capabilities, config.base_model)
    print(f"{prefix}: model={model}")

    with smoke_step(prefix, f"create LoRA training client for {model}"):
        training_client = service_client.create_lora_training_client(
            base_model=model,
            rank=config.lora_rank,
            train_mlp=True,
            train_attn=True,
            train_unembed=True,
        )
    with smoke_step(prefix, f"load tokenizer for {model}"):
        tokenizer = get_tokenizer(training_client, model)

    return SmokeContext(
        service_client=service_client,
        capabilities=capabilities,
        model=model,
        training_client=training_client,
        tokenizer=tokenizer,
        config=config,
    )


async def bootstrap_training_test_async(prefix: str, config: SmokeConfig | None = None) -> SmokeContext:
    load_env()
    config = config or SmokeConfig()
    set_default_smoke_timeout(config.timeout_seconds)
    api_key = (os.environ.get("MINT_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("Missing MINT_API_KEY in tests/.env or shell environment")
    os.environ.setdefault("MINT_API_KEY", api_key)

    with smoke_step(prefix, "create service client"):
        service_client = mint.ServiceClient(base_url=os.environ.get("MINT_BASE_URL"))
    async with smoke_step_async(prefix, "fetch server capabilities"):
        base_url = os.environ.get("MINT_BASE_URL")
        try:
            capabilities = await service_client.get_server_capabilities_async()
        except mint.APITimeoutError as exc:
            raise RuntimeError(f"Auth preflight timed out while contacting {base_url}") from exc
        except mint.APIConnectionError as exc:
            raise RuntimeError(f"Auth preflight could not reach {base_url}") from exc
        except mint.APIStatusError as exc:
            status_code = getattr(exc, "status_code", None)
            if not isinstance(status_code, int):
                response = getattr(exc, "response", None)
                response_status = getattr(response, "status_code", None)
                status_code = response_status if isinstance(response_status, int) else None
            if status_code in {401, 403}:
                raise RuntimeError(f"Auth preflight rejected the API key with HTTP {status_code}") from exc
            raise RuntimeError(f"Auth preflight failed with HTTP {status_code or 'unknown'} from {base_url}") from exc

    model = choose_model(capabilities, config.base_model)
    print(f"{prefix}: model={model}")

    async with smoke_step_async(prefix, f"create LoRA training client for {model}"):
        training_client = await service_client.create_lora_training_client_async(
            base_model=model,
            rank=config.lora_rank,
            train_mlp=True,
            train_attn=True,
            train_unembed=True,
        )
    with smoke_step(prefix, f"load tokenizer for {model}"):
        tokenizer = get_tokenizer(training_client, model)

    return SmokeContext(
        service_client=service_client,
        capabilities=capabilities,
        model=model,
        training_client=training_client,
        tokenizer=tokenizer,
        config=config,
    )


def resume_training_client_from_state(
    prefix: str,
    service_client: Any,
    model: str,
    state_path: str,
    config: SmokeConfig | None = None,
) -> tuple[Any, str]:
    config = config or SmokeConfig()
    with smoke_step(prefix, f"create resumed LoRA training client for {model}"):
        resumed_client = service_client.create_lora_training_client(
            base_model=model,
            rank=config.lora_rank,
            train_mlp=True,
            train_attn=True,
            train_unembed=True,
        )
    with smoke_step(prefix, "load state with optimizer into resumed training client"):
        resumed_client.load_state_with_optimizer(state_path).result()
    return resumed_client, "create_lora_training_client + load_state_with_optimizer"
