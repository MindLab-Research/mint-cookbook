#!/usr/bin/env python3
"""Minimal MinT repro for the duplicate-prompt concurrent sampling error.

This module sends two concurrent sampling requests with the same prompt through
the MinT compatibility SDK and prints the first failure. It is intentionally
standalone so the issue can be reproduced without running the full LawBench
eval loop.

Two entry points are exposed:

- ``python repro_mint_duplicate_prompt_error.py [flags]`` — CLI, drives
  everything from ``sys.argv`` via :func:`main`.
- ``from ... import run_repro; asyncio.run(run_repro(attempts=3))`` —
  importable entry that takes explicit keyword arguments so the repro can be
  driven from a notebook or another script without touching ``sys.argv``.
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import os
import traceback
from pathlib import Path
from typing import Any, Sequence

from transformers import AutoTokenizer

import mint
from mint import types

EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_HF_HOME = str(Path.home() / ".cache" / "huggingface")
DEFAULT_BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
LOCAL_ENV_KEYS = {"MINT_BASE_URL", "MINT_API_KEY"}

DEFAULT_DUPLICATE_PROMPT = (
    "依据给出的实体类型提取句子的实体信息，实体类型包括:犯罪嫌疑人、受害人、被盗货币、物品价值、"
    "盗窃获利、被盗物品、作案工具、时间、地点、组织机构。逐个列出实体信息。\n"
    "句子:经黑河市爱辉区价格认证中心价格鉴定：被盗虾仁价值人民币200.00元。"
)
DEFAULT_DISTINCT_PROMPT = (
    "依据给出的实体类型提取句子的实体信息，实体类型包括:犯罪嫌疑人、受害人、被盗货币、物品价值、"
    "盗窃获利、被盗物品、作案工具、时间、地点、组织机构。逐个列出实体信息。\n"
    "句子:破案后，公安机关将查获手机依法返还给了被害人严某某、肖某某。"
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--attempts", type=int, default=5, help="How many duplicate-pair attempts to try.")
    parser.add_argument("--timeout", type=float, default=120.0, help="MinT client timeout in seconds.")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument(
        "--mode",
        choices=("duplicate", "distinct"),
        default="duplicate",
        help="Use identical prompts or a distinct control pair.",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_DUPLICATE_PROMPT,
        help="First prompt. In duplicate mode it is used for both requests.",
    )
    parser.add_argument(
        "--other-prompt",
        default=DEFAULT_DISTINCT_PROMPT,
        help="Second prompt for --mode distinct.",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_arg_parser().parse_args(argv)


def load_local_env(path: Path) -> None:
    if path.exists():
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

    os.environ.setdefault("HF_HOME", DEFAULT_HF_HOME)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def cached_tokenizer_dir(model_name: str) -> Path | None:
    if "/" not in model_name:
        return None
    hf_home = Path(os.environ.get("HF_HOME", DEFAULT_HF_HOME)).expanduser()
    hub_root = hf_home / "hub" if (hf_home / "hub").exists() else hf_home / ".cache" / "huggingface" / "hub"
    org, repo = model_name.split("/", 1)
    repo_dir = hub_root / f"models--{org.replace('/', '--')}--{repo.replace('/', '--')}"
    snapshots_dir = repo_dir / "snapshots"
    if not snapshots_dir.exists():
        return None

    def has_tokenizer_files(path: Path) -> bool:
        if not path.is_dir():
            return False
        files = {f.name for f in path.iterdir() if f.is_file()}
        if "tokenizer_config.json" not in files:
            return False
        return "tokenizer.json" in files or {"vocab.json", "merges.txt"}.issubset(files)

    candidates = sorted((p for p in snapshots_dir.iterdir() if has_tokenizer_files(p)), reverse=True)
    return candidates[0] if candidates else None


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
    return [int(t) for t in tokenizer.encode(text + "\n\nassistant:", add_special_tokens=True)]


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


def resolve_api_result(value: Any) -> Any:
    result_fn = getattr(value, "result", None)
    if callable(result_fn):
        return result_fn()
    return value


async def resolve_api_result_async(value: Any) -> Any:
    if inspect.isawaitable(value):
        value = await value
    return resolve_api_result(value)


def create_service_client(*, timeout: float | None = None) -> Any:
    return mint.ServiceClient(
        base_url=os.environ.get("MINT_BASE_URL"),
        timeout=timeout,
    )


async def create_sampling_client(service_client: Any, base_model: str) -> Any:
    create_fn = getattr(service_client, "create_sampling_client_async", None)
    if callable(create_fn):
        return await resolve_api_result_async(create_fn(base_model=base_model))
    create_fn = getattr(service_client, "create_sampling_client", None)
    if callable(create_fn):
        return resolve_api_result(create_fn(base_model=base_model))
    raise RuntimeError("Service client must expose create_sampling_client")


async def sample_once(
    sampler: Any,
    tokenizer: Any,
    prompt: str,
    *,
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    messages = [{"role": "user", "content": prompt}]
    prompt_tokens = build_generation_prompt_tokens(tokenizer, messages)
    sample_fn = getattr(sampler, "sample_async", None)
    if not callable(sample_fn):
        raise RuntimeError("Sampler must expose sample_async for this repro script")
    result = await resolve_api_result_async(
        sample_fn(
            prompt=types.ModelInput.from_ints(tokens=prompt_tokens),
            num_samples=1,
            sampling_params=types.SamplingParams(
                max_tokens=max(int(max_tokens), 1),
                temperature=float(temperature),
                top_p=float(top_p),
            ),
        )
    )
    return tokenizer.decode(extract_completion_tokens(result)).strip()


async def main_async(args: argparse.Namespace) -> int:
    load_local_env(EXPERIMENT_DIR / ".env")
    if not (os.environ.get("MINT_API_KEY") or "").strip():
        raise RuntimeError(f"Missing MINT_API_KEY in {EXPERIMENT_DIR / '.env'} or environment")
    if not (os.environ.get("MINT_BASE_URL") or "").strip():
        raise RuntimeError(f"Missing MINT_BASE_URL in {EXPERIMENT_DIR / '.env'} or environment")

    service_client = create_service_client(timeout=args.timeout)
    sampler = await create_sampling_client(service_client, args.base_model)
    tokenizer = get_tokenizer(sampler, args.base_model)
    prompt_a = args.prompt
    prompt_b = args.prompt if args.mode == "duplicate" else args.other_prompt

    print(f"mint_version={getattr(mint, '__version__', 'unknown')}")
    print(f"mint_base_url={os.environ.get('MINT_BASE_URL')}")
    print(f"mode={args.mode} attempts={args.attempts} same_prompt={prompt_a == prompt_b}")

    for attempt in range(1, max(int(args.attempts), 1) + 1):
        print(f"===== attempt {attempt} =====")
        try:
            results = await asyncio.gather(
                sample_once(
                    sampler,
                    tokenizer,
                    prompt_a,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                ),
                sample_once(
                    sampler,
                    tokenizer,
                    prompt_b,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                ),
            )
            for idx, text in enumerate(results, start=1):
                preview = text.replace("\n", "\\n")[:120]
                print(f"result_{idx}={preview}")
        except Exception as exc:
            print(f"FAILED on attempt {attempt}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            return 1

    print("No failure reproduced within the configured attempts.")
    return 2


async def run_repro(**overrides: Any) -> int:
    """Importable entry point: drive the repro with explicit keyword args.

    Accepts any subset of the CLI flags as Python kwargs (with underscores in
    place of dashes, e.g. ``base_model``, ``max_tokens``). Returns the same
    exit code :func:`main_async` would return when driven from the CLI.
    """
    defaults = vars(parse_args([]))
    unknown = set(overrides) - set(defaults)
    if unknown:
        raise TypeError(f"Unknown run_repro argument(s): {sorted(unknown)}")
    defaults.update(overrides)
    return await main_async(argparse.Namespace(**defaults))


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
