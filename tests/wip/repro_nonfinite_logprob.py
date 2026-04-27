#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = REPO_ROOT / "experiments" / "dapo-aime24"
ARTIFACTS_DIR = EXPERIMENT_DIR / "artifacts"
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

import mint
from mint import types

import train as dapo_train
EMBEDDED_CASES = [
    {
        "id": "9a9b6eb4-a1cb-49d1-8c1e-62eaf2f74079",
        "question": "In triangle $ABC$, $\\sin \\angle A = \\frac{4}{5}$ and $\\angle A < 90^\\circ$. Let $D$ be a point outside triangle $ABC$ such that $\\angle BAD = \\angle DAC$ and $\\angle BDC = 90^\\circ$. Suppose that $AD = 1$ and that $\\frac{BD}{CD} = \\frac{3}{2}$. If $AB + AC$ can be expressed in the form $\\frac{a\\sqrt{b}}{c}$ where $a, b, c$ are pairwise relatively prime integers, find $a + b + c$.",
        "answer": "34",
        "source": "BytedTsinghua-SIA/DAPO-Math-17k",
    },
    {
        "id": "b426d104-244d-4831-a2c4-cd756b61700a",
        "question": "Let $ABCD$ be a unit square in the plane. Points $X$ and $Y$ are chosen independently and uniformly at random on the perimeter of $ABCD$. If the expected value of the area of triangle $\\triangle AXY$ can be expressed as $\\frac{m}{n}$ for relatively prime positive integers $m$ and $n$, compute $m+n$.",
        "answer": "113",
        "source": "BytedTsinghua-SIA/DAPO-Math-17k",
    },
    {
        "id": "6ff0b17f-7e5c-4ae9-b5e9-63ebecd2b9f7",
        "question": "Let $a, b, c$ be distinct numbers such that the equations $x^2 + ax + 1 = 0$ and $x^2 + bx + c = 0$ have a common real root, and the equations $x^2 + x + a = 0$ and $x^2 + cx + b = 0$ also have a common real root. Compute the sum $a + b + c$.",
        "answer": "-3",
        "source": "BytedTsinghua-SIA/DAPO-Math-17k",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Minimal repro for backend Non-finite sampled-token logprob failures during LoRA sampling."
    )
    parser.add_argument("--base-model", default="Qwen/Qwen3-4B-Thinking-2507")
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--mint-timeout", type=float, default=600.0)
    parser.add_argument("--row-index", type=int, default=1, help="1-based embedded-case index to sample")
    parser.add_argument("--scan-limit", type=int, default=0, help="Sample embedded cases 1..N until one fails")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stop-at-eos", action="store_true")
    return parser.parse_args()


def load_embedded_cases() -> list[dict[str, Any]]:
    return [
        dapo_train.make_math_record(
            row_id=case["id"],
            question=case["question"],
            answer=case["answer"],
            source=case["source"],
        )
        for case in EMBEDDED_CASES
    ]


def build_sampling_params(args: argparse.Namespace, tokenizer: Any, *, seed_offset: int) -> Any:
    kwargs: dict[str, Any] = {
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "seed": args.seed + seed_offset,
    }
    if args.stop_at_eos:
        kwargs["stop_token_ids"] = [tokenizer.eos_token_id]
    return types.SamplingParams(**kwargs)


def build_failure_report(
    args: argparse.Namespace,
    row_index: int,
    row: dict[str, Any],
    prompt_tokens: list[int],
    exc: BaseException,
) -> dict[str, Any]:
    return {
        "base_model": args.base_model,
        "rank": args.rank,
        "mint_timeout": args.mint_timeout,
        "row_index": row_index,
        "row_id": row["id"],
        "question": row["question"],
        "messages_preview": dapo_train.preview_messages(row["messages"]),
        "prompt_token_count": len(prompt_tokens),
        "sampling": {
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "seed": args.seed + row_index,
            "stop_at_eos": args.stop_at_eos,
        },
        "exception_type": type(exc).__name__,
        "exception": str(exc),
        "traceback": traceback.format_exc(),
        "timestamp_unix": time.time(),
    }


async def sample_row(
    sampler: Any,
    tokenizer: Any,
    row: dict[str, Any],
    args: argparse.Namespace,
    *,
    row_index: int,
) -> dict[str, Any]:
    prompt_tokens = dapo_train.render_chat_prompt_tokens(tokenizer, row["messages"])
    sampling_params = build_sampling_params(args, tokenizer, seed_offset=row_index)
    print(
        f"== sample row={row_index} id={row['id']} prompt_tokens={len(prompt_tokens)} temperature={args.temperature} max_tokens={args.max_tokens}",
        flush=True,
    )
    result = await dapo_train.sample_one_async(
        sampler=sampler,
        prompt_tokens=prompt_tokens,
        num_samples=1,
        sampling_params=sampling_params,
    )
    token_groups, logprob_groups = dapo_train.extract_sample_tokens_and_logprobs(result)
    response_tokens = token_groups[0] if token_groups else []
    response_text = tokenizer.decode(response_tokens).strip() if response_tokens else ""
    extracted = dapo_train.extract_answer(response_text)
    return {
        "row_index": row_index,
        "row_id": row["id"],
        "response_token_count": len(response_tokens),
        "logprob_count": len(logprob_groups[0]) if logprob_groups else 0,
        "response_preview": dapo_train.preview_text(response_text, 240),
        "extracted_answer": extracted,
    }


async def main_async() -> int:
    args = parse_args()
    api_key = (os.environ.get("MINT_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError(f"missing MINT_API_KEY in {EXPERIMENT_DIR / '.env'} or shell environment")

    train_rows = load_embedded_cases()
    if args.scan_limit > 0:
        candidate_indices = range(1, min(args.scan_limit, len(train_rows)) + 1)
    else:
        candidate_indices = [args.row_index]

    service_client = mint.ServiceClient(
        base_url=os.environ.get("MINT_BASE_URL"),
        timeout=args.mint_timeout,
    )
    capabilities = await dapo_train.preflight_connection_async(service_client)
    supported_models = getattr(capabilities, "supported_models", None)
    if isinstance(supported_models, list):
        print(f"OK mint_server supported_models={len(supported_models)}", flush=True)

    training_client = await dapo_train.create_lora_training_client_async(
        service_client,
        base_model=args.base_model,
        rank=args.rank,
    )
    print(f">> training_client {dapo_train.format_debug_ids(**dapo_train.get_training_client_ids(training_client))}", flush=True)
    tokenizer = dapo_train.get_tokenizer(training_client, args.base_model)
    print(f"@@ model={args.base_model} vocab={tokenizer.vocab_size:,}", flush=True)
    sampler = await dapo_train.save_sampling_client_async(training_client, name=f"repro-{time.time_ns()}")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = ARTIFACTS_DIR / "nonfinite_logprob_repro.json"

    for row_index in candidate_indices:
        row = train_rows[row_index - 1]
        try:
            payload = await sample_row(sampler, tokenizer, row, args, row_index=row_index)
            print(json.dumps({"status": "ok", **payload}, ensure_ascii=True), flush=True)
        except Exception as exc:
            prompt_tokens = dapo_train.render_chat_prompt_tokens(tokenizer, row["messages"])
            report = build_failure_report(args, row_index, row, prompt_tokens, exc)
            report_path.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
            print(f"!! failure_report={report_path}", flush=True)
            print(json.dumps({"status": "failed", "row_index": row_index, "row_id": row['id']}, ensure_ascii=True), flush=True)
            raise

    print("OK no failure reproduced in requested rows", flush=True)
    return 0


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    raise SystemExit(main())
