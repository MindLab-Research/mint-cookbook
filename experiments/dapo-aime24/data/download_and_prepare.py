#!/usr/bin/env python3
"""Download and materialize the local dapo-aime24 data files."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq

DAPO_DATASET = "BytedTsinghua-SIA/DAPO-Math-17k"
AIME_DATASET = "Maxwell-Jia/AIME_2024"
DAPO_SOURCE = DAPO_DATASET
DEFAULT_MIRROR_BASE = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com").rstrip("/")
DAPO_PAGE_PATH = f"/datasets/{DAPO_DATASET}"
AIME_PAGE_PATH = f"/datasets/{AIME_DATASET}"
DAPO_RAW_PATH = f"/datasets/{DAPO_DATASET}/resolve/main/data/dapo-math-17k.parquet"
AIME_RAW_PATH = f"/datasets/{AIME_DATASET}/resolve/main/train.jsonl"
REQUEST_HEADERS = {
    "User-Agent": "mint-cookbook-dapo-aime24-data-bootstrap/1.0",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--mirror-base", default=DEFAULT_MIRROR_BASE)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--local-only", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    return parser.parse_args()


def join_url(base: str, path: str) -> str:
    return base.rstrip("/") + path


def download_file(url: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers=REQUEST_HEADERS)
    with urllib.request.urlopen(request) as response, output_path.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)


def prompt_to_text(prompt: Any) -> str:
    if isinstance(prompt, str):
        return prompt.strip()
    if not isinstance(prompt, list):
        raise TypeError(f"Unsupported prompt type: {type(prompt).__name__}")
    parts: list[str] = []
    for item in prompt:
        if isinstance(item, str) and item.strip():
            parts.append(item.strip())
            continue
        if isinstance(item, dict):
            content = item.get("content")
            if isinstance(content, str) and content.strip():
                parts.append(content.strip())
    text = "\n\n".join(parts).strip()
    if not text:
        raise ValueError("Prompt is empty after normalization")
    return text


def extract_dapo_question(prompt_text: str) -> str:
    blocks = [block.strip() for block in prompt_text.split("\n\n") if block.strip()]
    if len(blocks) >= 2 and blocks[0].startswith("Solve the following math problem step by step."):
        tail = blocks[1:]
        if tail and tail[-1].startswith("Remember to put your answer on its own line after"):
            tail = tail[:-1]
        if tail:
            return "\n\n".join(tail).strip()
    return prompt_text.strip()


def build_train_rows(raw_parquet_path: Path) -> list[dict[str, str]]:
    rows = pq.read_table(raw_parquet_path).to_pylist()
    deduped: dict[str, dict[str, str]] = {}
    for row in rows:
        prompt = prompt_to_text(row.get("prompt"))
        extra_info = row.get("extra_info") or {}
        reward_model = row.get("reward_model") or {}
        row_id = str(extra_info["index"])
        if row_id in deduped:
            continue
        deduped[row_id] = {
            "id": row_id,
            "question": extract_dapo_question(prompt),
            "answer": str(reward_model["ground_truth"]),
            "source": DAPO_SOURCE,
        }
    return list(deduped.values())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if not isinstance(row, dict):
                raise TypeError(f"Expected object at {path}:{line_number}")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
            count += 1
    return count


def should_reuse_local(path: Path, force_download: bool) -> bool:
    return path.exists() and (path.is_symlink() or not force_download)


def require_local_or_downloadable(path: Path, local_only: bool, description: str) -> None:
    if local_only and not path.exists():
        raise RuntimeError(f"Missing local {description}: {path}")


def materialize_train(data_dir: Path, mirror_base: str, force_download: bool, local_only: bool) -> None:
    raw_parquet_path = data_dir / "raw_dapo_math_17k.parquet"
    train_jsonl_path = data_dir / "train.jsonl"
    if should_reuse_local(raw_parquet_path, force_download):
        print(f"Reusing local DAPO parquet from {raw_parquet_path}", flush=True)
    else:
        require_local_or_downloadable(raw_parquet_path, local_only, "DAPO parquet")
        dapo_url = join_url(mirror_base, DAPO_RAW_PATH)
        print(f"Downloading DAPO parquet from {dapo_url}", flush=True)
        download_file(dapo_url, raw_parquet_path)
    train_rows = build_train_rows(raw_parquet_path)
    count = write_jsonl(train_jsonl_path, train_rows)
    print(f"Wrote {count} train rows to {train_jsonl_path}", flush=True)


def materialize_eval(data_dir: Path, mirror_base: str, force_download: bool, local_only: bool) -> None:
    raw_aime_path = data_dir / "raw_aime_2024.jsonl"
    eval_jsonl_path = data_dir / "eval.jsonl"

    if should_reuse_local(raw_aime_path, force_download):
        print(f"Reusing local AIME raw file from {raw_aime_path}", flush=True)
        eval_rows = load_jsonl(raw_aime_path)
    else:
        require_local_or_downloadable(raw_aime_path, local_only, "AIME jsonl")
        aime_url = join_url(mirror_base, AIME_RAW_PATH)
        print(f"Downloading AIME rows from {aime_url}", flush=True)
        download_file(aime_url, raw_aime_path)
        eval_rows = load_jsonl(raw_aime_path)

    count = write_jsonl(eval_jsonl_path, eval_rows)
    print(f"Wrote {count} eval rows to {eval_jsonl_path}", flush=True)


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    if args.skip_train and args.skip_eval:
        print("Nothing to do: both --skip-train and --skip-eval were set.", file=sys.stderr)
        return 1

    if not args.skip_train:
        materialize_train(
            data_dir,
            mirror_base=args.mirror_base,
            force_download=args.force_download,
            local_only=args.local_only,
        )
    if not args.skip_eval:
        materialize_eval(
            data_dir,
            mirror_base=args.mirror_base,
            force_download=args.force_download,
            local_only=args.local_only,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
