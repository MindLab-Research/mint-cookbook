#!/usr/bin/env python3
"""Download the public FinGPT sentiment training dataset into local JSONL."""

from __future__ import annotations

import json
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "fingpt-sentiment-train"
DATASET_NAME = "FinGPT/fingpt-sentiment-train"


def write_split(split_name: str, rows: list[dict[str, object]]) -> None:
    path = OUT_DIR / f"{split_name}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for index, row in enumerate(rows, start=1):
            payload = dict(row)
            payload.setdefault("id", f"{split_name}-{index:05d}")
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    print(f"wrote {path} rows={len(rows)}")


def main() -> int:
    dataset = load_dataset(DATASET_NAME)
    for split_name in dataset.keys():
        write_split(split_name, list(dataset[split_name]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
