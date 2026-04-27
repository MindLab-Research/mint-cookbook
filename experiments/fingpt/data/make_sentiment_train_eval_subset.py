#!/usr/bin/env python3
"""Build a small sentiment eval subset for faster training-time checks."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
SENTIMENT_BENCHMARK_DIR = SCRIPT_DIR / "benchmarks" / "sentiment"
DEFAULT_OUTPUT_DIR = SENTIMENT_BENCHMARK_DIR / "train-eval-160"
BENCHMARKS = ("fpb", "fiqa-sa", "tfns", "nwgi")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a stratified small eval subset from the four FinGPT sentiment benchmarks."
    )
    parser.add_argument("--total", type=int, default=160, help="Total rows to keep.")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed.")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where the subset files will be written.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise RuntimeError(f"Expected object row at {path}:{line_no}")
            rows.append(payload)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def largest_remainder_alloc(total: int, weights: dict[str, int]) -> dict[str, int]:
    if total < 0:
        raise ValueError("total must be non-negative")
    if not weights:
        return {}
    weight_sum = sum(weights.values())
    if weight_sum <= 0:
        raise ValueError("weights must sum to a positive value")

    raw_targets = {
        name: (total * weight / weight_sum)
        for name, weight in weights.items()
    }
    allocation = {
        name: min(weights[name], math.floor(raw_target))
        for name, raw_target in raw_targets.items()
    }
    assigned = sum(allocation.values())
    remainder_order = sorted(
        weights,
        key=lambda name: (raw_targets[name] - allocation[name], weights[name], name),
        reverse=True,
    )
    for name in remainder_order:
        if assigned >= total:
            break
        if allocation[name] >= weights[name]:
            continue
        allocation[name] += 1
        assigned += 1

    if assigned != total:
        raise RuntimeError(
            f"Could not allocate exactly {total} rows; assigned {assigned} rows."
        )
    return allocation


def stratified_label_alloc(total: int, label_counts: Counter[str]) -> dict[str, int]:
    labels = [label for label, count in label_counts.items() if count > 0]
    if total < len(labels):
        raise ValueError(
            f"Need at least {len(labels)} rows to preserve all labels, got {total}."
        )

    # Reserve one row per non-empty class so the tiny eval still covers every label.
    base_alloc = {label: 1 for label in labels}
    if total == len(labels):
        return base_alloc

    remaining_weights = {
        label: label_counts[label] - 1
        for label in labels
    }
    remaining_alloc = largest_remainder_alloc(total - len(labels), remaining_weights)
    return {
        label: base_alloc[label] + remaining_alloc.get(label, 0)
        for label in labels
    }


def sample_rows_for_label(
    rows: list[tuple[int, dict[str, Any]]],
    target: int,
    seed: int,
    benchmark: str,
    label: str,
) -> list[tuple[int, dict[str, Any]]]:
    if target <= 0:
        return []
    if target > len(rows):
        raise ValueError(
            f"Cannot sample {target} rows from {benchmark}/{label} with only {len(rows)} rows."
        )
    shuffled = list(rows)
    random.Random(f"{seed}:{benchmark}:{label}").shuffle(shuffled)
    selected = shuffled[:target]
    return sorted(selected, key=lambda item: item[0])


def build_subset(rows: list[dict[str, Any]], benchmark: str, seed: int, target: int) -> tuple[list[dict[str, Any]], dict[str, int]]:
    indexed_rows = list(enumerate(rows))
    label_buckets: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for item in indexed_rows:
        label = str(item[1]["output"])
        label_buckets.setdefault(label, []).append(item)

    label_counts = Counter({label: len(items) for label, items in label_buckets.items()})
    label_targets = stratified_label_alloc(target, label_counts)

    selected: list[tuple[int, dict[str, Any]]] = []
    for label in sorted(label_targets):
        selected.extend(
            sample_rows_for_label(
                label_buckets[label],
                label_targets[label],
                seed,
                benchmark,
                label,
            )
        )

    selected.sort(key=lambda item: item[0])
    subset_rows = [row for _, row in selected]
    if len(subset_rows) != target:
        raise RuntimeError(
            f"{benchmark}: expected {target} sampled rows, got {len(subset_rows)}"
        )
    return subset_rows, label_targets


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)

    full_rows: dict[str, list[dict[str, Any]]] = {}
    dataset_weights: dict[str, int] = {}
    for benchmark in BENCHMARKS:
        source_path = SENTIMENT_BENCHMARK_DIR / benchmark / "test.jsonl"
        rows = load_jsonl(source_path)
        full_rows[benchmark] = rows
        dataset_weights[benchmark] = len(rows)

    dataset_targets = largest_remainder_alloc(args.total, dataset_weights)

    manifest: dict[str, Any] = {
        "kind": "fingpt_sentiment_train_eval_subset",
        "seed": args.seed,
        "target_total": args.total,
        "source_root": str(SENTIMENT_BENCHMARK_DIR),
        "output_root": str(output_dir),
        "benchmarks": {},
    }
    merged_rows: list[dict[str, Any]] = []

    for benchmark in BENCHMARKS:
        subset_rows, label_targets = build_subset(
            full_rows[benchmark],
            benchmark=benchmark,
            seed=args.seed,
            target=dataset_targets[benchmark],
        )
        write_jsonl(output_dir / benchmark / "test.jsonl", subset_rows)
        merged_rows.extend(subset_rows)

        full_label_counts = Counter(str(row["output"]) for row in full_rows[benchmark])
        subset_label_counts = Counter(str(row["output"]) for row in subset_rows)
        manifest["benchmarks"][benchmark] = {
            "source_path": str(SENTIMENT_BENCHMARK_DIR / benchmark / "test.jsonl"),
            "source_total": len(full_rows[benchmark]),
            "subset_total": len(subset_rows),
            "label_targets": dict(sorted(label_targets.items())),
            "source_label_counts": dict(sorted(full_label_counts.items())),
            "subset_label_counts": dict(sorted(subset_label_counts.items())),
        }

    write_jsonl(output_dir / "all" / "test.jsonl", merged_rows)
    manifest["merged"] = {
        "path": str(output_dir / "all" / "test.jsonl"),
        "total": len(merged_rows),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
