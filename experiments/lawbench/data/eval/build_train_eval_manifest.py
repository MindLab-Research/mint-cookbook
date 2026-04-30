#!/usr/bin/env python3
"""Build a small eval slice for periodic training-time LawBench evals."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


EXPERIMENT_DIR = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-eval",
        default="data/eval/full.jsonl",
        help="Full eval manifest used as the source for the smaller periodic-eval slice.",
    )
    parser.add_argument(
        "--output-eval",
        default="data/eval/train_eval_200.jsonl",
        help="Output path for the periodic training eval manifest.",
    )
    parser.add_argument(
        "--output-meta",
        default="data/eval/train_eval_200.meta.json",
        help="Output path for the periodic training eval metadata.",
    )
    parser.add_argument(
        "--rows-per-task",
        type=int,
        default=10,
        help="Number of rows to keep per LawBench task. Default 10 yields 200 rows across 20 tasks.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            item = json.loads(stripped)
            if not isinstance(item, dict):
                raise RuntimeError(f"Expected object rows in {path}:{index}")
            rows.append(item)
    return rows


def resolve_experiment_path(path_like: str | Path) -> Path:
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path
    cwd_path = Path.cwd() / path
    if cwd_path.exists() or cwd_path.parent.exists():
        return cwd_path
    return EXPERIMENT_DIR / path


def portable_path(path_like: str | Path) -> str:
    path = resolve_experiment_path(path_like)
    try:
        return path.resolve().relative_to(EXPERIMENT_DIR).as_posix()
    except ValueError:
        return str(path)


def build_train_eval_rows(rows: list[dict[str, Any]], rows_per_task: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if rows_per_task <= 0:
        raise RuntimeError(f"--rows-per-task must be positive, got {rows_per_task}")

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        task_id = str(row.get("task_id") or "").strip()
        if not task_id:
            raise RuntimeError("Every eval row must carry task_id")
        grouped.setdefault(task_id, []).append(row)

    selected: list[dict[str, Any]] = []
    task_counts: dict[str, int] = {}
    for task_id in sorted(grouped):
        task_rows = grouped[task_id]
        if len(task_rows) < rows_per_task:
            raise RuntimeError(
                f"Task {task_id} only has {len(task_rows)} rows, fewer than rows_per_task={rows_per_task}"
            )
        chosen = task_rows[:rows_per_task]
        selected.extend(chosen)
        task_counts[task_id] = len(chosen)

    summary = {
        "input_eval": None,
        "row_count": len(selected),
        "rows_per_task": rows_per_task,
        "task_counts": task_counts,
        "cognitive_level_counts": dict(sorted(Counter(str(row.get("cognitive_level") or "") for row in selected).items())),
        "notes": [
            "This manifest is a small balanced periodic-eval slice for training-time LawBench evals.",
            "It keeps the first N rows per task from the full frozen eval manifest and is not a replacement for the final full benchmark eval.",
        ],
    }
    return selected, summary


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    args = parse_args()
    input_eval = resolve_experiment_path(args.input_eval)
    output_eval = resolve_experiment_path(args.output_eval)
    output_meta = resolve_experiment_path(args.output_meta)
    rows = load_jsonl(input_eval)
    selected, summary = build_train_eval_rows(rows, int(args.rows_per_task))
    summary["input_eval"] = portable_path(input_eval)
    summary["output_eval"] = portable_path(output_eval)
    write_jsonl(output_eval, selected)
    write_json(output_meta, summary)
    print(f"Wrote {len(selected)} rows to {output_eval}")
    print(f"Wrote summary to {output_meta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
