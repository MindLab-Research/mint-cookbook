#!/usr/bin/env python3
"""Convert official LawBench task JSON files into a local eval manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


EXPERIMENT_DIR = Path(__file__).resolve().parents[2]

TASK_SPECS = {
    "1-1": {"task_name": "statute-recitation", "cognitive_level": "memory", "official_metric": "ROUGE-L", "task_type": "generation"},
    "1-2": {"task_name": "legal-qa", "cognitive_level": "memory", "official_metric": "Accuracy", "task_type": "single_choice"},
    "2-1": {"task_name": "document-correction", "cognitive_level": "understanding", "official_metric": "F0.5", "task_type": "generation"},
    "2-2": {"task_name": "dispute-focus-identification", "cognitive_level": "understanding", "official_metric": "F1", "task_type": "multi_choice"},
    "2-3": {"task_name": "marriage-dispute-identification", "cognitive_level": "understanding", "official_metric": "F1", "task_type": "multi_choice"},
    "2-4": {"task_name": "question-topic-identification", "cognitive_level": "understanding", "official_metric": "Accuracy", "task_type": "single_choice"},
    "2-5": {"task_name": "reading-comprehension", "cognitive_level": "understanding", "official_metric": "rc-F1", "task_type": "extraction"},
    "2-6": {"task_name": "named-entity-recognition", "cognitive_level": "understanding", "official_metric": "soft-F1", "task_type": "extraction"},
    "2-7": {"task_name": "public-opinion-summary", "cognitive_level": "understanding", "official_metric": "ROUGE-L", "task_type": "generation"},
    "2-8": {"task_name": "argument-mining", "cognitive_level": "understanding", "official_metric": "Accuracy", "task_type": "single_choice"},
    "2-9": {"task_name": "event-detection", "cognitive_level": "understanding", "official_metric": "F1", "task_type": "multi_choice"},
    "2-10": {"task_name": "trigger-word-extraction", "cognitive_level": "understanding", "official_metric": "soft-F1", "task_type": "extraction"},
    "3-1": {"task_name": "statute-prediction-fact", "cognitive_level": "application", "official_metric": "F1", "task_type": "multi_choice"},
    "3-2": {"task_name": "statute-prediction-scenario", "cognitive_level": "application", "official_metric": "ROUGE-L", "task_type": "generation"},
    "3-3": {"task_name": "charge-prediction", "cognitive_level": "application", "official_metric": "F1", "task_type": "multi_choice"},
    "3-4": {"task_name": "sentence-prediction-without-statute", "cognitive_level": "application", "official_metric": "Normalized log-distance", "task_type": "regression"},
    "3-5": {"task_name": "sentence-prediction-with-statute", "cognitive_level": "application", "official_metric": "Normalized log-distance", "task_type": "regression"},
    "3-6": {"task_name": "case-analysis", "cognitive_level": "application", "official_metric": "Accuracy", "task_type": "single_choice"},
    "3-7": {"task_name": "crime-amount-calculation", "cognitive_level": "application", "official_metric": "Accuracy", "task_type": "regression"},
    "3-8": {"task_name": "legal-consultation", "cognitive_level": "application", "official_metric": "ROUGE-L", "task_type": "generation"},
}

SMOKE_TASK_IDS = (
    "1-2",
    "2-5",
    "2-7",
    "3-3",
    "3-4",
)
DEFAULT_SMOKE_LIMIT_PER_TASK = 1

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--official-data-dir",
        default="data/eval/raw/lawbench-official",
        help="Directory containing the official LawBench <task_id>.json files.",
    )
    parser.add_argument(
        "--output-eval",
        default="data/eval/full.jsonl",
        help="Path to the generated local full eval manifest.",
    )
    parser.add_argument(
        "--output-meta",
        default="data/eval/full.meta.json",
        help="Path to a summary JSON file for the converted full eval manifest.",
    )
    parser.add_argument(
        "--task-id",
        action="append",
        default=[],
        help="Specific LawBench task id to keep. Repeat to build an explicit subset.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Build the repo-standard 5-task smoke subset. "
            "Defaults --limit-per-task to 1 unless you override it."
        ),
    )
    parser.add_argument(
        "--limit-per-task",
        type=int,
        default=None,
        help="Optional cap per task. Use 0 for all rows.",
    )
    return parser.parse_args()


def selected_task_ids(args: argparse.Namespace) -> tuple[str, ...]:
    if args.smoke and args.task_id:
        raise SystemExit("Use either --smoke or --task-id, not both.")
    task_ids = SMOKE_TASK_IDS if args.smoke else (tuple(args.task_id) if args.task_id else tuple(TASK_SPECS))
    unknown = [task_id for task_id in task_ids if task_id not in TASK_SPECS]
    if unknown:
        raise SystemExit(f"Unknown task ids: {', '.join(sorted(unknown))}")
    return task_ids


def resolved_limit_per_task(args: argparse.Namespace) -> int:
    if args.limit_per_task is None:
        return DEFAULT_SMOKE_LIMIT_PER_TASK if args.smoke else 0
    if args.limit_per_task < 0:
        raise SystemExit(f"--limit-per-task must be >= 0, got {args.limit_per_task}")
    return int(args.limit_per_task)


def manifest_kind(*, task_ids: tuple[str, ...], limit_per_task: int) -> str:
    if task_ids == tuple(TASK_SPECS) and limit_per_task == 0:
        return "full"
    if task_ids == SMOKE_TASK_IDS and limit_per_task == DEFAULT_SMOKE_LIMIT_PER_TASK:
        return "smoke"
    return "subset"


def build_prompt(instruction: str, question: str) -> str:
    instruction = instruction.strip()
    question = question.strip()
    if instruction and question:
        return instruction + "\n" + question
    return instruction or question


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


def load_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise RuntimeError(f"Expected a list in {path}")
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            raise RuntimeError(f"Expected object rows in {path}:{index}")
        rows.append(item)
    return rows


def build_eval_rows(
    official_data_dir: Path,
    *,
    task_ids: tuple[str, ...],
    limit_per_task: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not official_data_dir.exists():
        raise SystemExit(f"Official data directory does not exist: {official_data_dir}")

    eval_rows: list[dict[str, Any]] = []
    task_counts: dict[str, int] = {}
    current_manifest_kind = manifest_kind(task_ids=task_ids, limit_per_task=limit_per_task)
    is_full_manifest = current_manifest_kind == "full"
    is_repo_smoke_manifest = current_manifest_kind == "smoke"

    for task_id in task_ids:
        spec = TASK_SPECS[task_id]
        task_path = official_data_dir / f"{task_id}.json"
        if not task_path.exists():
            continue
        task_records = load_records(task_path)
        if limit_per_task > 0:
            task_records = task_records[:limit_per_task]
        task_counts[task_id] = len(task_records)
        for index, record in enumerate(task_records):
            instruction = str(record.get("instruction") or "").strip()
            question = str(record.get("question") or "").strip()
            answer = str(record.get("answer") or "").strip()
            if not instruction and not question:
                raise RuntimeError(f"{task_path}:{index + 1} is missing both instruction and question")
            if not answer and spec["official_metric"] not in {"soft-F1", "rc-F1"}:
                raise RuntimeError(f"{task_path}:{index + 1} is missing answer")
            eval_rows.append(
                {
                    "example_id": f"{task_id}-{index:04d}",
                    "task_id": task_id,
                    "task_name": spec["task_name"],
                    "cognitive_level": spec["cognitive_level"],
                    "prompt": build_prompt(instruction, question),
                    "expected": answer,
                    "instruction": instruction,
                    "question": question,
                    "official_metric": spec["official_metric"],
                    "task_type": spec["task_type"],
                    "metadata": {
                        "official_source_file": portable_path(task_path),
                        "manifest_kind": current_manifest_kind,
                        "full_eval_manifest": is_full_manifest,
                        "repo_smoke_manifest": is_repo_smoke_manifest,
                        "scoring_note": "Scored only through the vendored official LawBench task scorer.",
                    },
                }
            )

    if not eval_rows:
        raise SystemExit(
            "No official task files were found. Put the official LawBench <task_id>.json files under "
            f"{official_data_dir} first."
        )

    summary = {
        "official_data_dir": portable_path(official_data_dir),
        "manifest_kind": current_manifest_kind,
        "row_count": len(eval_rows),
        "task_count": len(task_counts),
        "task_counts": task_counts,
        "selected_task_ids": list(task_ids),
        "limit_per_task": limit_per_task,
        "notes": [
            "The generated JSONL is a local eval manifest for the repo scaffold.",
            "It preserves the original task_id, instruction, question, and answer fields for official scorer execution.",
        ],
    }
    if current_manifest_kind == "smoke":
        summary["notes"].append(
            "The repo-standard smoke manifest keeps 1 row for each of 5 tasks: "
            + ", ".join(SMOKE_TASK_IDS)
            + "."
        )
    elif current_manifest_kind == "subset":
        summary["notes"].append(
            "This manifest keeps an explicit task subset; use selected_task_ids plus task_counts to recover it."
        )
    return eval_rows, summary


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def write_outputs(output_eval: Path, output_meta: Path, eval_rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    payload = dict(summary)
    payload["output_eval"] = portable_path(output_eval)
    write_jsonl(output_eval, eval_rows)
    write_json(output_meta, payload)


def main() -> int:
    args = parse_args()
    official_data_dir = resolve_experiment_path(args.official_data_dir)
    output_eval = resolve_experiment_path(args.output_eval)
    output_meta = resolve_experiment_path(args.output_meta)
    task_ids = selected_task_ids(args)
    limit_per_task = resolved_limit_per_task(args)

    eval_rows, summary = build_eval_rows(
        official_data_dir,
        task_ids=task_ids,
        limit_per_task=limit_per_task,
    )
    write_outputs(output_eval, output_meta, eval_rows, summary)

    print(f"Wrote {len(eval_rows)} rows to {output_eval}")
    print(f"Wrote summary to {output_meta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
