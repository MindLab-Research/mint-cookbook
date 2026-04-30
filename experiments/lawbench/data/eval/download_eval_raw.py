#!/usr/bin/env python3
"""Download the official LawBench task JSON files used for benchmark eval."""

from __future__ import annotations

import argparse
import json
import shutil
import urllib.request
from pathlib import Path
from typing import Any

from build_eval_manifest import TASK_SPECS


EXPERIMENT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_OWNER = "open-compass"
DEFAULT_REPO = "LawBench"
DEFAULT_REVISION = "main"
DEFAULT_OUTPUT_DIR = "data/eval/raw/lawbench-official"
DEFAULT_METADATA_NAME = "download.meta.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory that stores the raw official <task_id>.json files.",
    )
    parser.add_argument("--repo-owner", default=DEFAULT_OWNER)
    parser.add_argument("--repo-name", default=DEFAULT_REPO)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument(
        "--task-id",
        action="append",
        default=[],
        help="Specific LawBench task id to download. Repeat to fetch a subset. Defaults to the full official task list.",
    )
    parser.add_argument(
        "--metadata-name",
        default=DEFAULT_METADATA_NAME,
        help="Metadata filename written under --output-dir.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Do not re-download task files that already exist locally.",
    )
    return parser.parse_args()


def selected_task_ids(args: argparse.Namespace) -> tuple[str, ...]:
    task_ids = tuple(args.task_id) if args.task_id else tuple(TASK_SPECS)
    unknown = [task_id for task_id in task_ids if task_id not in TASK_SPECS]
    if unknown:
        raise SystemExit(f"Unknown task ids: {', '.join(sorted(unknown))}")
    return task_ids


def build_download_url(owner: str, repo: str, revision: str, task_id: str) -> str:
    return f"https://raw.githubusercontent.com/{owner}/{repo}/{revision}/data/zero_shot/{task_id}.json"


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, destination.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def load_row_count(path: Path) -> int:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise RuntimeError(f"Expected a list in {path}")
    return len(payload)


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


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    task_ids = selected_task_ids(args)
    output_dir = resolve_experiment_path(args.output_dir)
    records: list[dict[str, Any]] = []

    for task_id in task_ids:
        destination = output_dir / f"{task_id}.json"
        skipped = args.skip_existing and destination.exists()
        if not skipped:
            download_file(
                build_download_url(args.repo_owner, args.repo_name, args.revision, task_id),
                destination,
            )
        records.append(
            {
                "task_id": task_id,
                "path": portable_path(destination),
                "download_url": build_download_url(args.repo_owner, args.repo_name, args.revision, task_id),
                "row_count": load_row_count(destination),
                "skipped_existing": skipped,
            }
        )
        print(f"{'Kept' if skipped else 'Downloaded'} {task_id}")

    metadata = {
        "repo_owner": args.repo_owner,
        "repo_name": args.repo_name,
        "revision": args.revision,
        "output_dir": portable_path(output_dir),
        "task_count": len(records),
        "tasks": records,
        "notes": [
            "This script snapshots the raw official LawBench task files for local benchmark eval preparation.",
            "Run data/eval/build_eval_manifest.py separately to build the frozen local eval manifest.",
        ],
    }
    write_json(output_dir / args.metadata_name, metadata)
    print(f"Wrote metadata to {output_dir / args.metadata_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
