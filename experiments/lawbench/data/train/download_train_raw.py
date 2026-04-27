#!/usr/bin/env python3
"""Download the public DISC-Law-SFT release used for LawBench SFT training."""

from __future__ import annotations

import argparse
import json
import shutil
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_REPO_ID = "ShengbinYue/DISC-Law-SFT"
DEFAULT_REVISION = "main"
DEFAULT_OUTPUT_DIR = "data/train/raw/disc-law-sft"
DEFAULT_METADATA_NAME = "download.meta.json"
DEFAULT_FILES = (
    "DISC-Law-SFT-Pair.jsonl",
    "DISC-Law-SFT-Pair-QA-released.jsonl",
    "DISC-Law-SFT-Triplet-released.jsonl",
    "DISC-Law-SFT-Triplet-QA-released.jsonl",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory that stores the raw DISC-Law-SFT files.",
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help="Hugging Face dataset repo in owner/name form.",
    )
    parser.add_argument(
        "--revision",
        default=DEFAULT_REVISION,
        help="Dataset revision to snapshot.",
    )
    parser.add_argument(
        "--file",
        action="append",
        default=[],
        help="Specific dataset file to download. Repeat to fetch a subset. Defaults to the maintained public release set.",
    )
    parser.add_argument(
        "--metadata-name",
        default=DEFAULT_METADATA_NAME,
        help="Metadata filename written under --output-dir.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Do not re-download files that already exist locally.",
    )
    return parser.parse_args()


def selected_files(args: argparse.Namespace) -> tuple[str, ...]:
    return tuple(args.file) if args.file else DEFAULT_FILES


def build_download_url(repo_id: str, revision: str, filename: str) -> str:
    return f"https://huggingface.co/datasets/{repo_id}/resolve/{revision}/{filename}"


def count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, destination.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def build_file_record(output_dir: Path, repo_id: str, revision: str, filename: str, *, skipped: bool) -> dict[str, Any]:
    path = output_dir / filename
    return {
        "filename": filename,
        "path": str(path.resolve()),
        "download_url": build_download_url(repo_id, revision, filename),
        "bytes": path.stat().st_size,
        "line_count": count_lines(path),
        "skipped_existing": skipped,
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    files = selected_files(args)
    records: list[dict[str, Any]] = []

    for filename in files:
        destination = output_dir / filename
        skipped = args.skip_existing and destination.exists()
        if not skipped:
            download_file(build_download_url(args.repo_id, args.revision, filename), destination)
        records.append(build_file_record(output_dir, args.repo_id, args.revision, filename, skipped=skipped))
        print(f"{'Kept' if skipped else 'Downloaded'} {filename}")

    metadata = {
        "repo_id": args.repo_id,
        "revision": args.revision,
        "output_dir": str(output_dir.resolve()),
        "file_count": len(records),
        "files": records,
        "notes": [
            "This script snapshots the maintained DISC-Law-SFT public release for local train-data materialization.",
            "Run data/train/build_train_manifest.py separately to build the benchmark-decoupled local train artifact.",
        ],
    }
    write_json(output_dir / args.metadata_name, metadata)
    print(f"Wrote metadata to {output_dir / args.metadata_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
