#!/usr/bin/env python3
"""Scan repeated top-level helpers and compare their implementation hashes."""

from __future__ import annotations

import argparse
import ast
import fnmatch
import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path


DEFAULT_PATTERNS = [
    "experiments/*/train.py",
    "scaffolds/single_file_experiment/train.py.tpl",
]


@dataclass(frozen=True)
class HelperRecord:
    path: str
    lineno: int
    kind: str
    hash12: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan repeated top-level helpers and compare hashes."
    )
    parser.add_argument(
        "--root",
        default=".",
        help="Repo root. Defaults to current working directory.",
    )
    parser.add_argument(
        "--pattern",
        action="append",
        default=[],
        help="Glob to include. May be repeated. Defaults to experiments/*/train.py and the single-file template.",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Glob to exclude after includes. May be repeated.",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=[],
        help="Restrict output to these helper names.",
    )
    parser.add_argument(
        "--ignore",
        nargs="*",
        default=[],
        help="Ignore these helper names after scanning. Useful for known intentional variants.",
    )
    parser.add_argument(
        "--show-identical",
        action="store_true",
        help="Also show repeated helpers whose hashes are already identical.",
    )
    return parser.parse_args()


def iter_target_files(root: Path, patterns: list[str], excludes: list[str]) -> list[Path]:
    chosen: dict[Path, None] = {}
    for pattern in patterns:
        for path in root.glob(pattern):
            if path.is_file():
                chosen[path.resolve()] = None
    kept = []
    for path in sorted(chosen):
        rel = path.relative_to(root.resolve()).as_posix()
        if any(fnmatch.fnmatch(rel, pat) for pat in excludes):
            continue
        kept.append(path)
    return kept


def scan_file(root: Path, path: Path) -> dict[str, HelperRecord]:
    mod = ast.parse(path.read_text(encoding="utf-8"))
    found: dict[str, HelperRecord] = {}
    for node in mod.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            src = ast.unparse(node)
            found[node.name] = HelperRecord(
                path=path.relative_to(root).as_posix(),
                lineno=node.lineno,
                kind="async" if isinstance(node, ast.AsyncFunctionDef) else "sync",
                hash12=hashlib.sha256(src.encode("utf-8")).hexdigest()[:12],
            )
    return found


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve()
    patterns = args.pattern or DEFAULT_PATTERNS
    target_files = iter_target_files(root, patterns, args.exclude)
    if not target_files:
        print("No target files matched.", file=sys.stderr)
        return 1

    by_name: dict[str, list[HelperRecord]] = {}
    for path in target_files:
        for name, record in scan_file(root, path).items():
            by_name.setdefault(name, []).append(record)

    only = set(args.only)
    repeated = {
        name: records
        for name, records in by_name.items()
        if len(records) > 1
        and (not only or name in only)
        and name not in set(args.ignore)
    }
    if not repeated:
        print("No repeated helpers found.")
        return 0

    drift_count = 0
    identical_count = 0
    for name in sorted(repeated):
        records = repeated[name]
        hashes = {record.hash12 for record in records}
        if len(hashes) == 1:
            identical_count += 1
            if not args.show_identical:
                continue
            status = "IDENTICAL"
        else:
            drift_count += 1
            status = "DRIFT"
        print(f"{status} {name}")
        for record in records:
            print(f"  {record.path}:{record.lineno} {record.kind} {record.hash12}")

    print()
    print(
        f"Summary: repeated={len(repeated)} drift={drift_count} identical={identical_count}"
    )
    return 0 if drift_count == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
