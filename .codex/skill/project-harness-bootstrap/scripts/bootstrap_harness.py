#!/usr/bin/env python3
"""Bootstrap the minimal harness for a MinT experiment monorepo."""

import argparse
import shutil
from pathlib import Path


PLACEHOLDER_KEYS = ("PROJECT_NAME", "MODE", "STACK")
TEMPLATE_TARGETS = (
    Path("README.md.template"),
    Path("AGENTS.md.template"),
    Path(".env.example.template"),
    Path(".gitignore.template"),
    Path("experiments/README.md.template"),
)
REQUIRED_SCAFFOLD_FILES = (
    Path("README.md"),
    Path("profiles/sft.md"),
    Path("profiles/eval.md"),
    Path("profiles/grpo.md"),
    Path("single_file_experiment/README.md"),
    Path("single_file_experiment/README.md.tpl"),
    Path("single_file_experiment/autoresearch.md.tpl"),
    Path("single_file_experiment/autoresearch.sh.tpl"),
    Path("single_file_experiment/capability_matrix.md"),
    Path("single_file_experiment/env.tpl"),
    Path("single_file_experiment/naming.md"),
    Path("single_file_experiment/pyproject.toml.tpl"),
    Path("single_file_experiment/train.py.tpl"),
    Path("single_file_experiment/data/sources.yaml.tpl"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap the root harness for an experiment repo.")
    parser.add_argument("--root", required=True, help="Target repository root")
    parser.add_argument("--project", required=True, help="Project name")
    parser.add_argument(
        "--mode",
        default="greenfield",
        choices=["greenfield", "retrofit"],
        help="Bootstrap mode",
    )
    parser.add_argument(
        "--stack",
        default="research",
        choices=["generic", "python", "node", "research", "fullstack"],
        help="Primary stack hint",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    return parser.parse_args()


def render_template(text: str, context: dict[str, str]) -> str:
    for key in PLACEHOLDER_KEYS:
        text = text.replace(f"{{{{{key}}}}}", context[key])
    return text


def write_file(path: Path, content: str, overwrite: bool) -> str:
    if path.exists() and not overwrite:
        return f"skip {path}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return f"write {path}"


def canonical_repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def canonical_scaffolds_root() -> Path:
    source = canonical_repo_root() / "scaffolds"
    missing = [str(source / rel) for rel in REQUIRED_SCAFFOLD_FILES if not (source / rel).exists()]
    if missing:
        joined = "\n".join(f"- {path}" for path in missing)
        raise RuntimeError(f"canonical scaffolds are incomplete:\n{joined}")
    return source


def sync_scaffolds(source_root: Path, target_root: Path, overwrite: bool) -> list[str]:
    actions: list[str] = []
    for source_path in sorted(source_root.rglob("*")):
        if source_path.is_dir():
            continue
        rel = source_path.relative_to(source_root)
        target_path = target_root / rel
        if target_path.exists() and not overwrite:
            actions.append(f"skip {target_path}")
            continue
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)
        actions.append(f"write {target_path}")
    return actions


def missing_scaffold_files(scaffolds_root: Path) -> list[Path]:
    return [rel for rel in REQUIRED_SCAFFOLD_FILES if not (scaffolds_root / rel).exists()]


def main() -> None:
    args = parse_args()
    skill_root = Path(__file__).resolve().parents[1]
    templates_root = skill_root / "templates"
    repo_root = Path(args.root).expanduser().resolve()
    repo_root.mkdir(parents=True, exist_ok=True)

    context = {
        "PROJECT_NAME": args.project,
        "MODE": args.mode,
        "STACK": args.stack,
    }

    actions: list[str] = []

    for rel in TEMPLATE_TARGETS:
        template_path = templates_root / rel
        target_rel = Path(str(rel)[: -len(".template")])
        target_path = repo_root / target_rel
        content = render_template(template_path.read_text(encoding="utf-8"), context)
        actions.append(write_file(target_path, content, args.overwrite))

    directory = repo_root / "experiments"
    directory.mkdir(parents=True, exist_ok=True)
    gitkeep = directory / ".gitkeep"
    if not gitkeep.exists():
        gitkeep.write_text("", encoding="utf-8")
        actions.append(f"write {gitkeep}")
    else:
        actions.append(f"skip {gitkeep}")

    scaffolds_root = repo_root / "scaffolds"
    source_scaffolds = canonical_scaffolds_root()
    actions.extend(sync_scaffolds(source_scaffolds, scaffolds_root, overwrite=args.overwrite))

    missing = missing_scaffold_files(scaffolds_root)
    if missing:
        joined = "\n".join(f"- {scaffolds_root / rel}" for rel in missing)
        raise RuntimeError(f"target scaffolds are incomplete after sync:\n{joined}")

    print("Bootstrapped harness files:")
    for action in actions:
        print(f"- {action}")

    print("\nNext steps:")
    print("- Review README.md and AGENTS.md so they match the repo you actually want.")
    print("- Review scaffolds/ before creating the first experiment so repo-level template changes stay centralized.")
    print("- Make sure each experiment exposes a benchmark through autoresearch.sh and METRIC output.")


if __name__ == "__main__":
    main()
