# mint-cookbook

`mint-cookbook` is a monorepo of independent, self-contained MinT reproductions.
Each directory under `experiments/` is a runnable work unit: enter the directory, read local `AGENTS.md`, then `PROMPT.md` when present, then `README.md` and `train.py`, run `uv sync`, and validate the local benchmark path before changing anything.

## What This Repo Optimizes For

- Eval-first reproductions with stable benchmark entrypoints
- Small, readable experiment directories instead of early shared frameworks
- Single-file experiment runtimes with harness-level reuse, not cross-experiment imports
- Clear separation between train data, eval data, benchmark protocol, and practical execution baseline

## Workflow

Default order inside one experiment:

1. read local `AGENTS.md`, then `PROMPT.md` when present, then `README.md` and `train.py`
2. `uv sync`
3. `uv run train.py --dry-run --eval-data <smoke_eval_path>`
4. `uv run train.py --eval-only --eval-data <full_eval_path>`
5. only then run training or the experiment wrapper

For expensive live benchmark reruns, prefer the experiment-local dry-run or single eval-only smoke path first. The full benchmark cost differs a lot across experiments, especially for `dapo-aime`, `fingpt`, and `lawbench`.

Repo-level docs by scope:

- `AGENTS.md`: stable repo rules and hard constraints
- `PROMPT.md`: current repo-wide task package
- `experiments/maintained.json`: machine-readable maintained experiment registry for developers, AI agents, and repo tooling
- `experiments/README.md`: shared experiment contract
- `tests/README.md`: repo-level verification entrypoint for local contract tests plus live MinT smoke scripts
- `scaffolds/README.md`: scaffold and ownership rules
- `skills/README.md`: shared skill layout plus local tool-routing guidance
- `docs/repo-overview.md`: longer current-state repo map when you need more context

## Shared Skills

Repo-local reusable agent skills live under `skills/`. Treat that directory as the single source of truth.
When adding or updating a repo-local skill, edit `skills/<name>/` directly.
For local discovery in a developer checkout, point tool-specific skill directories back to that source of truth with symlinks such as `.codex/skills -> ../skills` and `.claude/skills -> ../skills`.
Do not duplicate skill contents under `.codex/` or `.claude/`; `skills/` remains the only checked-in source of truth.

## Maintained Experiments

<!-- maintained-experiments:start -->
- `chat-dpo`: pairwise chat DPO with held-out preference eval
- `dapo-aime`: direct GRPO on DAPO-Math-17k with an AIME 2024 benchmark plus AIME 2025/2026 eval manifests
- `fingpt`: FinGPT reproduction scaffold with Fineval + sentiment eval and an SFT path
- `lawbench`: LawBench benchmark-first scaffold with a maintained LoRA SFT line
<!-- maintained-experiments:end -->

## Quick Start

```bash
cd experiments/fingpt
uv sync
uv run train.py --dry-run --task-type fineval --eval-data smoke:data/smoke_eval.jsonl
uv run train.py --eval-only --task-type fineval --eval-data fineval:data/fingpt-fineval/test.jsonl
```

Each experiment `README.md` should be enough to run that experiment without reading the rest of the repo.

## Notes

- Keep experiments self-contained.
- Do not import helpers across experiments.
- When logging, CLI, artifacts, or stdout contracts change, update code and docs in the same change set.
- The shared repo-root `.env.example` is the only checked-in live env template; runtime code still reads only the local `.env` beside the entrypoint or harness you actually run.
- New experiments still start from the scaffold flow in `scaffolds/`, and that scaffold now defaults to `import mint`, `MINT_*`, and `--mint-timeout` from the first draft rather than a tinker-first runtime skeleton.
