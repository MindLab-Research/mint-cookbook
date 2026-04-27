# mint-cookbook

`mint-cookbook` is a monorepo of independent, self-contained MinT reproductions.
Each directory under `experiments/` is a runnable work unit: enter the directory, read local `AGENTS.md` and `PROMPT.md` when present, then `README.md` and `train.py`, run `uv sync`, and validate the local benchmark path before changing anything.

## What This Repo Optimizes For

- Eval-first reproductions with stable benchmark entrypoints
- Small, readable experiment directories instead of early shared frameworks
- Single-file experiment runtimes with harness-level reuse, not cross-experiment imports
- Clear separation between train data, eval data, benchmark protocol, and practical execution baseline

## Workflow

Default order inside one experiment:

1. read local `AGENTS.md` and `PROMPT.md` when present, then `README.md` and `train.py`
2. `uv sync`
3. `uv run train.py --dry-run`
4. `uv run train.py --eval-only`
5. only then run training or the experiment wrapper

Repo-level docs by scope:

- `AGENTS.md`: stable repo rules and hard constraints
- `PROMPT.md`: current repo-wide task package
- `experiments/README.md`: shared experiment contract
- `scaffolds/README.md`: scaffold and ownership rules

## Maintained Experiments

- `experiments/dapo-aime24`: direct GRPO on DAPO-Math-17k with frozen AIME 2024 eval
- `experiments/fingpt`: FinGPT reproduction scaffold with Fineval + sentiment eval and an SFT path
- `experiments/lawbench`: LawBench benchmark-first scaffold with a maintained LoRA SFT line
- `experiments/chat-dpo`: pairwise chat DPO with held-out preference eval

## Quick Start

```bash
cd experiments/fingpt
uv sync
uv run train.py --dry-run
uv run train.py --eval-only
```

Each experiment `README.md` should be enough to run that experiment without reading the rest of the repo.

## Notes

- Keep experiments self-contained.
- Do not import helpers across experiments.
- When logging, CLI, artifacts, or stdout contracts change, update code and docs in the same change set.
- New experiments still start from the scaffold flow in `scaffolds/`, even though the currently maintained experiments use MinT-backed runtimes.
