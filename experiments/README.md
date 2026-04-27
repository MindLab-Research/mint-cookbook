# Experiments

Each subdirectory under `experiments/` is an independent, self-contained reproduction.
You should be able to enter one directory, read local `AGENTS.md` and `PROMPT.md` when present, then `README.md`, run `uv sync`, and validate the benchmark path without reading the rest of the repo.

## Shared Contract

Every experiment should keep these core files local:

```text
<name>/
├── pyproject.toml
├── README.md
├── train.py
├── autoresearch.sh
├── autoresearch.md
└── data/
```

Optional local context files:

- `AGENTS.md`: experiment-local truth for active or sharp-edged work
- `PROMPT.md`: current short-lived task package
- `.env`: experiment-local runtime config only

Preferred managed data layout:

- `data/eval/`: benchmark-side raw snapshots, build scripts, smoke/full artifacts
- `data/train/`: train-side raw snapshots, build scripts, smoke/full artifacts
- `data/sources.yaml`: provenance, split rules, build notes, and benchmark notes

Experiments may preserve baseline-native filenames when needed, but the chosen layout should be documented in the local `README.md` and `data/sources.yaml`.

## Default Lifecycle

Every experiment should support the same startup order:

1. `uv sync`
2. `uv run train.py --dry-run`
3. `uv run train.py --eval-only`
4. only then add or run training

## Shared Expectations

- `train.py` is the executable source of truth.
- `uv run train.py --dry-run` works without MinT credentials when feasible.
- `uv run train.py --eval-only` is the bare benchmark entrypoint.
- `train.py` prints stable `METRIC name=value` lines.
- Experiments stay self-contained; do not import helpers from sibling experiments.

If an experiment supports checkpoint restore, keep the semantics separate:

- same-run resume is directory-driven: rerun the same training command with the same `--log-path`
- fresh eval from a saved checkpoint uses `--eval-only --base-model <sampler_path>`
- `--load-checkpoint-path` is for fresh runs from saved weights, not same-run resume

## Read Order

When these files exist, read them in this order:

- `AGENTS.md`: stable local constraints and sharp edges
- `PROMPT.md`: current task package
- `README.md`: human-facing run instructions
- `train.py`: behavior and artifact truth

## Maintained Experiments

- `dapo-aime24`: direct GRPO on DAPO-Math-17k with frozen AIME 2024 eval
- `fingpt`: Fineval + sentiment eval with an SFT path
- `lawbench`: LawBench benchmark-first eval with a maintained LoRA SFT line
- `chat-dpo`: held-out pairwise chat DPO with preference eval
