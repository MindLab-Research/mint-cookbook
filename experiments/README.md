# Experiments

Each subdirectory under `experiments/` is an independent, self-contained reproduction.
You should be able to enter one directory, read local `AGENTS.md`, then `PROMPT.md` when present, then `README.md`, run `uv sync`, and validate the benchmark path without reading the rest of the repo.

For live MinT-backed eval, training, or smoke tests, run the documented command from the target experiment directory with the local `.env` or shell environment that fits your machine.
Personal routing details such as host aliases, repo paths, sync tools, and machine-specific layout do not belong in the checked-in repo contract.

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

The shared live-runtime template lives at the repo root `.env.example`; from an experiment directory, copy `../../.env.example` to `.env` when you need local credentials.
New scaffold-derived experiments should use `import mint`, `MINT_BASE_URL`, `MINT_API_KEY`, and `--mint-timeout` by default; do not start fresh experiment code from a tinker-first runtime skeleton unless that exception is documented locally.
When you are migrating older code that already assumes a local module name `tinker`, the narrow bridge is `import mint as tinker` plus MinT-facing endpoint and auth wiring through `MINT_BASE_URL` and `MINT_API_KEY`; treat that alias as a migration bridge, not as the default style for new experiment code.
Maintained experiments should ship `AGENTS.md` so developers and AI agents can see the benchmark anchor, wrapper distinction, sharp edges, and live-runtime bootstrap path before touching `train.py`.
The maintained experiment set itself is tracked in `experiments/maintained.json`; update that registry when the maintained set or its repo-level summary changes.
In maintained experiment `README.md` files, `## Current results` should start with a `Status:` line: use `Status: \`placeholder\`` until a maintained run is actually checked, switch it to `Status: \`checked\`` only when the reported run is actually checked, and then either record checked runs or say explicitly that no maintained reportable run is checked in yet; do not leave a raw `TODO` placeholder there.
For repo-wide maintenance work across those experiments, re-read `experiments/maintained.json`, root `PROMPT.md`, and the touched experiment `AGENTS.md` / `README.md` first, then validate from the experiment directory.

Preferred managed data layout:

- Single-dataset default: `data/eval/` for benchmark-side raw snapshots, build scripts, and smoke/full artifacts, plus `data/train/` for train-side raw snapshots, build scripts, and smoke/full artifacts
- Multi-dataset alternative: a documented dataset-family layout under `data/`, for example `data/<family>/...` plus `data/benchmarks/<suite>/...`, when that matches the real upstream split better than one shared `train/` and `eval/` tree
- `data/sources.yaml`: provenance, split rules, build notes, and benchmark notes

Experiments may preserve baseline-native filenames when needed, but the chosen layout should be documented in the local `README.md` and `data/sources.yaml`, and download/build scripts should stay under `data/`.

## Default Lifecycle

Every experiment should support the same startup order:

1. `uv sync`
2. `uv run train.py --dry-run --eval-data <smoke_eval_path>`
3. `uv run train.py --eval-only --eval-data <full_eval_path>`
4. only then add or run training

When those steps need a live MinT backend, run them from the target experiment directory with the credentials and routing that fit your own environment.

## Shared Expectations

- `train.py` is the executable source of truth.
- `uv run train.py --dry-run --eval-data <smoke_eval_path>` works without MinT credentials when feasible.
- `uv run train.py --eval-only --eval-data <full_eval_path>` is the canonical benchmark entrypoint.
- `train.py` prints stable `METRIC name=value` lines.
- Experiments stay self-contained; do not import helpers from sibling experiments.

If an experiment supports checkpoint restore, keep the semantics separate:

- same-run resume is directory-driven: rerun the same training command with the same `--log-path`
- fresh eval from a saved checkpoint uses `--eval-only --eval-data <full_eval_path> --base-model <sampler_path>`
- `--load-checkpoint-path` is for fresh runs from saved weights, not same-run resume

## Read Order

When these files exist, read them in this order:

- `AGENTS.md`: stable local constraints and sharp edges
- `PROMPT.md`: current task package
- `README.md`: human-facing run instructions
- `train.py`: behavior and artifact truth

## Maintained Experiments

<!-- maintained-experiments:start -->
- `chat-dpo`: pairwise chat DPO with held-out preference eval
- `dapo-aime`: direct GRPO on DAPO-Math-17k with an AIME 2024 benchmark plus AIME 2025/2026 eval manifests
- `fingpt`: FinGPT reproduction scaffold with Fineval + sentiment eval and an SFT path
- `lawbench`: LawBench benchmark-first scaffold with a maintained LoRA SFT line
<!-- maintained-experiments:end -->
