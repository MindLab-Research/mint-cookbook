# File Upgrade Map

Use this file when `new-experiment-plus` needs to strengthen a concrete experiment without redefining the baseline scaffold contract.

Layering reference: see `scaffolds/README.md` -> `Contract split: who owns what` for which changes belong in templates, `naming.md`, profiles, skills, or the concrete experiment. Read this file as an upgrade checklist layered on top of `$new-experiment`, not as a second template spec.

## Start Mode

### If Starting From Scratch

1. Reuse the same top-level experiment directory shape as `$new-experiment`:
   - `train.py`
   - `README.md`
   - `data/eval/` and `data/train/`, with raw snapshots, side-specific download or snapshot scripts, side-specific build or adjustment scripts, then smoke and full artifacts when the local pipeline needs them
   - `data/sources.yaml`
   - `autoresearch.sh`
   - `autoresearch.md`
2. Strengthen those files instead of inventing a different repo contract.

### If Upgrading In Place

Keep the existing experiment directory and benchmark contract unless the user explicitly wants a contract change.

## Per-File Upgrade Checklist

### `train.py`

This is where most of the upgrade happens. Keep the baseline eval snapshot intact and add the extra operability layers around it.

Typical additions:

- optional AI-native index: `analysis_manifest.json` + stdout `ANALYSIS_MANIFEST path=...` + `run.json` output pointer
- per-step timing such as `step_time_seconds`
- token throughput such as `tokens_per_second`, `num_tokens`, and `num_loss_tokens`
- training progress percentage
- structured step-level metrics in `train/metrics.jsonl`
- for SFT, the same logging shape as `tinker-cookbook/tinker_cookbook/supervised/train.py`, mapped onto local JSONL names
- explicit pre-step vs post-step eval semantics, documented in `README.md`
- run-scoped JSONL (`eval/periodic.jsonl`, `train/checkpoints.jsonl`, streamed `train/metrics.jsonl`, optional `train/batches.jsonl`, RL streams) cleared at fresh-run boundaries
- checkpoint restore flow with split semantics when long runs matter: directory-driven same-run resume versus fresh-run `--load-checkpoint-path`
- logical checkpoint `name` plus distinct runtime save names behind `state_path` and `sampler_path` when one row exports both artifacts
- throughput knobs clearly separated from algorithm knobs

### `README.md`

A plus experiment should usually add:

- benchmark contract
- dataset / provenance contract
- run commands for train, eval-only, and dry-run
- measured-run result sections that keep the evidence bundle together: train config when relevant, eval config, result metrics, and wall-clock timing
- explanation of throughput knobs vs algorithm knobs
- `Outputs / logging` section: files written, cadence, and eval timing semantics
- explicit note about which comparisons remain fair if settings change

Do not rewrite the baseline scaffold contract here; extend it with the extra research-grade streams the experiment actually ships.

### `autoresearch.sh`

Keep this file thin and reproducible:

- keep the current wrapper shape: `SCRIPT_DIR` + `cd`, named `OUTPUT_DIR` under `artifacts/runs/`, then `exec uv run train.py ... "$@" 2>&1`
- surface important throughput knobs as explicit CLI flags in the wrapper recipe; use env vars only when they materially help local operations
- keep wrapper names aligned with the experiment CLI
- document defaults in `README.md` and `autoresearch.md`
- when the wrapper recipe changes, update `autoresearch.sh`, `autoresearch.md`, and `README.md` together in the same commit

### `autoresearch.md`

Use this file to lock the research session framing:

- objective
- primary search signal
- current wrapper recipe
- mutable surface
- frozen benchmark contract
- recovery and confirmation path
- run budget and stopping rules

For plus experiments, this file should describe the current training harness, not only the dataset or task.

### `data/sources.yaml`

Treat provenance as part of the benchmark contract. Record:

- upstream datasets
- raw snapshot locations
- download or snapshot scripts
- build or adjustment scripts
- local materialization steps
- train/eval split rules
- deduplication or filtering that affects comparability

## After Editing: Self-Check

Before treating a plus upgrade as done, walk `skills/new-experiment-plus/SKILL.md` -> `Research-grade operability` and confirm:

- `README.md` keeps the baseline scaffold section flow unless there is a clear experiment-specific reason not to.
- `README.md` starts `Current results` with `Status: \`placeholder\`` until a checked run exists, and only switches to `Status: \`checked\`` when the reported run is actually checked.
- `README.md` keeps measured-run evidence bundles together; eval-only baselines still record eval config, primary metrics, and wall-clock timing.
- `README.md` has an `Outputs / logging` section with files written, cadence flags, and pre-step vs post-step eval semantics when relevant.
- `train.py` truncates the same run-scoped JSONL set on fresh starts that the docs promise.
- `run.json` carries the provenance you promised: `args`, `argv`, shell-safe `command`, and any optional git metadata you adopted.
- If you ship AI-native indices, `analysis_manifest.json` exists, stdout prints `ANALYSIS_MANIFEST path=...`, and `README.md` documents the read order.
- `autoresearch.md` still matches `autoresearch.sh` if wrapper flags or env overrides changed.
- The baseline readable eval snapshot still matches `scaffolds/README.md`; this overlay should add research-grade layers, not fork the baseline naming.

## What Not To Overcomplicate

Even in a plus experiment:

- keep the repo contract recognizable
- do not introduce shared cross-experiment imports unless the user explicitly wants infrastructure reuse
- do not add concurrency or output/logging complexity that the experiment cannot explain in docs and logs

## Practical Rule Of Thumb

- If a feature changes how people interpret results, document it in `README.md` and usually in logs.
- If a feature changes how a run survives or restores from checkpoints, put it in `train.py` and usually in a machine-readable output or registry, and keep directory-driven same-run resume separate from fresh-run `--load-checkpoint-path`.
- If a checkpoint row writes both `state_path` and `sampler_path`, keep one logical `name` in the registry but do not reuse that same runtime save name for both exports.
- If a feature changes throughput without changing the algorithm, make that split explicit in names and docs.
