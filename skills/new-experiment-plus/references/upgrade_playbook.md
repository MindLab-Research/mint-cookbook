# Upgrade Playbook

Use this file as an overlay on top of the canonical scaffold when the user wants a research-ready harness. Start from the `$new-experiment` baseline first, then layer long-run operability on top.

## First Decision

- If there is no experiment directory yet, create the directory with `$new-experiment`-level structure first, then layer upgrades on top.
- If an experiment already exists, preserve its benchmark contract and upgrade the harness in place.

## What Makes An Experiment "Plus"

A `new-experiment-plus` experiment usually adds most of the following:

- explicit benchmark contract in `README.md`
- explicit data / provenance contract in `README.md` and `data/sources.yaml`
- preserved local train/eval reconstruction workflow: raw snapshots when needed, side-specific download or snapshot scripts, side-specific build or adjustment scripts, then smoke and full artifacts
- measured-run result sections that keep config, metrics, and wall-clock timing together
- stable `--dry-run` and `--eval-only` behavior
- same-run resume and fresh checkpoint-loading paths documented in code and docs
- periodic saves and a machine-readable checkpoint registry when long runs matter
- structured train/eval outputs instead of a single summary file
- enough console and JSONL metadata to explain throughput changes and failures

It does **not** replace the baseline scaffold contract. The baseline readable eval snapshot still comes from `scaffolds/README.md` and `$new-experiment`.

## Upgrade Order

1. Lock train and eval file paths and the side-specific raw/download/build workflow.
2. Lock metric names and evaluation rules.
3. Add observability: per-step timing, token throughput, progress tracking, structured step metrics.
4. Add checkpoint restore support with split semantics: directory-driven same-run resume versus fresh-run `--load-checkpoint-path`, with `train/checkpoints.jsonl` as the restore source of truth.
5. Only then tune throughput or concurrency.

Do not introduce train-loop complexity before the experiment contract is stable.

Basic LR scheduling (`--lr-schedule`), periodic eval (`--eval-every`), and periodic checkpoint (`--save-every`) already belong to `scaffolds/profiles/sft.md` and the SFT-style pairwise path in `scaffolds/profiles/dpo.md`. This playbook starts after that layer: it adds per-step timing and throughput logging, non-blocking background eval, checkpoint-aware sampler snapshots, and richer checkpoint registries.

## Artifact Expectations

Baseline `$new-experiment` work stays on the scaffold-owned readable eval snapshot in `scaffolds/README.md` (`eval/examples.jsonl`, `eval/predictions.jsonl`, `eval/metrics.json`, plus `run.json` / `console.log`). This section describes the extra research-grade layers that may be added after that baseline is stable.

Keep the baseline README flow (`Quickstart`, `Fast contract tests`, `Live smoke tests`, `Data`, `Current results`, `References`) unless the experiment has a strong reason to diverge. Start `Current results` with `Status: \`placeholder\`` until a checked run exists, and switch to `Status: \`checked\`` only when the reported run is actually checked. Measured runs should keep the evidence bundle together: train config when relevant, eval config, result metrics, and wall-clock timing. Eval-only baselines still count as measured runs.

Where to edit together:

- `scaffolds/README.md` -> `Contract split: who owns what`
- `skills/new-experiment-plus/SKILL.md` -> `Research-grade operability`
- `references/file_upgrade_map.md` -> `After Editing: Self-Check`

Optional AI-native index pattern:

- `analysis_manifest.json`
- stdout `ANALYSIS_MANIFEST`

### SFT

Align local files with the shape of `tinker-cookbook/tinker_cookbook/supervised/train.py`:

- per-step merged scalars in `train/metrics.jsonl`
- eval history keyed by step in `eval/periodic.jsonl`
- checkpoint registry in `train/checkpoints.jsonl` plus optional `state_path.txt`
- optional timing or trace sidecars only when profiling matters
- `METRIC` lines for automation

For MinT-style same-run resume, keep the control flow uniform with the live
repo experiments: create a fresh LoRA training client first, then call
`load_state_with_optimizer(...)`. Use `train/checkpoints.jsonl` to restore loop
state; do not make checkpoint-path parsing the primary restore contract. When
one logical checkpoint exports both resumable state and eval-ready sampler
artifacts, keep one row `name` and back it with distinct runtime save names,
for example `<name>-state` and `<name>-sampler`.

Document whether periodic eval uses pre-step or post-step weights relative to `optim_step` so numbers stay comparable across runs. In-repo reference: `experiments/lawbench`.

### RL

For research-grade RL experiments, keep the same baseline snapshot and add explicit output files / streams such as:

- `eval/metrics.json`
- `train/metrics.jsonl`
- `train/checkpoints.jsonl`
- `train/rollouts.jsonl`
- `train/failures.jsonl` plus `train/failures.log`
- `eval/examples.jsonl`
- `eval/predictions.jsonl`
- `eval/metrics.jsonl` for periodic eval history
- failure logs in both human-readable and JSONL form
- rollout traces when RL behavior debugging matters

Append-only registries should still be run-scoped: truncate or rotate them when starting a fresh run or when using `--load-checkpoint-path`; preserve them only when the same training command is deliberately continuing the same registry in the same run directory.

## Benchmark Contract Expectations

Write down what must stay fixed across comparisons:

- eval file path
- prompt shape
- parsing rules
- chosen-answer rule
- grading rule
- metric names

If any of these changes, say so explicitly rather than treating the result as directly comparable.

## When Not To Use This Skill

Stay with `$new-experiment` if the user only wants:

- a quick scaffold
- a toy baseline
- a one-off smoke test
- a minimal train/eval script with no long-run infrastructure

Stay with `scaffolds/profiles/sft.md`, `scaffolds/profiles/dpo.md`, or `scaffolds/profiles/grpo.md` if the user only wants basic training added on top of the baseline and does not need extra operability layers.
