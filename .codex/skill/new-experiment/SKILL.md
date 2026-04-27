---
name: new-experiment
description: |
  Create a new self-contained MinT cookbook experiment under `experiments/`.

  Covers the full eval-first lifecycle: scaffold the experiment directory,
  establish the eval baseline with the scaffold-owned eval artifact set,
  and add basic SFT or GRPO training by following `scaffolds/profiles/`.

  This skill materializes experiments from the repo-level `scaffolds/`
  directory. It does not own a separate template tree.
argument-hint: "<experiment-name>"
---

# New Experiment

Create a new experiment directory, establish an eval baseline with the scaffold-owned artifact layout, and optionally add basic training.

## What this skill covers

1. **Create** the experiment directory from `scaffolds/` templates
2. **Eval baseline** — customize five adapter functions, `--dry-run`, `--eval-only`
3. **Basic training** — follow `scaffolds/profiles/sft.md` or `scaffolds/profiles/grpo.md` to add SFT or GRPO

## What this skill does NOT cover

Benchmark-specific or high-volume research infrastructure beyond the generic scaffold-owned SFT path: streaming minibatches, tail early-stop, throughput optimization, richer failure traces, AI-native indices, and advanced long-run operability. For those, use `$new-experiment-plus`.

**Logging boundary:** `$new-experiment` should fully describe and materialize the scaffold-owned baseline artifact contract from `scaffolds/README.md`: `run.json`, `console.log`, `eval/examples.jsonl`, `eval/predictions.jsonl`, `eval/metrics.json`, plus `METRIC` stdout. Treat that as the default readable eval snapshot for every new experiment. The promoted generic SFT path in the canonical template may additionally emit `eval/periodic.jsonl`, `train/metrics.jsonl`, `train/batches.jsonl`, or `train/checkpoints.jsonl` when the matching cadence flags are enabled, but newly scaffolded experiments should not *require* those files unless the user explicitly wants that training path or longer-run logging.

## When this skill applies

- User asks to create a new experiment
- User says "add experiment for X" or "reproduce X"
- User wants to add basic SFT or GRPO training to an existing eval baseline

Use `$project-harness-bootstrap` first when the root-level harness or `experiments/README.md`
does not exist yet.
Use `$helper-contract-alignment` when the task is repo-wide reconciliation of
existing helper drift rather than creating one new experiment.

## Source of truth

Treat `scaffolds/` as the only experiment-template source of truth.
This skill should orchestrate those scaffold assets directly through the canonical template and `scaffolds/profiles/`, not through any preset layer or private alternatives.

Primary startup sources:

- `AGENTS.md` (router + **When you change logging, artifacts, or stdout contracts** whenever you touch eval/train outputs or stdout markers)
- `scaffolds/README.md`
- `scaffolds/single_file_experiment/train.py.tpl`
- `scaffolds/single_file_experiment/README.md.tpl`
- `scaffolds/single_file_experiment/autoresearch.sh.tpl`
- `scaffolds/single_file_experiment/autoresearch.md.tpl`
- `scaffolds/single_file_experiment/pyproject.toml.tpl`
- `scaffolds/single_file_experiment/env.tpl`
- `scaffolds/single_file_experiment/data/sources.yaml.tpl`
- `scaffolds/single_file_experiment/naming.md`
- `scaffolds/profiles/eval.md`

Later training references:

- `scaffolds/profiles/sft.md`
- `scaffolds/profiles/grpo.md`

Do not generate `train.py` from skill-local `train_sft.py.template`, `train_rl.py.template`,
or `train_custom.py.template`. Those patterns belong in the canonical scaffold and profiles.

## Keeping templates, skills, and contracts aligned

When a user request changes **baseline** behavior (default artifacts, baseline eval/train directory layout, `run.json` shape for new scaffolds, a helper name that new experiments should copy, or the default README flow / measured-run reporting convention), treat **`scaffolds/` as source of truth** and update **every** layer that encodes the same contract in **one commit**: the relevant `scaffolds/single_file_experiment/*.tpl` files, `scaffolds/single_file_experiment/naming.md` if names are involved, `scaffolds/profiles/*.md` if the extension recipe changes, this skill's wording, **`$new-experiment-plus`** only where it references the scaffold contract, `experiments/README.md` if shared policy text changes, and **`AGENTS.md`** if a hard rule changes. The canonical **split table** and the **logging/artifact sync checklist** live in `scaffolds/README.md` (**Contract split: who owns what**) and **`AGENTS.md` -> *When you change logging, artifacts, or stdout contracts***.

## Quick repo summary

Keep this repo-level summary in mind when creating or updating experiments:

- `scaffolds/` is the source of truth for repo-wide experiment templates, profiles, naming, and the full artifact naming contract for new experiments.
- Every experiment starts eval-first from the canonical single-file scaffold, then adds training only after `--dry-run` and `--eval-only` are stable.
- Each experiment stays self-contained under `experiments/<name>/` and should not import helpers from another experiment.
- `train.py` remains the readable source of truth, must support `--eval-only`, should support `--dry-run` when possible, and must emit `METRIC name=value` lines.
- `uv run train.py --eval-only` is the canonical bare benchmark entrypoint; `autoresearch.sh` is the canonical automation wrapper for the current practical line around that path.
- Baseline scaffold outputs should stay small and readable: `run.json` (two-phase: `status:"running"` at start, `status:"completed"/"failed"` at end), `console.log` (TeeStream stdout+stderr mirror), and the `eval/` snapshot trio: `examples.jsonl`, `predictions.jsonl`, `metrics.json`. Smoke runs get an `artifacts/runs/latest` symlink via `prepare_run_dir`; formal runs do not. The canonical template's generic SFT path can optionally add the scaffold-owned `train/metrics.jsonl` / `eval/periodic.jsonl` / `train/checkpoints.jsonl` / `train/batches.jsonl` family via cadence flags; benchmark-specific long-run logging beyond that is deferred to `$new-experiment-plus`.
- The default scaffold README flow is `Quickstart`, `Data`, `Current results`, `Outputs`, `References`.
- Even eval-only baselines count as measured runs in `README.md`: record the checked eval config, primary metric, and wall-clock timing together.
- If a reusable contract changes, update the matching templates, docs, naming specs, profiles, skills, experiment docs, and `AGENTS.md` in the same commit.

## House style for scaffold code and docs

When materializing or standardizing scaffold-facing files, follow this default house style unless a file already has a stronger established convention:

- Use explicit top-level responsibility sections in scaffold Python with comment headers such as `# ===== CLI =====`.
- Leave exactly one blank line after each section header in scaffold Python.
- Keep normal Python top-level spacing: two blank lines between top-level `def` / `async def` / `class` blocks.
- Order scaffold code from shared framework to task-local logic: constants/config -> env loading -> backend compatibility -> CLI -> infrastructure helpers -> task adapters -> runtime entrypoints -> artifact writing -> entrypoint.
- Keep task-specific prompts, graders, reward logic, and dataset mapping inside the adapter layer instead of mixing them into shared helpers.
- Match helper names to the returned layer: keep `sample_assistant_text(...)` for text-returning eval helpers, and use `sample_assistant_message(...)` when the sampler path is message-first and text is extracted later (per `scaffolds/single_file_experiment/naming.md`).
- Keep scaffold Markdown concise and operational: short sections, explicit commands, explicit outputs, and benchmark-facing constraints stated directly.
- Prefer behavior-preserving cleanup when standardizing templates; avoid sneaking in logic changes during formatting passes.
- Treat `scaffolds/single_file_experiment/train.py.tpl`, `README.md.tpl`, and `autoresearch.md.tpl` as the style references for future scaffold files unless the repo deliberately revises them.

## Usage

```text
$new-experiment <name>
```

Examples:

```text
$new-experiment lawbench
$new-experiment finqa
$new-experiment gsm8k-chat-eval
```

Every new experiment starts eval-first.
Only add training after the eval contract is stable.

## What gets created

```text
experiments/<name>/
|-- pyproject.toml
|-- README.md
|-- train.py
|-- .env
|-- autoresearch.sh
|-- autoresearch.md
`-- data/
    |-- eval/
    |   |-- raw/                                # preferred when local materialization is needed
    |   |-- download_*.py or snapshot_*.py     # preferred
    |   |-- build_*.py or adjust_*.py          # preferred
    |   |-- smoke.jsonl                        # preferred when a cheap slice exists
    |   `-- full.jsonl
    |-- train/                                 # optional until training is enabled
    |   |-- raw/                                # preferred when local materialization is needed
    |   |-- download_*.py or snapshot_*.py      # preferred
    |   |-- build_*.py or adjust_*.py           # preferred
    |   |-- smoke.jsonl                         # preferred when a cheap slice exists
    |   `-- full.jsonl
    `-- sources.yaml
```

Concrete train/eval artifacts stay local to the experiment as required by the repo contract,
but their exact names should follow the selected benchmark instead of being forced
into a second generic template family.

## Instructions

1. Parse the experiment name and benchmark scope from the user's request.
2. Create `experiments/<name>/`.
3. Materialize experiment files from the repo-level scaffold templates listed above.
4. Start from `scaffolds/single_file_experiment/train.py.tpl` for every experiment.
5. Always read `scaffolds/profiles/eval.md` first.
6. If the benchmark uses tools or multi-turn episodes, keep that benchmark-specific loop logic local to the experiment while still starting from `scaffolds/profiles/eval.md`.
7. Keep the required section order, helper names, and artifact names from the canonical scaffold.
8. Data paths are passed via `--train-data` and `--eval-data` CLI args; do not hardcode data path constants.
9. Prefer the repo-wide data workflow for new experiments: separate `data/eval/` and `data/train/`, preserve raw snapshots when local materialization is needed, keep one download or snapshot script per side, keep one build or adjustment script per side, and materialize smoke before full when a cheap slice exists.
10. Keep `train.py` single-file and self-contained. Do not import helpers from another experiment.
11. `train.py` must support `--eval-only`.
12. `train.py` should support `--dry-run` whenever local data and prompt formatting can be validated without credentials.
13. `train.py` must print `METRIC name=value` lines.
14. Keep the scaffold-owned **baseline eval artifact set** (`run.json`, `console.log`, `eval/examples.jsonl`, `eval/predictions.jsonl`, `eval/metrics.json`) as the default readable snapshot for new experiments. The promoted generic SFT path may opt into `train/metrics.jsonl`, `eval/periodic.jsonl`, `train/checkpoints.jsonl`, and `train/batches.jsonl` via the scaffold cadence flags; benchmark-specific or heavier long-run logging still belongs to `$new-experiment-plus` (see `scaffolds/README.md` artifact contract).
15. Keep generated scaffold docs concise, operational, and aligned with the canonical template wording, including the default README section flow and measured-run reporting convention.
16. Update `experiments/README.md` only when the shared experiment index needs to mention the new experiment.
17. Do not add SFT or GRPO code until the eval contract, prompt shape, parser, grader, and metric are stable.
18. If a reusable pattern emerges, promote it back into `scaffolds/` rather than creating skill-private drift.

## Runtime naming policy

This repo is a MinT cookbook, but new experiments should always start on the tested `tinker` runtime surface. The standard development sequence is: get the experiment working on tinker first (dry-run → eval-only → training), then migrate to mint once the tinker path is stable.

Scaffolded `train.py` files should use:

- `import tinker`
- `from tinker import types`
- `TINKER_API_KEY`
- `TINKER_BASE_URL`

Do not start a new experiment directly on `mint`. Keep backend-specific code inside one clearly labeled backend compatibility section so migration stays localized, as described in `scaffolds/single_file_experiment/naming.md` and the canonical `train.py.tpl`.

## Startup principle

- Implement `--eval-only` first and make it run on the frozen benchmark path.
- Run base-model evals early to establish the real starting point.
- Record the initial metric and failure modes before adding training.
- Only then follow `scaffolds/profiles/sft.md` or `scaffolds/profiles/grpo.md` to add training.

## pi-autoresearch expectations

- `autoresearch.sh` must stay a thin wrapper around the project command, usually `uv run train.py`.
- Use the current wrapper shape: resolve `SCRIPT_DIR`, `cd` into the experiment, define a named `OUTPUT_DIR` under `artifacts/runs/`, then `exec uv run train.py ... "$@" 2>&1`.
- Before training exists, pin `--eval-only` in `autoresearch.sh`; when a train-and-eval recipe becomes canonical, pin that recipe directly in the wrapper instead of relying on env-var defaults.
- `train.py` must emit structured `METRIC name=value` lines.
- Default primary metric should be the benchmark's canonical metric, usually `eval_accuracy` or `eval_success_rate`.
- `autoresearch.md` should stay a short protocol for the current wrapper: objective, current recipe, signals, mutable surface, frozen contract, recovery path, and stopping rules.
- When wrapper flags, output-dir naming, or the canonical recipe changes, update `autoresearch.sh`, `autoresearch.md`, and the experiment `README.md` in the same commit.

## After scaffolding

Tell the user to do the following next:

- replace placeholder local data artifacts with real eval data and any explicit target-method-aligned or benchmark-aligned manifests
- update `data/sources.yaml` with real provenance and merge rules
- customize task adapters, prompt formatting, parser logic, and grading logic in `train.py`
- run `cd experiments/<name> && uv sync && uv run train.py --dry-run`
- run `cd experiments/<name> && uv run train.py --eval-only` to establish the frozen runnable benchmark baseline
- to add basic training, follow `scaffolds/profiles/sft.md` or `scaffolds/profiles/grpo.md`
- for research-grade upgrades (resume, checkpointing, throughput), use `$new-experiment-plus`
