# Scaffolds

This directory is the repo-level harness for new MinT cookbook experiments.
Use it to keep experiments self-contained, single-file runnable, and aligned to one shared contract.

## Quick Summary

- `scaffolds/` is the source of truth for repo-wide experiment templates, profiles, naming, and the full artifact naming contract for new experiments.
- Every new experiment starts from the canonical single-file template and the eval profile.
- Every experiment follows the same lifecycle: `--dry-run` -> `--eval-only` -> optional training.
- `uv run train.py --eval-only` stays the bare benchmark confirmation entrypoint; `autoresearch.sh` is the canonical automation wrapper for the current practical line, and `autoresearch.md` should explain that wrapper instead of acting as loose scratch notes.
- New experiments should preserve their local data pipeline: separate `data/eval/` and `data/train/` sides, raw snapshots when applicable, one download or snapshot script per side, one build or adjustment script per side, then smoke and full artifacts.
- Experiments stay self-contained under `experiments/<name>/`; do not import helpers across experiments.
- Baseline scaffolds write a complete eval artifact set (`eval/examples.jsonl`, `eval/predictions.jsonl`, `eval/metrics.json`); the canonical generic SFT path can also emit the shared train/eval append streams when its cadence flags are enabled, while benchmark-specific long-run logging still belongs to research-grade upgrades.

## Canonical Template

Start from one template family only:

- `scaffolds/single_file_experiment/train.py.tpl` - canonical `train.py`; ships the eval path plus a generic SFT baseline distilled from `fingpt` / `lawbench` (shared CLI, generic `run_train`, automatic same-run resume helpers, and optional cadence-driven append streams), so supervised experiments can copy more directly and only replace task adapters or benchmark-specific training details
- `scaffolds/single_file_experiment/README.md.tpl` - experiment README; default top-level flow now follows `experiments/fingpt/README.md`: `Quickstart`, `Live smoke tests`, `Data`, `Current results`, `References`. Experiment-specific benchmark contract notes, run-mode notes, checkpoint semantics, and artifact pointers should usually live as subsections inside `Quickstart` or `Data` rather than as extra top-level sections.
- `scaffolds/single_file_experiment/autoresearch.sh.tpl` - wrapper entrypoint; the default template shape is `SCRIPT_DIR` + named `LOG_PATH` under `artifacts/runs/` + `exec uv run train.py ... "$@" 2>&1`
- `scaffolds/single_file_experiment/autoresearch.md.tpl` - short wrapper protocol: objective, current recipe, signals, mutable surface, frozen contract, recovery, and stopping
- `scaffolds/single_file_experiment/pyproject.toml.tpl` - dependencies; current scaffold pin follows the checked-in MinT compatibility chain: `mindlab-toolkit` + `tinker==0.15.0` + `transformers`
- `scaffolds/single_file_experiment/env.tpl` - local credentials
- `scaffolds/single_file_experiment/data/sources.yaml.tpl` - data provenance

See `scaffolds/single_file_experiment/naming.md` for helper naming and function signatures.

## Lifecycle

Every experiment follows the same eval-first sequence:

1. Copy the canonical template.
2. Customize the five task adapter functions.
3. Run `uv run train.py --dry-run`.
4. Run `uv run train.py --eval-only`.
5. Add training only after the eval contract is stable.

See `scaffolds/lifecycle.md` for the full rationale and taxonomy.

## Profiles

Profiles extend the template. They are documentation, not alternate template trees.

| Profile | When to use |
|---------|-------------|
| `profiles/eval.md` | Default startup for every experiment |
| `profiles/sft.md` | Add supervised fine-tuning after eval is stable |
| `profiles/grpo.md` | Add RL optimization after eval is stable |

Start from `eval.md`. Add the others only when the experiment actually needs them.
There is no promoted `profiles/dpo.md` yet; use `experiments/chat-dpo` as the concrete DPO reference until the shared shape is stable enough to promote.

## Contract Split: Who Owns What

Use this table when deciding where a change belongs and which files must move together in one commit if the repo-wide contract shifts. For logging, artifacts, and stdout markers, also follow `AGENTS.md` -> `When you change logging, artifacts, or stdout contracts`.

| Layer | Primary locations | Owns | Does not own |
|-------|-------------------|------|--------------|
| Scaffold templates | `scaffolds/single_file_experiment/*.tpl` | Baseline directory shape, default eval/train artifact layout, baseline `write_outputs` behavior, comments pointing at profiles | Task-specific prompts, graders, or metric definitions |
| Scaffold docs | `scaffolds/README.md` | Baseline vs research artifact tables, lifecycle, template architecture, and repo-wide README reporting conventions for measured runs | Per-benchmark README prose |
| Naming spec | `scaffolds/single_file_experiment/naming.md` | Stable infrastructure helper names; promoted SFT/RL helper names once two experiments share the same shape | Experiment-local grader names |
| Profiles | `scaffolds/profiles/*.md` | Extension recipes, flag sketches, eval timing notes | Concrete `train.py` implementations |
| Concrete experiment | `experiments/<name>/train.py`, `README.md`, tests | Task adapters, benchmark contract, artifact/logging notes, cadence CLI, optional research JSONL, optional `analysis_manifest.json` + `ANALYSIS_MANIFEST` stdout | Cross-experiment imports |
| `$new-experiment` skill | `.codex/skill/new-experiment/SKILL.md` | Bootstrap workflow plus scaffold-owned artifact/logging guidance for new experiments | Deep long-run upgrade playbooks |
| `$new-experiment-plus` skill | `.codex/skill/new-experiment-plus/SKILL.md`, `references/*.md` | Upgrade workflow, SFT/RL checklists, tinker-cookbook field mapping, throughput/resume guidance | Canonical baseline artifact naming |

Same-commit sync:

- If you change baseline artifact names, baseline eval/train directory layout, required `run.json` keys for new scaffolds, or a promoted helper name, update the relevant templates, `naming.md`, affected profiles, any reference experiment, both skills, `experiments/README.md` if shared policy changes, and `AGENTS.md` if a hard rule changes.
- If you change the preferred data-management workflow for new experiments, update the relevant templates, profiles, `data/sources.yaml.tpl`, both skills, `experiments/README.md`, and `AGENTS.md` together so train/eval split, raw snapshots, and build-script expectations do not drift.
- If you change the promoted `autoresearch.sh` wrapper shape or its documented recipe conventions, update `autoresearch.sh.tpl`, `autoresearch.md.tpl`, `README.md.tpl`, the matching skills/reference docs, and any touched experiment `README.md` / `autoresearch.md` / `autoresearch.sh` in the same commit. Keep the wrapper protocol aligned: objective, current recipe, search signals, mutable surface, frozen contract, recovery path, and stopping rules.
- If you change the default README section flow or measured-run reporting convention for new experiments, update `README.md.tpl`, the relevant profiles (`eval.md`, `sft.md`, `grpo.md`), and any skills or reference docs that repeat that guidance in the same commit.
- If you promote the optional `analysis_manifest.json` / `ANALYSIS_MANIFEST` pattern repo-wide, update the research logging table here, `naming.md`, `AGENTS.md`, and the matching skills in the same commit.
- Keep deep tinker-cookbook-specific field mapping in `.codex/skill/new-experiment-plus/references/logging.md` so this file stays repo-neutral.

## README Reporting Convention

When an experiment `README.md` reports measured runs, keep the evidence bundle together:

- train config
- eval config
- primary result metrics
- wall-clock timing

Specific requirements:

- Eval-only baseline reruns still count as measured runs: record at least the eval config, primary metric, and wall-clock timing.
- For SFT runs, record at least `num_epochs`, `batch_size`, `learning_rate`, and `rank`.
- For non-SFT training runs, record the smallest set of training knobs needed to interpret the result inside that experiment.
- For eval timing, record the relevant `max_concurrent_requests`.
- Use wall clock consistently for reported time.
- When eval timing is batch wall-clock rather than per-run wall-clock, say so explicitly.
- When parallel vs sequential execution materially affects interpretation, say so explicitly.
- Throughput may be included as a supplementary metric, but it does not replace wall-clock timing.

## Template Architecture

The canonical template separates shared framework from task-local benchmark logic.

## Data Workflow

For new experiments, keep dataset management reproducible inside the experiment directory:

1. keep train and eval as separate workflows under `data/train/` and `data/eval/`
2. preserve raw snapshots when local materialization is required
3. keep one download or snapshot script per side
4. keep one build or adjustment script per side
5. materialize a smoke artifact first when a cheap validation slice is practical
6. materialize the full local artifact after the smoke path is validated

The exact filenames may follow the benchmark or upstream baseline, but the split between train and eval, the provenance chain, and the ability to reconstruct the local manifests should stay explicit in `README.md` and `data/sources.yaml`.

### Shared Framework

Keep these layers structurally stable unless the scaffold contract itself changes:

- env loading, CLI, and backend compatibility
- tokenizer cache and loading
- data loading helpers such as `load_records` and `load_jsonl`
- sampling helpers such as `sample_assistant_text`, message-first variants such as `sample_assistant_message`, and `build_generation_prompt_tokens`
- artifact writing helpers such as `write_outputs`, `write_json`, and `write_jsonl`
- metric emission via `emit_metric_lines`
- `main_async` orchestration

### Task Adapters

Customize these five functions for each benchmark:

```python
normalize_eval_rows(path)
normalize_train_rows(path)
build_eval_messages(row)
grade_assistant_text(assistant_text, row)
compute_eval_metrics(predictions)
```

The template ships simple defaults. Replace them with benchmark-specific logic without moving task semantics into shared helpers.

Naming rule: use `assistant_message` / `assistant_text` for chat outputs, and reserve `completion` for token-level suffixes or SFT target spans where Tinker already uses prompt/completion terminology.

### Runtime Flow

`main_async` owns the high-level flow:

1. load data
2. create the service client
3. dispatch to dry run, eval, or training
4. write outputs and emit metrics

The canonical template now includes a generic SFT path after eval is stable.
Benchmark-specific training policy, richer logging, and non-SFT variants remain explicit extension points.

## Source Experiments

The canonical scaffold is distilled from the bundled experiments:

- `experiments/fingpt` - multi-benchmark SFT with merged periodic-eval rows and cadence flags
- `experiments/lawbench` - SFT with periodic eval/checkpointing and prompt lineage
- `experiments/dapo-aime24` - GRPO runtime, checkpointing, and resume
- `experiments/chat-dpo` - pairwise DPO with held-out preference eval and directory-driven same-run resume

## Artifact And Logging Contract

Treat the baseline scaffold and the research-grade upgrade as two layers:

- **Baseline (`$new-experiment`)** gives every new experiment the same small, readable eval snapshot.
- **Research-grade (`$new-experiment-plus`)** keeps that baseline and adds benchmark-specific or high-volume train/eval streams when long runs need better observability, recovery, or throughput than the generic scaffold-owned SFT path.

### Baseline Logging

Every baseline run should be understandable from the same default layout:

```text
<run-dir>/
├── run.json
├── console.log
└── eval/
    ├── examples.jsonl
    ├── predictions.jsonl
    └── metrics.json
```

| File | Purpose |
|------|---------|
| `run.json` | Run metadata and provenance. Write it in two phases: `status:"running"` at start, then `status:"completed"` or `status:"failed"` at the end. Include `args`, raw `argv`, shell-safe `command`, `env`, timing, lightweight git provenance when available, and nested `artifacts.eval` / `artifacts.train` pointers when present. |
| `console.log` | Tee of stdout+stderr for the whole run. |
| `eval/examples.jsonl` | Input-side eval snapshot: rendered `messages` plus the normalized `eval_row` used for that eval. |
| `eval/predictions.jsonl` | Output-side per-example predictions for the same eval snapshot. |
| `eval/metrics.json` | Aggregate metrics for the current eval snapshot. |

Also emit `METRIC name=value` lines to stdout for automation.

Run directory convention:

- `--log-path artifacts/latest` is the scratch-pad default for quick interactive runs.
- Formal runs should use `artifacts/runs/{mode}-{detail}-{model}-{YYYYMMDD}-{HHMMSS}`.
- Smoke runs (directory name contains `smoke`) may also create `artifacts/runs/latest` -> run dir via `prepare_run_dir`.
- Formal runs should not create that symlink, so parallel runs do not fight over one shared pointer.

Baseline naming rules:

- Use phase directories, not file-name prefixes: `eval/` and `train/`.
- Keep the default eval snapshot flat: `eval/examples.jsonl`, `eval/predictions.jsonl`, `eval/metrics.json`.
- Do not add an extra `main/` layer unless one run truly stores multiple named eval snapshots.
- Keep basenames role-based (`examples`, `predictions`, `metrics`, `periodic`, `checkpoints`, `batches`) so the directory already answers whether the file belongs to eval or train.

Optional `analysis_manifest.json` plus stdout `ANALYSIS_MANIFEST path=...` is not part of the default baseline template until the repo promotes it.

The canonical generic SFT path keeps that eval snapshot as the required
minimum, then opt-ins to `train/checkpoints.jsonl`, `eval/periodic.jsonl`,
streamed `train/metrics.jsonl`, and optional `train/batches.jsonl` only when
the matching cadence flags are enabled.

### Research-Grade Logging

Keep the baseline eval snapshot above. Add the following only when long runs must stay inspectable and recoverable:

| File | Purpose |
|------|---------|
| `train/metrics.jsonl` | Streamed per-step train diagnostics. SFT usually logs loss/lr/timing/throughput; RL usually logs reward/accuracy/format/datums/timing/throughput. |
| `train/batches.jsonl` (optional) | SFT prompt/batch lineage when reviewers need to see which prompts reached a step. |
| `eval/periodic.jsonl` | SFT periodic eval history keyed by train step; may include bounded prediction samples. |
| `train/rollouts.jsonl` | RL rollout records with per-group rewards, correctness, advantages, and text preview. |
| `train/failures.jsonl` + `train/failures.log` | RL failure logs in structured and human-readable form. |
| `eval/metrics.jsonl` | RL periodic eval history; step-specific details can live under `eval/steps/step-NNNN/`. |
| `train/checkpoints.jsonl` | Checkpoint registry and restore source of truth. Rows should usually carry a human-readable logical `name`. SFT rows should carry `state_path`, `step`, and the next loop position (`epoch`, `batch`); RL rows should carry `state_path`, `completed_steps`, and `next_step`. `sampler_path` stays alongside those fields when later eval needs a durable sampler export, but the runtime save names behind `state_path` and `sampler_path` should stay distinct. |
| `analysis_manifest.json` (optional) | End-of-run AI/script index when the experiment explicitly implements it. |

Research-grade rules:

- Reset run-scoped append streams at the start of each fresh run unless the run is intentionally resuming: truncate/create the streams enabled by the active cadence flags, and remove disabled stream files left behind by an older run in the same directory so the file set stays exact.
- Overwrite summary files such as `run.json`, `eval/examples.jsonl`, `eval/predictions.jsonl`, and `eval/metrics.json` on each run as usual.
- Prefer `run.json` provenance fields such as `args`, raw `argv`, and shell-safe `command`.
- For SFT, mirror the information shape of upstream tinker-cookbook supervised logging while keeping this repo's local JSONL names.
- For RL, keep the same phase-first naming and borrow record shape / loop structure from `experiments/dapo-aime24`.
- The scaffold default keeps same-run resume directory-driven across both SFT and GRPO profiles whenever the current run directory actually has a resumable checkpoint row: rerun the same training command with the same run directory (exposed as `--log-path`). `train.py` auto-reads the latest resumable `state_path` from that run's `train/checkpoints.jsonl`, restores optimizer plus loop state from the matched row through a fresh LoRA training client and `load_state_with_optimizer(...)`, keeps the run-scoped JSONL, and appends `console.log`. No dedicated `--resume-from` flag is reserved for this default. SFT reads loop state as `(step, epoch, batch)`; GRPO reads it as `(completed_steps, next_step)` from the same kind of checkpoint row.
- `--load-checkpoint-path` starts a fresh run from saved weights only. It takes effect only when the current run directory has no resumable checkpoint row, and it starts fresh append-only logs plus a fresh `console.log`. The generic scaffold SFT path only creates `train/checkpoints.jsonl` when checkpointing is actually enabled, so the minimal train-and-final-eval path can stay small while still supporting later opt-in resume.
- `experiments/dapo-aime24` uses the same directory-driven training resume and `--load-checkpoint-path` behavior as the SFT experiments; `--eval-only` uses `--base-model` with a `sampler_path`. Prefer the checkpoint row dict from `get_last_resumable_checkpoint(...)` over experiment-local `ResumeLoopState` wrappers in new GRPO code.
- Do not gate automatic same-run resume on run-scoped append-only streams being exactly aligned to the last checkpoint row. SFT streams (`train/metrics.jsonl`, `train/batches.jsonl`, `eval/periodic.jsonl`) and RL streams (`train/metrics.jsonl`, `train/rollouts.jsonl`, `train/failures.jsonl`, `eval/metrics.jsonl`) may slide past the last resumable checkpoint without that being a resume failure.
- Do not treat checkpoint path text as the scaffold's restore oracle. The checkpoint row in `train/checkpoints.jsonl` is the restore contract; parsing step numbers from checkpoint names is experiment-local compatibility logic only.

Checkpoint row guidelines:

- SFT rows should record completed optimizer `step` plus the next loop position as `epoch` and `batch`, matching the tinker-cookbook supervised resume shape.
- RL rows should record `completed_steps` and `next_step`, matching the tinker-cookbook RL resume shape.
- Rows should usually carry one logical `name`, while the underlying runtime save names for the resumable state and the durable sampler export stay distinct, for example `<name>-state` and `<name>-sampler`.
- Final checkpoint rows may also carry `final: true`, but that flag does not replace the loop-state fields above.

## Runtime Naming

The canonical scaffold still uses current `tinker` runtime names inside `train.py`:

- `import tinker` / `from tinker import types`
- `TINKER_API_KEY` / `TINKER_BASE_URL`
- `pyproject.toml.tpl` currently pins `mindlab-toolkit` plus `tinker==0.15.0` to match the compatible MinT stack used by the maintained experiments, even though the scaffold code still keeps `tinker` runtime names until the repo deliberately promotes a mint-first template

Concrete experiments may intentionally use `mint` / `MINT_*` instead, but the runtime choice should stay self-consistent within that experiment's `README.md`, CLI flags, and backend compatibility section.
Keep backend-specific code concentrated in that compatibility section so future migration stays localized.
