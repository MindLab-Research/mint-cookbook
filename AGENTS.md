# Agent Guide

This repository is a monorepo of independent MinT paper and method reproductions.
The default work unit is one directory under `experiments/`.
Current work may focus on one topic such as AIME, but the repo itself is not AIME-only.

## Context layers

Treat agent context as a layered system with different scopes and half-lives.

- **Global instructions / developer prompt** - long-lived reasoning discipline and evidence standards
- **This file (`AGENTS.md`)** - repo-wide structure, lifecycle, and hard rules that should stay stable across tasks
- **`experiments/<name>/AGENTS.md`** - experiment-local truth such as benchmark contracts, canonical commands, sharp edges, and wrapper/runtime differences; keep this stable within one experiment and do not duplicate repo-wide policy there
- **`PROMPT.md`** - the current task package; use the repo root `PROMPT.md` for repo-wide work and `experiments/<name>/PROMPT.md` for single-experiment work
- **Harness-mode prompt** - disposable unattended-execution policy injected by the runner

Keep these layers separate:

- do not turn `AGENTS.md` into a task tracker
- do not turn `PROMPT.md` into a long-lived knowledge base
- do not copy experiment-local facts into the repo root unless they truly become cross-experiment policy

## Default workflow

1. Enter the target experiment directory.
2. Read that experiment's `AGENTS.md` if it exists.
3. Read the active `PROMPT.md` for the current task package if one is provided.
4. Read that experiment's `README.md`.
5. Read that experiment's `train.py`.
6. If the task starts from only a benchmark name, identify the target method first; default to the latest benchmark-tested method unless the user wants an exact alternative.
7. Validate with `uv run train.py --dry-run --eval-data <smoke_eval_path>` (or the experiment's equivalent explicit smoke eval manifest) before making logic changes.
8. When an experiment has smoke train/eval data or cheap download scripts for those smoke slices, fetch or materialize the smoke data first and use it to validate the fast local path before touching the full dataset.
9. Use `uv run train.py --eval-only --eval-data <full_eval_path>` to lock the benchmark path before changing training.
10. Only after the eval path is stable should you add or modify the training route that pushes the runnable baseline toward the target method effect.

## Hard rules

- Keep each experiment self-contained.
- Do not import helpers from another experiment.
- Prefer editing one experiment at a time unless the task is explicitly repo-wide.
- Maintained experiments should ship a local `AGENTS.md` with benchmark anchor, canonical commands, sharp edges, and any wrapper/runtime exceptions.
- Keep one shared checked-in live env template at the repo root `.env.example`; experiments and `tests/` may keep local `.env` files, but not separate checked-in `.env.example` copies.
- Keep stable repo policy in the repo root `AGENTS.md`, experiment-local truth in `experiments/<name>/AGENTS.md`, and short-lived task intent in `PROMPT.md`.
- Keep the maintained experiment set and machine-readable summaries in `experiments/maintained.json`; when the maintained set changes, update that registry and the human-facing repo docs in the same change set.
- In maintained experiment `README.md` files, `## Current results` should start with a `Status:` line: use `Status: \`placeholder\`` until a maintained run is actually checked, switch it to `Status: \`checked\`` only when the reported run is actually checked, and keep that line aligned with `experiments/maintained.json`.
- Keep personal machine routing, sync setup, host/path details, and local agent-tool configuration out of checked-in repo contract files; the open-source repo should document experiment commands, not one developer's execution topology.
- For repo-wide work, put the active task package in the repo root `PROMPT.md`. For single-experiment work, prefer `experiments/<name>/PROMPT.md`.
- Keep `PROMPT.md` focused on objective, workbench, context, tasks, and constraints; remove or rewrite it when the task package changes materially.
- `train.py` must support `--eval-only`.
- `train.py` should support `--dry-run` when possible for credential-free validation.
- `train.py` must print `METRIC name=value` lines for benchmark metrics.
- When experiments share the same workflow shape, keep the same `# ===== ... =====` section order and the same promoted helper order from `scaffolds/single_file_experiment/naming.md`; omit sections or helpers that do not apply, but do not reshuffle shared logic into different blocks.
- If an experiment has both train and benchmark-eval data, manage them as separate artifacts and separate workflows: make it explicit which upstream source feeds train, which upstream source feeds eval, keep the local train and eval manifests distinct, keep overlap or leakage checks explicit in code and docs, and preserve separate download or snapshot scripts plus adjustment or materialization scripts for each side.
- Prefer a split dataset layout under `data/` for single-train and single-benchmark experiments: keep train-side artifacts under `data/train/` and eval-side artifacts under `data/eval/`. Put full manifests, raw snapshots, and helper scripts under the matching side unless the baseline contract makes that impossible.
- Multi-dataset or multi-benchmark experiments may instead use a dataset-family layout like `fingpt`, for example `data/<family>/...` plus `data/benchmarks/<suite>/...`, when forcing everything into one `train/` and one `eval/` tree would hide real dataset boundaries. Keep the train/eval contract, provenance, and local artifact meaning explicit in code and docs.
- For new experiments, treat dataset management as a preserved local pipeline, not a one-off file drop: keep the raw snapshot directory, the download or snapshot script, and the build or adjustment script under `data/`, colocated with the artifacts they materialize, so later agents can reconstruct the exact local manifest.
- Keep one smoke tier in the same documented layout. The preferred split shape is `data/train/smoke.*` plus `data/eval/smoke.*`; dataset-family layouts may keep smoke artifacts in another explicit location under `data/` when that better matches the experiment contract.
- Prefer a staged flow for both sides when feasible: raw snapshot -> smoke artifact -> full artifact. If a benchmark or upstream source makes one stage impossible, document that exception explicitly in `README.md` and `data/sources.yaml`.
- When an experiment `README.md` reports measured runs, keep the evidence bundle together: train config, eval config, result metrics, and wall-clock timing. For SFT runs, record at least `num_epochs`, `batch_size`, `learning_rate`, and `rank`; for eval wall-clock timing, record the relevant `max_concurrent_requests`.
- Time reporting in experiment `README.md` files must use wall clock consistently. When reporting eval time, say whether it is batch wall-clock or per-run wall-clock, and state any parallel vs sequential execution detail that affects interpretation. Throughput may be included as a supplementary metric, but it does not replace wall-clock timing.
- Optional **AI-native** aid: an experiment may emit a single-line **`ANALYSIS_MANIFEST path=…`** stdout marker and write **`analysis_manifest.json`** under `--output-dir` so agents open one index before parsing JSONL; if that filename or marker becomes a **repo-wide** contract, update `scaffolds/README.md`, `naming.md`, skills, and this file in the same commit.
- If an experiment keeps provenance symlinks or references a shared corpus, keep the default train/eval files local to the experiment and make the split rule explicit in `README.md` and in code.
- `uv run train.py --eval-only --eval-data <full_eval_path>` is the canonical benchmark entrypoint. When an experiment ships `autoresearch.sh`, treat it as the canonical automation wrapper for that experiment's current practical line. If the wrapper differs from the benchmark confirmation path, make the distinction explicit in `README.md` and `autoresearch.md`.
- Keep docs and code in sync. Any change — function names, signatures, CLI flags, metric names, artifact names, feature additions, architecture decisions, or behavioral conventions — must update all affected files in the same commit. Use **`scaffolds/README.md` → *Contract split: who owns what*** as the checklist for *where* each kind of change lives; typical touch sets are:
  - **Templates:** `scaffolds/single_file_experiment/train.py.tpl`, `README.md.tpl`, `autoresearch.sh.tpl`, `autoresearch.md.tpl`, and other `.tpl` files when baseline defaults, wrapper recipes, or comments change
  - **Naming:** `scaffolds/single_file_experiment/naming.md` when a helper is promoted or renamed repo-wide
  - **Profiles:** `scaffolds/profiles/sft.md` / `dpo.md` / `grpo.md` / `eval*.md` when extension recipes or default flag sketches change
  - **Skills:** `skills/new-experiment/SKILL.md` and `skills/new-experiment-plus/SKILL.md` (plus `new-experiment-plus/references/*.md` when research or tinker-mapping text moves) when the baseline vs plus boundary or agent workflow text changes
  - **Shared policy:** `experiments/README.md` when the cross-experiment scaffold policy changes
  - **This file (`AGENTS.md`)** when a hard rule changes
  - **Concrete experiment** `README.md` / `autoresearch.md` / `autoresearch.sh` / `train.py` whenever that experiment implements or documents the new behavior

## Project harness rules

For repo-wide or new-experiment work, treat `scaffolds/` as the harness source of truth.

- Start new experiments from the canonical single-file template under `scaffolds/single_file_experiment/`.
- Start every new experiment from `scaffolds/profiles/eval.md`.
- Do not look for a preset layer; `scaffolds/profiles/` is the only specialization layer.
- Add `scaffolds/profiles/sft.md`, `scaffolds/profiles/dpo.md`, or `scaffolds/profiles/grpo.md` only after the eval path is stable.
- The promoted DPO profile is documentation-first: use `scaffolds/profiles/dpo.md` to standardize pairwise preference training without turning DPO into the canonical template default.
- Keep the experiment runtime single-file: `experiments/<name>/train.py`.
- Preserve stable helper names when the behavior is the same across experiments.
- If two experiments need the same helper shape, update the canonical scaffold instead of letting both drift.
- Prefer promoting reusable patterns into `scaffolds/` rather than creating a fourth local style.
- **Baseline** experiments (`$new-experiment`): keep logging to the minimal artifact set in `scaffolds/README.md` (the scaffold-owned eval snapshot files + `METRIC` lines). **Research-grade** harnesses (`$new-experiment-plus`): when they add run-scoped training JSONL (`eval/periodic.jsonl`, `train/checkpoints.jsonl`, streamed `train/metrics.jsonl`, optional prompt-lineage `train/batches.jsonl`, RL stream logs, …), those files must stay **run-scoped** on reused `--output-dir` (truncate at fresh-run entry unless intentionally resuming); when `train/batches.jsonl` is present, prefer storing only the first few prompt rows per completed step unless the experiment needs the full batch; see `scaffolds/README.md`.
- Keep task-specific logic local: prompts, graders, reward functions, tool semantics, and dataset mapping still belong in the concrete experiment.
- In code, prefer the current maintained `mint` runtime surface for new scaffold-derived experiments and keep runtime naming consistent across code, `.env`, CLI flags, and README wording.
- The standard development sequence is: **get the experiment working on `mint` first** (dry-run, eval-only, then training if applicable). Do not start a new scaffold-derived experiment from a tinker-first runtime skeleton unless the concrete experiment documents that exception explicitly.

## Repo-level files

- `README.md` explains the monorepo and the lifecycle-first experiment contract.
- `PROMPT.md` is the active repo-wide task package when the current work spans multiple experiments or scaffold layers.
- `experiments/maintained.json` is the machine-readable source of truth for the maintained experiment set.
- `experiments/README.md` defines the shared experiment contract and startup order.
- `tests/README.md` documents the repo-level verification tests plus the live MinT smoke scripts.
- `scaffolds/README.md` defines the canonical single-file harness, naming rules, and profile system.
- `requirements/` contains per-experiment requirement documents. Current directories include `aime-on-mint/`, `chat-dpo-on-mint/`, `fingpt-on-mint/`, `finqa-on-mint/`, and `lawbench-on-mint/`, each with a `README.md` defining that experiment's requirements and acceptance criteria.

## When you change logging, artifacts, or stdout contracts

Treat documentation as part of the feature: **no behavior-only commits** for things another agent will infer from logs.

1. **`scaffolds/README.md`** — baseline vs research artifact tables; **Contract split: who owns what**; optional **`analysis_manifest.json` / `ANALYSIS_MANIFEST`** row if the pattern moves or is promoted repo-wide.
2. **`scaffolds/single_file_experiment/naming.md`** — promoted helper names; stdout marker notes (`METRIC`, optional `ANALYSIS_MANIFEST`).
3. **`scaffolds/profiles/`** (`sft.md`, `dpo.md`, `grpo.md`, `eval*.md`) — extension recipes and flag sketches that match the code.
4. **Templates** (`scaffolds/single_file_experiment/*.tpl`) — baseline comments and defaults when new scaffolds should match.
5. **Skills** — `skills/new-experiment/SKILL.md` and `skills/new-experiment-plus/SKILL.md`; plus **`new-experiment-plus/references/*.md`** (`file_upgrade_map.md`, `upgrade_playbook.md`, `logging.md`) when checklists, upgrade path, or tinker mapping change.
6. **`experiments/README.md`** — cross-experiment policy lines.
7. **This file (`AGENTS.md`)** — hard rules and this checklist when contracts shift.
8. **Concrete experiment** — `experiments/<name>/README.md` (artifact/logging notes + any optional **AI-native** read order) and `train.py` / tests.

## Experiment index

<!-- maintained-experiments:start -->
- `chat-dpo`: pairwise chat DPO with held-out preference eval
- `dapo-aime`: direct GRPO on DAPO-Math-17k with an AIME 2024 benchmark plus AIME 2025/2026 eval manifests
- `fingpt`: FinGPT reproduction scaffold with Fineval + sentiment eval and an SFT path
- `lawbench`: LawBench benchmark-first scaffold with a maintained LoRA SFT line
<!-- maintained-experiments:end -->
