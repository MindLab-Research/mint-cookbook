---
name: project-harness-bootstrap
description: |
  Define and bootstrap the repo-level harness for this MinT cookbook experiment
  monorepo.

  Use when initializing the repository, retrofitting the root-level conventions,
  or clarifying how independent experiment subdirectories should be structured,
  run, benchmarked, and maintained.

  Core behavior: create a minimal root harness, make each `experiments/<name>/`
  directory self-contained and easy for a developer or agent to enter, and make
  `scaffolds/` the source of truth for experiment templates.
---

# Project Harness Bootstrap

Use this skill to define the repository contract.

The harness is important, but in this repo the harness should stay narrow:

- make the project shape obvious
- make each experiment runnable in isolation
- make the benchmark entry point unambiguous
- make agent behavior predictable
- make `scaffolds/` the single source of truth for experiment scaffolding

Do not use this skill to generate a large documentation tree just because it is possible.

## When this skill applies

- Initializing the repository
- Refactoring the root structure of the repo
- Writing or updating the root-level repo contract
- Clarifying how experiment directories integrate with `pi-autoresearch`
- Defining the contract that all experiments under `experiments/` and the scaffold taxonomy under `scaffolds/` must follow

Do not use for:

- Implementing one specific experiment's `train.py`
- Debugging API calls inside an experiment
- Writing one-off training logic for a single algorithm

For those cases, use:

- `$mint-api` for API or SDK usage
- `$new-experiment` when adding a concrete experiment directory within the repo contract
- `$new-experiment-plus` when the experiment needs a research-ready harness, richer artifacts, or throughput-aware train and eval design
- `$helper-contract-alignment` when the task is repo-wide reconciliation of same-named helpers, scaffold drift, or shared contract sync across experiments/templates/profiles/skills

## Repo-level harness contract

After bootstrap, the repo should make these facts obvious:

1. This is a monorepo of independent MinT cookbook experiment reproductions.
2. The canonical work unit is `experiments/<name>/`.
3. `scaffolds/` is the single source of truth for experiment templates, profiles, and naming.
4. A person or agent should be able to enter one experiment directory and understand:
   - what the experiment reproduces
   - what data it uses
   - how to train it
   - how to evaluate it
   - how to benchmark it through `pi-autoresearch`

## Required root files

The harness should create and maintain only the root files that carry repo-wide meaning:

- `README.md` - what this repo is and how experiments are organized
- `AGENTS.md` - how agents should navigate the repo
- `.env.example` - credential and endpoint placeholders
- `.gitignore` - Python, uv, and local env ignores
- `experiments/README.md` - contract and index for experiment directories
- `scaffolds/README.md` - root entrypoint for the canonical experiment scaffold

Anything beyond that is optional and should only be added when the repo genuinely needs it.

## Required experiment contract

Every experiment directory under `experiments/` should contain:

- `pyproject.toml` - uv project config
- `README.md` - self-explanatory experiment guide
- `train.py` - self-contained training and evaluation entry point
- default local train and eval data artifacts, or explicitly documented baseline-native equivalents
- `data/sources.yaml` - training data provenance, baseline mapping, and multi-source merge recipe
- `autoresearch.sh` - benchmark command for `pi-autoresearch`
- `autoresearch.md` - scope file for the benchmark path and ongoing `pi-autoresearch` work

The details of those files should be defined in `scaffolds/`, not re-invented independently in each skill.

## Hard rules

- Prefer experiment-local code over shared utilities while the repo is still young.
- Do not require an agent to read half the repo before it can work on one experiment.
- Avoid non-essential defensive code, dead branches, or placeholder helpers that do not serve the current experiment contract.
- Use `try` and `except` only when the failure mode is understood and the recovery or error-reporting behavior is part of the contract.
- `train.py` must be the human-readable source of truth for training and evaluation.
- `train.py` must support `--eval-only`.
- `train.py` should support `--dry-run` when the experiment can validate local data and prompt formatting without credentials.
- `train.py` must print `METRIC <name>=<value>` lines.
- Each experiment should expose one stable primary metric for benchmarking.
- Data layout and primary metrics should align with the chosen baseline whenever possible.
- `autoresearch.sh` must be the benchmark entry point that `pi-autoresearch` can run repeatedly.
- Reusable experiment-template changes should land in `scaffolds/` first so old and new experiments do not drift.
- Keep docs and code in sync. Any change — function names, signatures, CLI flags, metric names, artifact names, feature additions, architecture decisions, or behavioral conventions — must update all affected files in the same commit. This includes `scaffolds/`, `naming.md`, experiment `README.md`, `.codex/skill/` references, `AGENTS.md`, and any other file that references the changed concept. Do not leave docs out of sync with code.

## Quick repo summary

Keep this repo-level summary in mind when writing or refactoring harness code:

- `scaffolds/` is the source of truth for repo-wide experiment templates, profiles, naming, and baseline artifact contracts.
- Every new experiment starts eval-first from the canonical single-file scaffold, then adds training only after `--dry-run` and `--eval-only` are stable.
- Each experiment stays self-contained under `experiments/<name>/` and should not import helpers from another experiment.
- `train.py` remains the readable source of truth, must support `--eval-only`, should support `--dry-run` when possible, and must emit `METRIC name=value` lines.
- `uv run train.py --eval-only` is the canonical bare benchmark entrypoint; `autoresearch.sh` is the canonical automation wrapper around that path.
- Baseline scaffold outputs stay minimal (`predictions.jsonl`, `eval_metrics.json`, `run.json`) unless the experiment explicitly graduates to research-grade logging.
- If a reusable contract changes, update the matching templates, docs, naming specs, profiles, skills, experiment docs, and `AGENTS.md` in the same commit.

## House style for scaffold code

When writing or reformatting scaffold-facing code, follow this default house style unless a file already has a stronger established convention:

- Use explicit top-level responsibility sections with comment headers such as `# ===== CLI =====`.
- Leave exactly one blank line after each section header.
- Keep normal Python top-level spacing: two blank lines between top-level `def` / `async def` / `class` blocks.
- Order scaffold code from shared framework to task-local logic: constants/config → env loading → backend compatibility → CLI → infrastructure helpers → task adapters → runtime entrypoints → artifact writing → entrypoint.
- Keep task-specific prompts, graders, reward logic, and dataset mapping inside the adapter layer instead of mixing them into shared helpers.
- Prefer behavior-preserving cleanup when standardizing templates; avoid sneaking in logic changes during formatting passes.
- Treat `scaffolds/single_file_experiment/train.py.tpl` as the style reference for future scaffold Python unless the repo deliberately revises that template.

## Agent workflow

Agents should follow this order:

1. Start in the target experiment directory.
2. Read that experiment's `README.md` and `train.py`.
3. Use root harness docs only when changing repo-wide conventions or when the experiment docs are incomplete.
4. Use `scaffolds/` when creating a new experiment or reconciling style drift across experiments.
5. Keep edits scoped to one experiment unless the task is explicitly cross-cutting.

## Bootstrap behavior

Use the bundled `project-harness-bootstrap/scripts/bootstrap_harness.py` to create the minimal root harness files and directories.

The bootstrap script should:

- write the required root files listed above
- create `experiments/`
- treat the repo-root `scaffolds/` directory as the canonical experiment-template source of truth
- in `greenfield` mode, copy the canonical `scaffolds/` tree into the target root
- in `retrofit` mode, validate the target `scaffolds/` tree and sync any missing canonical scaffold files without inventing a second template family

It should not assume a shared utility package, planning system, or oversized doc tree is already warranted.

## Bundled resources

- `scripts/bootstrap_harness.py` - writes the minimal root harness and syncs the canonical repo `scaffolds/` tree
- `references/harness-principles.md` - optional guidance for how a minimal harness can grow in a mature repo
- `templates/` - the minimal root templates emitted by the bootstrap script
- `scaffolds/` - repo-level experiment scaffold docs and companion templates that downstream experiment skills should reuse
