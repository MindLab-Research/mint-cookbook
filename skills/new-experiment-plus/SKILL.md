---
name: new-experiment-plus
description: |
  Upgrade or strengthen a MinT cookbook experiment into a research-ready
  harness under `experiments/`.

  Use when an experiment needs infrastructure beyond what scaffolds/profiles
  provide: resume and recovery support, checkpoint registries, richer
  run outputs, **research-grade structured logs** (SFT row cadence, merged train+eval
  rows, run-scoped JSONL), throughput-aware train design, or clearer benchmark and
  provenance contracts.

  Helps authors **design and document** research-level operability and logging
  (README result sections, `run.json` provenance, SFT/DPO/RL checklists) and align SFT with
  tinker-cookbook supervised patterns without importing that repo into experiments.

  This skill does not define baseline artifact naming or an alternate scaffold.
  It upgrades experiments that already follow the repo's canonical `scaffolds/`
  contract.
argument-hint: "<experiment-name>"
---

# New Experiment Plus

Upgrade an existing experiment into a research-ready harness. The experiment should already
have the baseline eval snapshot in place (and optionally basic training) via `$new-experiment`.

## When this skill applies

- User wants help **designing or documenting** research-grade **operability and logs** (SFT/DPO JSONL cadence, `run.json` provenance, README logging / result sections, tinker-cookbook alignment)
- User wants long-run training operability: same-run resume, fresh checkpoint loading, checkpoint registries, richer run outputs, or failure analysis
- User wants throughput-aware train and eval design, especially for RL
- User says training is slow and wants train-loop or scheduling upgrades rather than only more raw request fan-out
- User wants benchmark, provenance, or evaluation contracts written down more explicitly
- User wants an existing experiment upgraded into a research-ready harness

If the user only wants to create a new experiment and establish an eval baseline, use `$new-experiment`.
If the user wants to add basic SFT, DPO, or GRPO training, they can follow `scaffolds/profiles/sft.md`, `scaffolds/profiles/dpo.md`, or `scaffolds/profiles/grpo.md` directly — that does not require this skill.
Use this skill when the experiment needs infrastructure **beyond** what the profiles provide.
If the task is repo-wide reconciliation of existing shared helper drift across
experiments, templates, profiles, and skills, use `$helper-contract-alignment`.

## Boundary with `$new-experiment` and `scaffolds/profiles`

- `$new-experiment` creates the experiment directory and establishes the default readable eval snapshot (`run.json`, `console.log`, `eval/examples.jsonl`, `eval/predictions.jsonl`, `eval/metrics.json`, `METRIC` stdout, `prepare_run_dir` + smoke-only symlink per `scaffolds/README.md`)
- `scaffolds/profiles/sft.md`, `scaffolds/profiles/dpo.md`, and `scaffolds/profiles/grpo.md` guide adding basic training; the canonical template already carries the promoted generic SFT loop plus optional cadence-driven append streams, while DPO and GRPO remain profile-driven extensions
- `$new-experiment-plus` upgrades an experiment that already has eval (and optionally basic training) into a research-grade harness
- Keep basic bounded async primitives in `$new-experiment`; move system-level train and eval scheduling decisions here
- Do not create a second family of templates, baseline artifact names, or helper names here

### Same-commit split (templates + skills + docs)

Research upgrades still **inherit** the canonical scaffold. When you add or rename **research-grade** streams, registries, helpers, or CLI that other experiments are expected to copy, update together:

1. **`experiments/<name>/train.py` and `README.md`** — working implementation and any added logging / outputs table or subsection
2. **`scaffolds/single_file_experiment/naming.md`** — if the helper name is promoted repo-wide
3. **`scaffolds/profiles/sft.md`, `dpo.md`, or `grpo.md`** — if the profile sketch or default flags should match
4. **`scaffolds/single_file_experiment/train.py.tpl` / `README.md.tpl`** — only when the **baseline** template should mention the hook (comments or optional pattern); keep canonical artifact naming in scaffolds, and do not push benchmark-specific or high-volume research logging into the default template unless the repo explicitly promotes it
5. **`scaffolds/README.md`** — if the baseline vs research artifact contract table changes
6. **`$new-experiment` and `$new-experiment-plus` skills** — if the boundary wording or checklists change
7. **`references/logging.md`** — when aligning columns with tinker-cookbook (SFT-only deep dive)
8. **`AGENTS.md` / `experiments/README.md`** — if a hard or shared policy changes

The **authoritative “who owns what” matrix** (repo-neutral) is **`scaffolds/README.md` → *Contract split: who owns what***. The **mandatory doc-sync checklist** for logging/artifacts/stdout is **`AGENTS.md` → *When you change logging, artifacts, or stdout contracts***. This skill’s checklists and `references/*.md` sit **on top** of those two layers.

## Quick repo summary

Keep this repo-level summary in mind when upgrading experiments:

- `scaffolds/` is the source of truth for repo-wide experiment templates, profiles, naming, and the full artifact naming contract for new experiments.
- Every experiment starts eval-first from the canonical single-file scaffold, then adds training only after `--dry-run` and `--eval-only` are stable.
- Each experiment stays self-contained under `experiments/<name>/` and should not import helpers from another experiment.
- `train.py` remains the readable source of truth, must support `--eval-only`, should support `--dry-run` when possible, and must emit `METRIC name=value` lines.
- `uv run train.py --eval-only --eval-data <full_eval_path>` is the canonical benchmark entrypoint; `autoresearch.sh` is the canonical automation wrapper for the current practical line around that path.
- Baseline scaffold outputs should stay small and readable: `run.json` (two-phase status), `console.log` (TeeStream), and the `eval/` snapshot trio (`examples.jsonl`, `predictions.jsonl`, `metrics.json`) with `prepare_run_dir` (smoke-only symlink). The canonical generic SFT path may already add the scaffold-owned `train/metrics.jsonl`, `eval/periodic.jsonl` (with prediction samples), `train/checkpoints.jsonl`, and optional `train/batches.jsonl` family when its cadence flags are enabled. This skill starts where that promoted generic path stops: benchmark-specific logging schemas, extra streams, throughput policy, and longer-run operability.
- The baseline scaffold README flow is `Quickstart`, `Fast contract tests`, `Live smoke tests`, `Data`, `Current results`, `References`; start `Current results` with `Status: \`placeholder\`` until a checked run exists, then switch it to `Status: \`checked\`` only when the reported run is actually checked, and let research-grade experiments extend that structure rather than replacing it with an unrelated layout.
- Measured runs in `README.md` should keep the evidence bundle together: relevant train config, eval config, result metrics, and wall-clock timing. Eval-only baselines still count as measured runs.
- `autoresearch.md` should stay a short wrapper protocol, not a grab bag of notes: objective, current recipe, search signals, mutable surface, frozen contract, recovery path, and stopping rules.
- If a reusable contract changes, update the matching templates, docs, naming specs, profiles, skills, experiment docs, and `AGENTS.md` in the same commit.

## House style for scaffold code and docs

When upgrading scaffold-facing files, follow this default house style unless a file already has a stronger established convention:

- Use explicit top-level responsibility sections in scaffold Python with comment headers such as `# ===== CLI =====`.
- Leave exactly one blank line after each section header in scaffold Python.
- Keep normal Python top-level spacing: two blank lines between top-level `def` / `async def` / `class` blocks.
- Order scaffold code from shared framework to task-local logic: constants/config -> env loading -> backend compatibility -> CLI -> infrastructure helpers -> task adapters -> runtime entrypoints -> artifact writing -> entrypoint.
- Keep task-specific prompts, graders, reward logic, and dataset mapping inside the adapter layer instead of mixing them into shared helpers.
- Match helper names to the returned layer: keep `sample_assistant_text(...)` for text-returning eval helpers, and use `sample_assistant_message(...)` when the sampler path is message-first and text is extracted later (per `scaffolds/single_file_experiment/naming.md`).
- Keep scaffold Markdown concise and operational: short sections, explicit commands, explicit outputs, and benchmark-facing constraints stated directly.
- Prefer behavior-preserving cleanup when standardizing templates and docs; avoid sneaking in logic changes during formatting passes.
- Treat `scaffolds/single_file_experiment/train.py.tpl`, `README.md.tpl`, and `autoresearch.md.tpl` as the style references for future scaffold files unless the repo deliberately revises them.

## Research-grade operability (what this skill helps you ship)

Goal: `--output-dir` stays **self-explanatory months later**—steps, weights snapshot, eval numbers, and CLI provenance—without re-running the job. Baseline experiments stop at the scaffold-owned readable eval snapshot plus `run.json` + `METRIC` lines (`scaffolds/README.md`). This skill covers everything **beyond** that for long-running **SFT**, **DPO**, or **RL**.

### Agent workflow (use in order)

1. **Pick algorithm family:** SFT (LoRA + cross-entropy), DPO (pairwise preference optimization), or RL (GRPO-style) — the logging and registry checklists differ.
2. **Lock benchmark contract** before growing train-side logs (frozen eval path, grader, primary `METRIC` names).
3. **Declare eval timing in README:** whether periodic eval uses **pre-step** or **post-step** weights relative to `optim_step` / `forward_backward` (LawBench SFT is post-step; tinker-cookbook supervised often pre-step for pipelining—do not compare runs across different conventions without saying so).
4. **Run-scoped JSONL:** on each **fresh** training start, reset the append-stream family so a reused `--output-dir` cannot splice unrelated histories: truncate/create the streams enabled by the active cadence flags, and remove disabled stream files left behind by an older run in the same directory. Same-run resume is directory-driven: rerun the same training command with the same run directory when the current `train/checkpoints.jsonl` already contains a resumable `state_path` row. Reserve `--load-checkpoint-path` for starting a fresh run from saved weights only; treat the matched `train/checkpoints.jsonl` row, not checkpoint-path parsing, as the restore contract.
5. **SFT row cadence:** when the run keeps a streamed `train/metrics.jsonl`, keep **every** row that carries merged eval (`test/eval_*`) once that stream is enabled; subsample pure train rows with `--train-metrics-every` / tame stdout with `--train-print-every` (see `experiments/lawbench` and `experiments/fingpt`); **stream-append** `train/metrics.jsonl` instead of buffering all steps in RAM. If the metrics stream is disabled, periodic eval should still land in `eval/periodic.jsonl` only. When reviewers need to see which prompts actually hit each step, pair it with `train/batches.jsonl` rather than bloating the scalar file; keep the preview bounded to the first few prompt rows per completed step unless the experiment truly needs full batches.
6. **`run.json` provenance:** `args`, raw `argv`, shell-safe `command` (e.g. `shlex.join`), plus output-path pointers when helpful.
7. **`README.md`:** keep the baseline section flow unless the experiment has a strong reason not to, add a logging / outputs subsection or section when it materially improves readability (outputs table + cadence flags + eval semantics), and keep measured-run evidence bundles together (train config when relevant, eval config, result metrics, wall-clock timing).
8. If the reporting pattern is repo-wide, follow **`scaffolds/README.md` → *Contract split: who owns what*** and update templates, `naming.md`, profiles, skills, `experiments/README.md`, and `AGENTS.md` in the **same commit** per repo contract.
9. **AI-native index (optional):** ship `analysis_manifest.json` plus a single-line stdout **`ANALYSIS_MANIFEST path=…`** after `run.json` so agents open one structured index before JSONL; document the read order in **README**. If the filename or marker becomes **repo-wide**, extend **`scaffolds/README.md`** research table and **`AGENTS.md`** in the same commit.

### SFT checklist (research-grade)

- [ ] `train/metrics.jsonl` — **append** during training; merge held-out eval scalars into the **same** step object under `test/eval_*` when periodic eval runs.
- [ ] `train/batches.jsonl` (optional but recommended when prompt visibility matters) — one row per completed train step with `example_id`s, task / level mix, token-count summaries, and bounded prompt / assistant-text previews (the repo default is the first five rows); treat it as the SFT analogue of RL rollout traces.
- [ ] `eval/periodic.jsonl` — append-only eval stream `{ "step": N, **metrics, "samples": [...] }` for each periodic eval; include first `PERIODIC_EVAL_SAMPLE_COUNT` prediction previews (`id`, `prediction`, `correct`, `expected`, `assistant_text`) so model output quality is observable without re-running eval.
- [ ] `train/checkpoints.jsonl` — when the backend exposes `save_state` and `save_weights_for_sampler`, register one logical checkpoint `name` plus resumable `state_path` URIs beside eval-ready `sampler_path` URIs in the same append-only registry.
- [ ] Resume split — rerunning the same training command with the same run directory means same-run continuation with restored optimizer plus loop state from the matched checkpoint row; `--load-checkpoint-path` means fresh-run weight load only; the runtime save names behind `state_path` and `sampler_path` should stay distinct.
- [ ] For MinT-style same-run resume, keep the flow uniform: `create_lora_training_client(...)` first, then `load_state_with_optimizer(...)`.
- [ ] **Cadence flags** — e.g. `--train-metrics-every`, `--train-print-every` (or documented equivalents) so long jobs do not spam disk or stdout.
- [ ] **Helpers** — e.g. `reset_supervised_append_streams`, `merge_eval_metrics_into_step_record`, `compute_mean_nll`, `save_sampler_checkpoint`, `should_record_train_metrics_row(..., has_merged_eval=True)` on merged rows once the streamed `train/metrics.jsonl` path is enabled (names and signatures live in `scaffolds/single_file_experiment/naming.md`; promote a helper only when two experiments already share the exact shape).
- [ ] **AI-native (optional)** — `analysis_manifest.json` + stdout `ANALYSIS_MANIFEST path=…` + `run.json` `artifacts.analysis_manifest`; README section explains the read order for LLM-led review.
- [ ] Optional `timing_spans.jsonl` / `trace_events.jsonl` **only** if profiling is a real requirement.

### DPO checklist (research-grade)

- [ ] `train/metrics.jsonl` — append per-step pairwise metrics such as `dpo_loss`, `accuracy`, `margin`, reward summaries, learning rate, timing, and throughput.
- [ ] `train/batches.jsonl` (optional but recommended when prompt visibility matters) — one row per completed train step with bounded prompt / chosen / rejected previews for the first few pairs, plus any `group_id` or prompt-group metadata needed to debug batch construction.
- [ ] `eval/periodic.jsonl` — append-only held-out pairwise eval stream keyed by train step, with bounded sampled prediction previews (`pair_id`, `margin`, `correct`, score summaries, prompt preview).
- [ ] `train/checkpoints.jsonl` — checkpoint registry with logical `name`, loop-state fields (`step`, `epoch`, `batch`), resumable `state_path`, and eval-ready `sampler_path`; keep runtime save names distinct for state vs sampler exports.
- [ ] Resume split — rerunning the same training command with the same run directory means same-run continuation with restored optimizer plus loop state from the matched checkpoint row; `--load-checkpoint-path` means fresh-run weight load only.
- [ ] README documents the held-out pairwise scoring rule, the reference-model policy (`--reference-model` defaulting to `--base-model` when applicable), and the main DPO knobs (`dpo_beta`, `batch_size`, `num_epochs`, `max_length`, grouping policy).

### RL checklist (research-grade)

- [ ] `train/metrics.jsonl` — per-step RL metrics: `reward_mean`, `accuracy`, `format`, `datums`, `num_trajectories`, `prompt_groups`, `accepted_groups`/`trained_groups`/`dropped_groups`, `samples_per_sec`, `time/step`, `time/total`, `time/eta`, `progress`.
- [ ] `train/rollouts.jsonl` — per-group rollout records with `question`, `gold_answer`, `num_trajectories`, `num_correct`, `num_formatted`, `status`, per-trajectory `trajectory_rewards`/`trajectory_correct`/`trajectory_advantages`/`trajectory_answers`/`trajectory_token_counts`, and `first_trajectory_text` preview.
- [ ] `train/failures.jsonl` + `train/failures.log` — paired structured (JSONL) + human-readable failure logs for rollout errors; structured records share the rollout schema with `status` != `"ok"`.
- [ ] `train/checkpoints.jsonl` — checkpoint registry with a logical `name`, loop-state fields, for example SFT `step`/`epoch`/`batch` or RL `step`/`completed_steps`/`next_step`, plus `state_path` (resumable) and `sampler_path` (eval-ready).
- [ ] Resume split — rerunning the same training command with the same run directory means same-run continuation with restored optimizer plus loop state from the matched checkpoint row; `--load-checkpoint-path` means fresh-run weight load only; keep the runtime save names behind `state_path` and `sampler_path` distinct.
- [ ] `eval/metrics.jsonl` — periodic eval metrics appended per eval step; per-step eval details under `eval/steps/step-NNNN/`.
- [ ] Algorithm knobs (`group_size`, `groups_per_batch`, `rl_learning_rate`, `rl_temperature`, `rl_loss`) separated from throughput knobs (`stream_minibatches_per_step`, `tail_grace_seconds`, `max_concurrent_requests`, `min_accepted_groups`).
- [ ] Run-scoped append logs (`train/metrics.jsonl`, `train/rollouts.jsonl`, `train/failures.jsonl`, `train/checkpoints.jsonl`, `eval/metrics.jsonl`) truncated on fresh starts unless resume story is explicit.
- [ ] README documents output files / streams, their schemas, and the algorithm-vs-throughput knob split.

### RL ↔ tinker-cookbook field mapping

Reference: `tinker-cookbook/tinker_cookbook/rl/train.py`, `rl/rollout_logging.py`, and `rl/metrics.py`. That stack uses `Config` for algorithm + throughput knobs, `ml_logger.log_metrics(metrics, step=)` for a shared step index, `rollout_logging.serialize_rollout_summaries()` for per-trajectory JSONL, and `checkpoint_utils.save_checkpoint_async` for checkpoint metadata.

| tinker-cookbook RL idea | Typical cookbook `train.py` mapping |
|---|---|
| `rollout_logging.serialize_rollout_summaries()` — per-trajectory JSONL with schema_version, rewards, steps | `train/rollouts.jsonl`: per-group records with trajectory-level reward/correct/format arrays and text preview |
| `metrics.compute_kl_sample_train()` — KL between sampling and training logprobs | Optional KL metrics in `train/metrics.jsonl`; dapo-aime does not currently compute KL |
| `metrics.compute_sampling_client_metrics()` — step lag and sampling time | `sampler_step` field in rollout records tracks staleness; `samples_per_sec` in `train/metrics.jsonl` |
| `Config.eval_every` / `Config.save_every` | `--eval-every-steps` / `--save-every-steps` |
| `Config.remove_constant_reward_groups` | `group_lacks_reward_signal()` — skip groups with identical rewards |
| `Config.stream_minibatch_config` — streaming forward_backward within step | `--stream-minibatches-per-step` with tail control knobs |
| `Config.rollout_error_tolerance` — retry or crash on rollout errors | `train/failures.jsonl` + `train/failures.log` paired logs |
| `WrappedTrajectoryGroup` — trajectory + builder + sampling_client_step | `PromptSamplingResult` — rollout result + sampler_step + row metadata |
| `ml_logger.log_metrics(metrics, step=)` — shared step index for dashboards | `append_jsonl(metrics_jsonl_path, record)` with step field |
| `trace.IterationWindow` — per-step timing spans | `time/step`, `time/total`, `time/eta` in `train/metrics.jsonl` records |
| `checkpoint_utils.save_checkpoint_async` — periodic + rolling checkpoints | `train/checkpoints.jsonl` with logical `name`, `state_path`, and `sampler_path` |

### SFT ↔ tinker-cookbook field mapping

Reference: `tinker-cookbook/tinker_cookbook/supervised/train.py` and `tinker_cookbook/utils/ml_log.py`. That stack uses a `log_path` directory, `ml_logger.log_metrics(metrics, step=)` so **train and eval scalars share one step index** for dashboards, `checkpoint_utils.save_checkpoint_async` metadata beside the train loop, per-step `timing_spans.jsonl` from `trace.IterationWindow`, and optional `trace_events.jsonl` when tracing is enabled.

Cookbook **single-file SFT** experiments (for example `experiments/lawbench`) should mirror the same information shape without pulling in tinker-cookbook as a library:

| tinker-cookbook supervised idea | Typical cookbook `train.py` mapping |
|----------------------------------|-------------------------------------|
| One merged metrics row per optim step (`log_metrics`) | `train/metrics.jsonl`: one JSON object per completed step with `train_mean_nll`, `learning_rate`, `epoch`, `progress`, token counts, wall-time / throughput; merge in optimizer extras when the SDK exposes them; merge held-out eval scalars into the **same** row under `test/eval_*` when periodic eval runs (LawBench implements this) |
| Step-level prompt lineage when scalar rows are not enough | `train/batches.jsonl`: one JSON object per completed step with batch `example_id`s, task / level mix, token-count summaries, and bounded prompt / assistant-text previews (the repo default is the first five rows) |
| Held-out eval keyed by training step (often `test/` prefixes in the logger) | `eval/periodic.jsonl`: `{ "step": N, **eval_metrics }` for each periodic eval |
| Checkpoint / state registry next to logs | `train/checkpoints.jsonl` for periodic `save_state` paths plus a final row; each row carries a logical `name`, `state_path` (resumable), `sampler_path` (eval-ready), and the loop-state fields needed for same-run continuation; the runtime save names behind the two paths stay distinct |
| Bounded row cadence for long SFT jobs | `--train-metrics-every` / `--train-print-every` (LawBench): append `train/metrics.jsonl` without buffering the whole run in RAM; reset it with the other run-scoped append streams on fresh starts so disabled streams do not linger |
| Optional fine-grained timing / chrome trace | Optional `timing_spans.jsonl`-style or `trace_events.jsonl`-style files **only** if you add profiling; not part of the default LawBench contract |
| WandB / Neptune / Trackio hooks | Optional; default remains local JSONL + `METRIC name=value` for `pi-autoresearch` |

**Eval timing semantics (document in the experiment README):** upstream supervised SFT often runs evaluators **before** submitting the current step’s `forward_backward` (pre-step snapshot semantics for pipelining). LawBench SFT uses **post-step** weights for `--eval-every` (see `scaffolds/profiles/sft.md`). Either is valid; do not silently compare runs that use different conventions.

## Canonical scaffold contract

Before upgrading an experiment, make sure it aligns with these repo sources of truth:

- `scaffolds/README.md`
- `scaffolds/single_file_experiment/train.py.tpl`
- `scaffolds/single_file_experiment/naming.md`
- `scaffolds/profiles/eval.md`
- `scaffolds/profiles/sft.md`
- `scaffolds/profiles/dpo.md`
- `scaffolds/profiles/grpo.md`

Upgrades should preserve the canonical section order, stable helper names, and stable output naming
unless the contract itself is being intentionally revised in `scaffolds/`.

## Default workflow

1. Decide whether the task is "upgrade an existing eval baseline" or "create a new eval baseline first, then upgrade it".
2. If needed, first align the experiment to the canonical scaffold instead of layering upgrades onto drifted local structure.
3. Lock the eval contract first.
   - What data layout and file naming are canonical in the chosen baseline?
   - What are the eval outputs in this repo: mirrors, adapters, manifests, or native baseline files?
   - What metrics are canonical in the chosen baseline?
   - What prompt, parser, and grader rules must stay fixed for fair comparison?
4. Assume basic training is already added via `scaffolds/profiles/` or will be added alongside the upgrade.
5. Separate algorithm knobs from throughput knobs.
   - Algorithm semantics: `group_size`, `groups_per_batch`, reward shape, eval contract
   - Throughput semantics: `max_concurrent_requests`, `stream_minibatches_per_step`, `min_accepted_groups`, `min_started_minibatches`, `tail_grace_seconds`
6. Diagnose bottlenecks before changing concurrency.
   - Do not assume training is slow because request fan-out is too small.
   - Check whether the real bottleneck is a step-level barrier: rollout collection may already be async, but training still waits for the full nominal batch before any `forward_backward` starts.
7. Prefer throughput fixes in this order unless the user explicitly wants something more aggressive:
   - same-step streaming minibatches
   - tail early-stop or tail truncation for the last slow or invalid groups
   - bounded async rollout fan-out or queue workers
   - only then more complex train-engine redesigns
8. Make long-run runs inspectable.
   - **SFT:** prefer `train/metrics.jsonl`, `eval/periodic.jsonl`, and checkpoint registries (tinker-cookbook-shaped fields) over opaque one-number summaries.
   - **RL:** prefer structured rollout logs, failure logs, checkpoint registries, eval histories, and explicit train and eval metadata.
9. Keep the experiment self-contained under `experiments/<name>/`, even when the harness becomes more structured than a minimal one-file baseline.

## Concurrency layers

Treat concurrency as multiple layers, not one knob:

- Sampling shape: `num_samples`, repeated `num_samples=1` requests vs grouped requests
- In-flight request fan-out: `max_concurrent_requests`
- In-step overlap: `stream_minibatches_per_step`
- Tail control or step exit: `min_accepted_groups`, `min_started_minibatches`, `tail_grace_seconds`
- Train and eval overlap: background eval, queued eval draining, sampler snapshot alignment

Document which layer each knob controls so throughput changes do not get confused with algorithm changes.

## Typical upgrades this skill may add

### Observability (SFT and GRPO)

- per-step timing (step_time_seconds, tokens_per_second)
- training progress tracking (progress percentage, cumulative tokens consumed)
- structured step-level metrics in `train/metrics.jsonl` beyond just loss and lr (for SFT, mirror the fields upstream `supervised/train.py` merges before `log_metrics`)
- console or logger lines with step timing to diagnose slow steps (LawBench defaults to every-step stdout; throttle with `--train-print-every` on long runs)

### Resume and recovery (SFT and GRPO)

- clearer same-run resume and fresh checkpoint-loading flows for long-running experiments
- checkpoint-aware periodic eval, including queued background evals when needed

### Throughput optimization (primarily GRPO/RL)

- same-step streaming minibatches for RL train steps
- tail early-stop for slow or invalid final groups
- bounded async rollout collection with preserved request order
- lightweight semaphore-backed async eval with deterministic postprocessing

### Artifacts and contracts (summary)

- **SFT:** ship the checklist above; keep `METRIC name=value` for automation; duplicate eval scalars in `eval/periodic.jsonl` **and** merged into `train/metrics.jsonl` on eval steps so dashboards can use one file or the other.
- **RL:** ship the RL checklist; orient on `experiments/dapo-aime` for naming and file shapes.
- **Both:** run-scoped JSONL truncation on fresh starts (unless intentional resume); benchmark + provenance text in `README.md`; promote reusable helper names to `scaffolds/` + `naming.md` when the behavior matches across experiments.

## Local references

Read only what the task needs:

- **`AGENTS.md`** — especially **When you change logging, artifacts, or stdout contracts** (mandatory multi-file sync list for AI-led work)
- **This file** — section **Research-grade operability** (workflow + SFT / DPO / RL checklists); use as the closing self-check with `references/file_upgrade_map.md` → *After editing (self-check)*.
- `scaffolds/README.md` - repo-wide scaffold contract and source-of-truth policy
- `scaffolds/single_file_experiment/naming.md` - stable naming and backend-neutral helper rules
- `references/upgrade_playbook.md` - practical upgrade path from eval-first scaffold to research-ready harness
- `references/concurrency_layers.md` - how to reason about concurrency, barriers, overlap, and tail control
- `references/file_upgrade_map.md` - which files to strengthen when upgrading a baseline into a research-ready experiment
- `references/logging.md` - sibling-repo SFT logging lessons and a **scaffold / profile / experiment / skill** responsibility matrix (read-only; do not import tinker-cookbook into `experiments/`)
- `tinker-cookbook/tinker_cookbook/supervised/train.py` - upstream SFT train loop, `ml_logger.log_metrics`, checkpoint + trace file layout (read-only reference; do not import into experiments)
- `experiments/lawbench/train.py` and `experiments/lawbench/README.md` - in-repo SFT harness with research-grade JSONL logging, provenance, and bounded per-batch prompt lineage in `train/batches.jsonl`
- `experiments/fingpt/train.py` and `experiments/fingpt/README.md` - in-repo SFT harness with `--task-type` driven eval/training, merged periodic-eval rows and cadence flags (same patterns as lawbench, without `train/batches.jsonl`)
- `experiments/dapo-aime/train.py` - current in-repo example of a richer RL harness
- `experiments/dapo-aime/README.md` - current in-repo example of output/logging and benchmark contract documentation

## Runtime decision matrix

Before growing research-grade logs, double-check the runtime choice matches
the benchmark's deployment form. The scaffold now defaults to `mint` because
that is the maintained runtime surface used across this repo. Keep that
mint-first default unless a concrete benchmark or deployment constraint
requires a different documented backend.

Whatever is chosen, keep runtime imports, env keys, CLI defaults, and README
wording consistent. Mixed states such as "code uses mint but the README still
says tinker" block later cross-experiment comparisons.

Current runtime choices in this repo (see
`scaffolds/single_file_experiment/naming.md` for the canonical matrix):

| Experiment   | Runtime | Why |
|--------------|---------|-----|
| lawbench     | mint    | cluster deployment + official benchmark backend |
| fingpt       | mint    | same cluster plane as lawbench |
| chat-dpo     | mint    | concrete DPO experiment; mint-native preference training and sampler-checkpoint eval |
| dapo-aime  | mint    | RL path on MinT (`mint` SDK; same patterns as lawbench/fingpt/chat-dpo) |

## Relationship to `$mint-api`

Use `$mint-api` for API details, async helper implementations, and repo-standard helper patterns.
Use this skill to decide which training-system structure to build on top of those primitives.

## Relationship to `$new-experiment`

- `$new-experiment` creates the experiment and establishes the eval baseline
- `scaffolds/profiles/` guides adding basic SFT, DPO, or GRPO training
- This skill upgrades beyond profiles: resume, checkpointing, throughput, **research-grade logs and operability outputs** (checklists above), and README/run.json contracts
