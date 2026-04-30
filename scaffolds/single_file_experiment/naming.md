# Naming Rules

These naming rules keep experiments readable and reduce drift across the repo.

## Repo vs runtime language

Use `MinT` in repo-level docs when describing the cookbook or platform.
The canonical scaffold now uses `mint` runtime names by default:

- `import mint` / `from mint import types`
- `MINT_API_KEY` / `MINT_BASE_URL`
- `--mint-timeout`

The current compatibility dependency is still pinned as `tinker==0.15.0`, but scaffold-derived code, env keys, and CLI flags should be written in `mint` terms from the start.
A concrete experiment may intentionally diverge, but its `README.md`, CLI flags, and backend compatibility code should keep that choice self-consistent.

### Runtime choice guidance

The scaffold defaults to `mint` runtime because that is the maintained runtime
surface used across this repo today. The standard development sequence is:

1. Start the experiment on `mint`: scaffold from template, validate with `--dry-run` and `--eval-only`, then add training if applicable.
2. Keep the `mint` runtime choice unless a concrete benchmark or deployment constraint requires a different documented backend.

Do not start a new experiment from a tinker-first scaffold. If a concrete
experiment intentionally diverges from the mint-first default, keep **runtime,
environment keys, CLI defaults, and README wording** in sync — mixed states
such as "code uses mint but the README still says tinker" are not acceptable.

The bundled maintained experiments are already mint-backed today, and new scaffold-derived experiments should match that default.

### Legacy tinker-script bridge

If you are migrating existing code that already assumes a local module name
`tinker`, prefer the narrowest compatibility bridge first:

```python
import mint as tinker
```

Then move endpoint and auth wiring onto `MINT_BASE_URL` and `MINT_API_KEY`.
Treat this alias as a migration bridge for old code, not as the default style
for new scaffold-derived experiments.

Current runtime choices across this repo:

| Experiment   | Runtime | Why |
|--------------|---------|-----|
| lawbench     | mint    | cluster deployment + LawBench official benchmark runs on mint |
| fingpt       | mint    | same cluster / account plane as lawbench |
| dapo-aime  | mint    | RL path on MinT (`mint` SDK; same patterns as lawbench/fingpt/chat-dpo) |
| chat-dpo     | mint    | concrete DPO experiment; mint-native preference training and sampler-checkpoint eval |

## Chat vs token language

Use `assistant_message` / `assistant_text` for chat-structured outputs or decoded assistant strings.
Reserve `completion` for token-level suffixes, SFT target spans, or upstream Tinker names that are already standardized around prompt/completion token splits.

## Artifact directory naming

Run directories inside `artifacts/` follow a fixed format so that both
humans browsing with `ls` and coding agents scanning for results can
identify runs at a glance.

### Format

```
{mode}-{detail}-{model}-{YYYYMMDD}-{HHMMSS}
```

Each segment uses kebab-case.  Only `mode` and the date-time suffix are
required; `detail` and `model` may be omitted when there is nothing
meaningful to encode.

| Segment | Required | Description | Examples |
|---------|:--------:|-------------|----------|
| `mode`  | yes | What the run did | `eval-only`, `sft`, `grpo`, `smoke-eval`, `smoke-sft`, `dry-run` |
| `detail`| no  | Key hyperparameters or distinguishing tags | `1epoch`, `rank16`, `lr1e-4`, `dedup` |
| `model` | no  | Model short name (drop org prefix and `-Instruct` suffix) | `qwen3-4b`, `llama3-8b` |
| date    | yes | `YYYYMMDD` | `20260417` |
| time    | yes | `HHMMSS` | `143022` |

Examples:

```
artifacts/
├── latest/                                        # scratch pad for interactive runs
├── sft-1epoch-qwen3-4b-20260417-143022/
├── eval-only-qwen3-4b-20260417-091530/
├── smoke-eval-20260416-220000/
└── smoke-sft-20260416-223000/
```

### Rules

1. **Date-time goes last.**  This makes `ls` sort by mode first (same
   type of run groups together) while still being chronologically
   sortable within each group.
2. **`latest` is for quick interactive use only.**  The default
   `--log-path` in `parse_args` points to `artifacts/latest` so a
   short interactive eval command still works without extra run-dir
   flags once `--eval-data` and model flags are supplied. Keep the eval
   manifest itself explicit: standard split-layout experiments should
   pass `data/eval/smoke.jsonl` for `--dry-run` validation and
   `data/eval/full.jsonl` for the frozen benchmark, while multi-dataset
   experiments may use documented dataset-family specs, instead of
   injecting those paths implicitly in code.
   `latest` is a plain directory that gets overwritten each time — this
   is intentional: it is a scratch pad, not a permanent record.
   Formal runs (and any parallel runs) must pass an explicit
   `--log-path` with a properly named directory; they must **not**
   write to `latest`.  This avoids race conditions when multiple runs
   execute concurrently — each writes to its own named directory and
   `latest` is never contested.  Historical experiments that still expose
   this knob as `--output-dir` follow the same rule under that name.
3. **No random suffix.**  Timestamp precision to the second is
   sufficient — this repo does not run large-scale parallel sweeps from
   the same experiment directory.  If that changes, append a 4-char
   random hex (e.g. `-a8f3`).
4. **Smoke runs use a `smoke-` prefix** on the mode segment:
   `smoke-eval`, `smoke-sft`.  This lets `ls artifacts/smoke-*` or
   `rm -rf artifacts/smoke-*` target all smoke runs without touching
   real runs.
5. **Metadata lives in files, not the dirname.**  The directory name
   carries enough context for quick identification; full configuration
   belongs in `run.json` (already written by `write_outputs`).  Do not
   encode every hyperparameter into the directory name.
6. **Adopt incrementally.**  Existing artifact directories in the repo
   do not need to be renamed.  Apply this convention to new runs going
   forward.

### Why this format

- **Human-friendly**: description first → `ls` output is scannable
  without opening any file.
- **Agent-friendly**: fixed date-time suffix → regex
  `(\d{8})-(\d{6})$` extracts timestamp from any dirname; an agent
  that needs the latest formal run sorts by that suffix rather than
  relying on a symlink.
- **Avoids common pitfalls**: no opaque UUIDs (MLflow/Guild AI style);
  no auto-incrementing integers that break across machines (Sacred
  style); no unbounded hyperparameter encoding that hits path-length
  limits (Ray Tune style).

## Artifact path naming

Within a run directory, phase comes first and artifact role comes second.
Do not repeat phase names in file basenames when the directory already
answers that question.

### Baseline eval layout

```text
run.json
console.log
eval/
├── examples.jsonl
├── predictions.jsonl
└── metrics.json
```

Rules:

1. Use `eval/` for eval artifacts and `train/` for train artifacts.
2. Keep the default eval snapshot flat: `eval/examples.jsonl`,
   `eval/predictions.jsonl`, `eval/metrics.json`.
3. Do **not** insert an extra `main/` layer unless one run truly stores
   multiple named eval snapshots.
4. Use role-based basenames: `examples`, `predictions`, `metrics`,
   `periodic`, `checkpoints`, `batches`.
5. When `run.json` stores artifact pointers, prefer the same nested shape:
   `artifacts.eval.examples`, `artifacts.eval.predictions`,
   `artifacts.eval.metrics`, `artifacts.train.metrics`, and so on.

### Research-grade additions

```text
eval/periodic.jsonl
train/metrics.jsonl
train/batches.jsonl
train/checkpoints.jsonl
```

Only add a deeper subdirectory when a single run truly needs multiple
named eval snapshots. In that case, add the extra layer under `eval/`
rather than inventing new basename prefixes.

## Section order

Every `train.py` uses the same top-level section layout, delimited by
`# ===== Section Name =====` headers.  The canonical order is:

```
# ===== Paths and constants =====
# ===== Local env loading =====
# ===== Runtime backend compatibility =====
# ===== CLI =====
# ===== Infrastructure helpers =====
# ===== Training helpers =====               (SFT, DPO, or GRPO — only when the experiment trains)
# ===== Task-specific helpers =====          (domain-specific, non-adapter helpers)
# ===== Task-specific adapters =====
# ===== Runtime entrypoints =====
# ===== Artifact writing =====
# ===== Entry point =====
```

**Rules**:

1. **Omit, don't reorder.**  If an experiment has no training helpers, drop that
   section entirely — do not move its functions into a later section.
2. **Define-before-use.**  Within each section and across sections, callees
   appear above their callers.  Training helpers
   (SFT: `build_supervised_datum`, `create_training_client`, `save_*`, `run_train`;
   GRPO: `reward_from_response`, `build_grpo_datums_from_rollout_groups`,
   `save_sampling_client`, `grpo_train_loop`) must appear *before*
   `Runtime entrypoints` — never after `main_async`.
3. **Domain helpers stay in their section.**  Experiment-specific functions
   that only serve eval belong in `Task-specific helpers` or inside
   `Task-specific adapters`, before the adapter that calls them.
   Experiment-specific functions that only serve training belong in
   `Training helpers` or `Task-specific helpers`, again before their
   caller.  For GRPO, reward shaping helpers (e.g. `compute_soft_overlong_penalty`,
   `classify_overlong_response`) and answer grading (e.g. `grade_math_answer`)
   go in `Task-specific helpers` when they are benchmark-specific; generic
   RL machinery (datum construction, rollout sampling, the train loop) stays
   in `Training helpers`.
4. **Infrastructure helpers are scaffold-level only.**  Do not place
   domain-specific functions (e.g. task-type dispatch helpers,
   `run_official_task_scorer`) in `Infrastructure helpers`; move them down
   to `Task-specific helpers`.

These will be corrected incrementally when each experiment is next touched for substantive changes.

### Common helper order

When two experiments share the same training workflow family, keep the shared
helpers in the same relative order so diffs stay structural instead of
stylistic.

For SFT-shaped loops, keep the promoted helper family in this order when the
functions exist:

```python
build_supervised_datum
compute_lr_multiplier
compute_total_train_steps
compute_mean_nll                # when forward_backward NLL logging is used
should_record_train_metrics_row
should_print_train_step
reset_supervised_append_streams
extract_api_path
build_state_save_name
checkpoint_save_names
step_to_loop_position
shuffled_train_rows_for_epoch
max_logged_step                 # optional compatibility helper
get_last_resumable_checkpoint
validate_resume_contract
resolve_resume_state
# task-local SFT batch helpers such as:
preview_text
count_text_tokens
build_supervised_batch_trace_record
build_supervised_row_datum
load_training_state
load_training_state_with_optimizer
create_training_client
save_weights_for_sampling
save_training_state
save_sampler_checkpoint
merge_eval_metrics_into_step_record
merge_optimizer_metrics_into_step_record
```

For DPO-shaped loops, keep the shared SFT-style schedule / resume / save
helpers in the same relative order where the behavior matches. The promoted
pairwise helper family typically includes:

```python
reset_dpo_append_streams
extract_api_path
build_state_save_name
checkpoint_save_names
step_to_loop_position
max_logged_step                 # optional compatibility helper
get_last_resumable_checkpoint
validate_resume_contract
resolve_resume_state
load_training_state
load_training_state_with_optimizer
create_training_client
save_weights_for_sampling
save_training_state
save_sampler_checkpoint
to_float_tensor
weighted_sequence_score
compute_dpo_loss
compute_logprobs
compute_logprobs_with_semaphore
align_logprob_sequence
tensor_like_length
# task-local DPO helpers such as:
build_epoch_batches
build_preference_datum
build_pair_payload
score_reference_pair_payloads
build_dpo_batch_trace_record
```

## Key function signatures

These signatures define the contract between infrastructure and task-specific logic.
When a function exists in an experiment, its name and signature shape should match.
Benchmark-specific exceptions are allowed when the benchmark contract truly requires them, such as official task-level scorers that need extra aggregation context or progress callbacks.

### Task-specific adapters (every experiment customizes these)

```python
def normalize_eval_rows(path: Path) -> list[dict[str, Any]]
def normalize_train_rows(path: Path) -> list[dict[str, Any]]
def build_eval_messages(row: dict[str, Any]) -> list[dict[str, str]]
def grade_assistant_text(assistant_text: str, row: dict[str, Any]) -> tuple[bool, str]
def compute_eval_metrics(predictions: list[dict[str, Any]]) -> dict[str, float]
```

Note: `compute_eval_metrics` is an **aggregation** function over the full prediction list.
Row-level scoring belongs in `grade_assistant_text`.

Accepted adapter variants:
- `grade_assistant_text` may return extended tuples (e.g. 3-tuple `(bool, str, str | None)`) when the grader needs to pass extra info such as an extracted label to downstream aggregation; the first two elements must keep the `(correct, prediction_text)` contract.
- `compute_eval_metrics` may accept extra keyword arguments (e.g. `task_type` in fingpt, `eval_rows` in lawbench) when benchmark-specific aggregation requires context beyond the prediction list.
- `build_eval_messages` may accept extra arguments (e.g. `task_type`, `eval_setting`) when prompt construction depends on the benchmark profile.
- Pairwise DPO experiments with held-out preference eval may replace the generation-oriented `build_eval_messages(...)` / `grade_assistant_text(...)` path with pairwise scorers such as `score_pair(...)`, while keeping `normalize_eval_rows(...)`, `normalize_train_rows(...)`, and `compute_eval_metrics(...)` explicit. In that shape, `run_eval(...)` owns the chosen-vs-rejected scoring contract.

### Runtime entrypoints

```python
def run_dry_run(args, *, eval_rows, overlap) -> int
async def run_eval(sampler, tokenizer, eval_rows, args) -> tuple[list[dict], list[dict], dict[str, Any]]
async def run_train(training_client, tokenizer, train_rows, eval_rows, args, output_dir) -> dict[str, float]
```

`run_eval` and `run_train` receive clients as parameters.
`main_async` handles client creation and dispatch.
Accepted variants: `experiments/dapo-aime` uses `evaluate_with_sampler` in place of `run_eval`.
Accepted variant: fingpt splits `eval_rows` into explicit `periodic_eval_name` + `periodic_eval_rows` parameters when multi-benchmark eval requires dataset routing during training.
Accepted variant: DPO may expand `run_train(...)` to accept `reference_client` and `reference_tokenizer` ahead of `train_rows` when pairwise loss needs a frozen reference model.

### Infrastructure helpers

The template is the naming spec. Use the same names in every experiment:

- `load_local_env(path)`
- `create_service_client(*, timeout)`
- `is_sampler_model_path(model_ref)` — recognizes both `tinker://` and `mint://` sampler-weight URIs
- `create_sampling_client(service_client, base_model)`
- `resolve_api_result(value)` / `resolve_api_result_async(value)`
- `cached_tokenizer_dir(model_name)` / `get_tokenizer(client, model_name)`
- `load_jsonl(path)` / `load_existing_jsonl(path)` / `load_records(path)` — `load_existing_jsonl(path)` returns `[]` for a missing or empty file before delegating to `load_jsonl(path)`
- `write_json(path, payload)` / `write_jsonl(path, rows)` / `append_jsonl(path, record)`
- `strip_internal_row_fields(row)`
- `build_eval_uid(*, eval_name, dataset_name, example_id)`
- `eval_example_artifact_row(row, *, messages, example_id, eval_name="final")`
- `prediction_artifact_rows(predictions)`
- `optional_git_provenance(repo_subdir)` — best-effort `git` HEAD + dirty flag for `run.json`
- `emit_metric_lines(metrics: dict[str, Any])` — filters bool / non-numeric entries before printing
- `resolve_data_path` is no longer used; data paths are passed via `--train-data` / `--eval-data` CLI args
- keep `--eval-data` explicit in `parse_args(...)`, docs, wrappers, and tests; standard split-layout experiments pass `data/eval/smoke.jsonl` for `--dry-run` validation and `data/eval/full.jsonl` for the benchmark, while multi-dataset experiments may use documented dataset-family specs, instead of auto-filling those paths in code
- `require_eval_data_arg(raw, *, dry_run)` — shared front-door validation for the explicit `--eval-data` contract before any loader-specific parsing happens
- `parse_eval_data_arg(raw)` — parse comma-separated `name:path` eval specs
- `extract_row_id(row, *, fallback)`
- `audit_overlap(train_ids, eval_ids)`
- `build_generation_prompt_tokens(tokenizer, messages)`
- `sample_assistant_text(sampler, tokenizer, messages, args)` when the helper returns a decoded string
- `sample_assistant_message(sampler, tokenizer, messages, args)` when the helper returns a structured assistant message first
- `collect_run_artifacts(log_path, *, include_append_streams)` — build the current `run.json` artifact index from files already present under `log_path`
- `write_outputs(log_path, *, args, eval_examples, predictions, metrics, extra_payload, include_existing_artifacts=True)`
- `TeeStream` — class that mirrors writes to multiple streams (stdout + console.log file)
- `prepare_run_dir(log_path)` — create run directory; symlink `artifacts/runs/latest` → run_dir for smoke runs only (name contains `smoke`)
- `write_run_metadata(log_path, args, *, status, started_at, ended_at, error)` — two-phase run.json: `status:"running"` at start, `status:"completed"/"failed"` at end
- `main_async()` / `main()`

Prefer names that match the returned layer:

- use `sample_assistant_text(...)` for simple prompt -> decoded-text eval loops such as the baseline template
- use `sample_assistant_message(...)` when the sampler path is message-first and `assistant_text` is extracted later from the message content

## Capability-specific names

Add these only when the experiment enables the matching capability.

**SFT** (see `scaffolds/profiles/sft.md`):
- `build_supervised_datum(tokenizer, prompt_tokens, assistant_text)` — shared response-only-loss datum constructor; benchmark-specific row/message rendering belongs in a differently named local wrapper such as `build_supervised_row_datum(...)`
- `create_training_client(service_client, args, *, resume_state_path)` — the `resume_state_path` keyword is the auto-resume handle discovered by `get_last_resumable_checkpoint(...)`; when it is `None`, the client optionally falls back to weights-only `--load-checkpoint-path`
- `load_training_state(training_client, state_path)` — shared weights-only restore helper used by `create_training_client(...)` for `--load-checkpoint-path`
- `load_training_state_with_optimizer(training_client, state_path)` — shared optimizer-state restore helper used by `create_training_client(...)` for automatic same-run resume
- `save_weights_for_sampling(training_client)`
- `save_training_state(training_client, save_name)`
- `save_sampler_checkpoint(training_client, save_name)`
- `checkpoint_save_names(base_name)` — derive distinct runtime save names for
  the resumable state export and the durable sampler export while keeping one
  logical checkpoint `name` in `train/checkpoints.jsonl`
- `compute_lr_multiplier(schedule, step, total_steps)`
- `compute_total_train_steps(n_rows, *, batch_size, num_epochs, max_steps)`
- `compute_mean_nll(fwd_bwd_result, datums) -> tuple[float, float]` — weighted NLL reduction after `forward_backward(..., loss_fn="cross_entropy")`
- `should_record_train_metrics_row(step, total, every, *, has_merged_eval)` — cadence gate for callers that already decided the streamed `train/metrics.jsonl` path is enabled; keep the same signature when periodic eval is **not** merged yet, pass `has_merged_eval=False` in that case, and treat non-positive `every` as cadence `1` instead of silently suppressing the first/final row
- `should_print_train_step(step, total, every)`
- `scalar_metric_items(metrics)` — filters bool / non-numeric entries before downstream logging
- `extract_api_path(payload)` — normalizes sampler / state path responses
- `build_state_save_name(base_model, log_path)`
- `shuffled_train_rows_for_epoch(train_rows, *, seed, epoch_idx)` — stable epoch-local row shuffler used by the current SFT experiments
- `PERIODIC_EVAL_SAMPLE_COUNT` — number of prediction samples to include in each `eval/periodic.jsonl` row

**DPO** (see `scaffolds/profiles/dpo.md`):
- `create_training_client(service_client, args, *, resume_state_path)` — shared with SFT; same automatic same-run resume handle and weights-only fallback to `--load-checkpoint-path`
- `load_training_state(training_client, state_path)` — shared with SFT; weights-only restore helper
- `load_training_state_with_optimizer(training_client, state_path)` — shared with SFT; optimizer-state restore helper for automatic same-run resume
- `save_weights_for_sampling(training_client)` — shared with SFT; export a live eval client from the current training weights
- `save_training_state(training_client, save_name)` — shared with SFT; save a resumable checkpoint
- `save_sampler_checkpoint(training_client, save_name)` — shared with SFT; save a durable sampler path for later eval-only reruns
- `checkpoint_save_names(base_name)` — shared with SFT; derive distinct runtime save names for the resumable state export and the durable sampler export
- `compute_lr_multiplier(schedule, step, total_steps)` — shared with SFT; keep cadence semantics aligned unless DPO intentionally changes the schedule meaning
- `compute_total_train_steps(n_rows, *, batch_size, num_epochs, max_steps, allow_partial_batch)` — DPO training step count, including optional last partial batch
- `should_record_train_metrics_row(step, total, every, *, has_merged_eval)` — shared with SFT; cadence gate for streamed `train/metrics.jsonl`
- `should_print_train_step(step, total, every)` — shared with SFT; cadence gate for step-summary stdout
- `scalar_metric_items(metrics)` — shared with SFT; filters bool / non-numeric entries before downstream logging
- `extract_api_path(payload)` — shared with SFT; normalize sampler / state path responses
- `build_state_save_name(base_model, log_path)` — shared with SFT; generate one logical checkpoint base name for the run, with GRPO implementations typically using a `-grpo` suffix
- `PERIODIC_EVAL_SAMPLE_COUNT` — number of periodic eval prediction previews to include in each `eval/periodic.jsonl` row
- `step_to_loop_position(step, *, n_batches, num_epochs)` — shared with SFT; map a completed optim step count to the next `(epoch_idx, batch_idx)` loop position
- `reset_dpo_append_streams(run_dir, *, resume_checkpoint=None, include_periodic_eval=True, include_checkpoints=True, include_train_metrics=True, include_batch_trace=True)` — DPO variant of the SFT append-stream reset helper; on fresh runs, truncate/create the enabled `eval/periodic.jsonl`, `train/checkpoints.jsonl`, `train/metrics.jsonl`, and optional `train/batches.jsonl`, and remove disabled leftovers from an older run in the same directory
- `build_epoch_batches(rows, *, batch_size, allow_partial_batch, batch_group_key, seed)` — DPO batch constructor that can spread repeated prompts or group IDs across batches
- `build_preference_datum(tokenizer, messages, completion, *, max_length)` — build one response-only-loss datum for a chosen or rejected continuation and return the full model input alongside it
- `build_pair_payload(tokenizer, row, *, max_length)` — package chosen + rejected datums, model inputs, and token counts for one preference pair
- `score_reference_pair_payloads(ref_logprob_seqs, pair_payloads)` — reduce reference-model token logprobs into one chosen and one rejected sequence score per pair
- `compute_dpo_loss(chosen_logprobs, rejected_logprobs, chosen_ref_logprobs, rejected_ref_logprobs, dpo_beta)` — return `(loss, metrics)` where metrics typically include `dpo_loss`, `accuracy`, `margin`, `chosen_reward`, and `rejected_reward`
- `build_dpo_batch_trace_record(batch_rows, *, limit=DEFAULT_BATCH_PAIR_RECORD_LIMIT)` — optional bounded pair-preview lineage record for `train/batches.jsonl`

**Research-grade SFT / DPO / long-run logging** (typical `$new-experiment-plus`, only when append-only streams exist):
- `get_last_resumable_checkpoint(run_dir)` — **shared with DPO and GRPO**; scan the current run directory's `train/checkpoints.jsonl` and return the latest row with a non-empty `state_path`, or `None`. This is the directory-driven trigger for automatic same-run resume and does not rely on any CLI flag
- `step_to_loop_position(step, *, n_batches, num_epochs)` — SFT/DPO-specific; map a completed optim step count to the next `(epoch_idx, batch_idx)` loop position. GRPO uses `completed_steps` / `next_step` directly and does not need this helper
- `reset_supervised_append_streams(run_dir, *, resume_checkpoint=None, include_periodic_eval=True, include_checkpoints=True, include_train_metrics=True, include_batch_trace=True)` — SFT variant; on fresh runs, truncate/create the enabled run-scoped streams and remove disabled stream files left behind by an older run in the same directory; preserve everything when `resume_checkpoint is not None`. Typical callers set the booleans from the enabled cadence flags so `--eval-every` alone only prepares `eval/periodic.jsonl`, `--save-every` alone only prepares `train/checkpoints.jsonl`, and `--train-metrics-every` enables both streamed `train/metrics.jsonl` and optional prompt-lineage `train/batches.jsonl`. The DPO equivalent is `reset_dpo_append_streams(run_dir, *, resume_checkpoint=None, include_periodic_eval=True, include_checkpoints=True, include_train_metrics=True, include_batch_trace=True)` with the same keyword shape but pairwise-eval semantics; the GRPO equivalent is `reset_rl_append_streams(run_dir, *, resume_checkpoint=None)` with the RL-specific stream list (`train/metrics.jsonl`, `train/rollouts.jsonl`, `train/failures.jsonl`, `train/failures.log`, `train/checkpoints.jsonl`, `eval/metrics.jsonl`)
- `validate_resume_contract(run_dir, args, *, resume_checkpoint)` — **shared with DPO and GRPO**; automatic same-run resume guard. When `resume_checkpoint is not None`, reject mismatched run-defining args recorded in `run_dir/run.json` before append-only logs continue; missing `run.json` is tolerated. The `RUN_DEFINING_ARGS` list is experiment-local (SFT: `base_model`, `train_data`, `eval_data`, `seed`, `rank`, `learning_rate`, `num_epochs`, `max_steps`, `batch_size`, `lr_schedule`, `eval_every`, `save_every`; DPO: the SFT list plus pairwise knobs such as `reference_model`, `dpo_beta`, `max_length`, `batch_group_key`, `allow_partial_batch`; GRPO: `base_model`, `train_data`, `eval_data`, `seed`, `rank`, `grpo_steps`, `groups_per_batch`, `group_size`, `rl_learning_rate`, `rl_loss`, `eval_every_steps`, `save_every_steps`, plus any RL-specific dataset or scorer knobs). It does **not** require run-scoped append-only JSONL to be exactly aligned to the last checkpoint
- `resolve_resume_state(run_dir, resume_checkpoint, *, total_steps, n_batches, num_epochs)` — SFT/DPO-specific; return `(start_step, start_epoch, start_batch)` from the matched checkpoint row, consistency-checked against `step_to_loop_position(...)`. GRPO reads `resume_checkpoint["completed_steps"]` directly inside `grpo_train_loop` instead of going through a dedicated helper. The checkpoint row in `train/checkpoints.jsonl` is the primary restore source for all three profiles, not checkpoint path text
- `merge_eval_metrics_into_step_record(step_record, eval_metrics, *, prefix="test/")` — fold held-out eval scalars into a train step dict (e.g. `test/eval_*` keys); both `lawbench` and `fingpt` implement this shape. GRPO uses the `eval/` prefix instead of `test/`
- `merge_optimizer_metrics_into_step_record(step_record, optim_metrics, *, prefix="optim/")` — merge backend optimizer metrics into a train step dict without overwriting canonical loop fields such as cookbook-managed `step`, `epoch`, `batch`, `progress`, or `learning_rate`; colliding backend keys should be namespaced under `optim/`
- `build_supervised_batch_trace_record(...)` — optional per-step SFT prompt / assistant-text lineage record for `train/batches.jsonl`; prefer first-few prompt / assistant-text previews plus batch-level token summaries when prompt visibility matters, analogous to RL rollout traces. DPO's pairwise analogue is `build_dpo_batch_trace_record(...)`
- `optional_git_provenance(repo_subdir)` — best-effort `git_commit` / `git_worktree_dirty` for `run.json`; optional per experiment until promoted

**Stdout markers (automation / AI-native)**:
- **`METRIC name=value`** — required primary benchmark scalars (repo contract).
- **`ANALYSIS_MANIFEST path=…`** — optional; if an experiment adopts this token when writing `analysis_manifest.json`, update `AGENTS.md`, `scaffolds/README.md`, and skills in the same commit.

**GRPO** (see `scaffolds/profiles/grpo.md`):
- `async grpo_train_loop(service_client, training_client, tokenizer, train_rows, eval_rows, args, output_dir, *, resume_checkpoint=None)` — core RL loop: sample → score → build datums → train → log → eval → checkpoint. `resume_checkpoint` is the raw row dict returned by `get_last_resumable_checkpoint(...)`; the loop derives `start_step` from `resume_checkpoint["completed_steps"]`. `experiments/dapo-aime` maps that row into an internal `ResumeLoopState` before calling its `grpo_train_loop`; new GRPO experiments should pass the row dict directly as `resume_checkpoint`
- `reward_from_response(response, gold_answer)` — per-response reward; returns `(reward, extracted_answer, correct, format_valid)`
- `build_grpo_datums_from_rollout_groups(rollout_groups, tokenizer, args)` — GRPO datum construction with group advantage normalization; returns `(datums, metrics, rollout_records)`
- `make_prompt_group_batch(rows, step, groups_per_batch)` — select prompt groups for a training step
- `create_training_client(service_client, args, *, resume_state_path)` — shared with SFT; the GRPO scaffold uses the same keyword-driven shape. Callers pass the matched checkpoint's `state_path` (or `None`) as `resume_state_path`; when `resume_state_path` is `None`, the helper optionally falls back to weights-only `--load-checkpoint-path`. `experiments/dapo-aime` matches this keyword shape for training; `--eval-only` uses `create_sampling_client(..., args.base_model)` with a `sampler_path`, like the SFT/DPO experiments
- `load_training_state(training_client, state_path)` — shared with SFT; weights-only restore for `--load-checkpoint-path`
- `load_training_state_with_optimizer(training_client, state_path)` — shared with SFT; optimizer-state restore for automatic same-run resume in SFT, DPO, and GRPO scaffold defaults
- `provision_training_client(service_client, args, *, resume_state_path=None)` — GRPO-specific runtime wrapper for extras such as timeout, create-time logging, or RL-specific retry policies; forwards `resume_state_path` to the shared `create_training_client(...)` rather than inventing a GRPO-local `resume_from` keyword
- `reset_rl_append_streams(run_dir, *, resume_checkpoint=None)` — GRPO equivalent of `reset_supervised_append_streams`; truncate the RL run-scoped streams (`train/metrics.jsonl`, `train/rollouts.jsonl`, `train/failures.jsonl`, `train/failures.log`, `train/checkpoints.jsonl`, `eval/metrics.jsonl`) on fresh runs; preserve them on automatic same-run resume. Shared name because the shape mirrors SFT; the stream list is the only intentional divergence
- `save_training_state(training_client, save_name)` — shared with SFT; save a resumable checkpoint
- `save_sampler_checkpoint(training_client, save_name)` — shared with SFT; save a durable sampler path for later offline eval
- `checkpoint_save_names(base_name)` — shared with SFT; derive distinct runtime
  save names for the resumable state export and the durable sampler export
- `save_sampling_client(training_client, name)` — GRPO-specific live sampler export for rollout generation; use a distinct name because it returns a sampler client, not a durable path
- `save_weights_for_sampling(training_client)` — shared with SFT; export weights to a live sampler for immediate eval
- `append_periodic_eval_record(path, *, step, total_steps, eval_output_dir, status, sampler_path=None, eval_metrics=None, error=None)` — append RL periodic eval record to `eval/metrics.jsonl`
- `is_train_rollout_failure(record)` — detect rollout failures for paired failure logging
- `save_recovery_checkpoint(training_client, step)` — GRPO-specific wrapper that usually calls `save_training_state(...)` + `save_sampler_checkpoint(...)` together and logs both artifacts

If the same helper name appears in SFT, DPO, and GRPO, keep the implementation
semantically identical. Historical GRPO names such as
`save_state_async_compat(...)` and `save_weights_for_sampler_async_compat(...)`
should converge toward `save_training_state(...)` and
`save_sampler_checkpoint(...)` rather than forcing SFT helpers to adopt the
older GRPO-specific names.
The same applies to client creation: if GRPO needs runtime orchestration around
the shared `create_training_client(...)` helper, add a wrapper such as
`provision_training_client(...)` rather than changing the shared helper's
meaning under the same name.

## Checkpoint CLI semantics

When an experiment supports checkpoint restore, keep same-run resume and
fresh-run weight loading as two separate meanings:

- **Automatic same-run resume** (SFT/DPO scaffold default): rerun the same
  training command with the same run-dir flag. The canonical scaffold
  exposes that directory via `--log-path`; historical experiments may
  still call the same knob `--output-dir`, depending on the experiment's
  run-dir flag. The scaffold reads the latest row with a non-empty
  `state_path` from that run directory's `train/checkpoints.jsonl`,
  restores optimizer plus loop state through a fresh LoRA training client and
  `load_state_with_optimizer(...)`, and continues the same append-only
  registries (`train/checkpoints.jsonl`, `train/metrics.jsonl`,
  optional `train/batches.jsonl`, `eval/periodic.jsonl`) and `console.log`.
  No dedicated `--resume-from` flag is reserved for this SFT/DPO default.
- `--load-checkpoint-path` — start a fresh run from saved weights only; do
  not reuse optimizer state, do not reuse the previous run's append-only
  logs, and do not treat it as same-run continuation. A resumable
  checkpoint already present in the current run directory always takes
  priority over `--load-checkpoint-path`.

Scaffold default across profiles:

- The SFT profile (`scaffolds/profiles/sft.md`), the DPO profile
  (`scaffolds/profiles/dpo.md`), and the GRPO profile
  (`scaffolds/profiles/grpo.md`) all describe automatic same-run resume
  as the directory-driven default. RL adds its own loop-state shape
  (`completed_steps`, `next_step`) on the same checkpoint row, but the
  trigger mechanism — rerunning with the same run directory — is the
  same.

Experiment-local variants:

- `experiments/dapo-aime` follows the same directory-driven same-run resume
  and `--load-checkpoint-path` shape as LawBench/FinGPT/Chat-DPO; `--eval-only`
  takes a `sampler_path` as `--base-model` only. The `ResumeLoopState`
  dataclass remains internal glue; new GRPO code should prefer the raw
  checkpoint row dict from `get_last_resumable_checkpoint(...)`.

Checkpoint registry rules for promoted templates:

- Do not make the scaffold depend on parsing step numbers from checkpoint path text.
- `train/checkpoints.jsonl` is the restore contract and must carry the loop state needed for same-run continuation.
- Rows should usually carry a human-readable logical `name`.
- For SFT/DPO, the matched row should carry at least:
  - `state_path`
  - `step`
  - `epoch`
  - `batch`
- For RL, the matched row should carry at least:
  - `state_path`
  - `completed_steps`
  - `next_step`
- When one logical checkpoint row exports both `state_path` and `sampler_path`,
  keep the runtime save names distinct, for example `<name>-state` and
  `<name>-sampler`.
- `sampler_path` is the durable eval export, not the same-run resume handle.
- Automatic same-run resume should not be gated on `train/metrics.jsonl`,
  `train/batches.jsonl`, or `eval/periodic.jsonl` being exactly aligned to
  the last checkpoint; those streams are append-only diagnostics that may
  slide past the last resumable row.

## Task-specific names

These stay local to each experiment:
- dataset-specific row fields
- task-specific grader helpers
- tool schema constants
- benchmark-specific metric names
- reward functions
