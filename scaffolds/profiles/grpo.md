# GRPO Profile

Add RL optimization with rollout collection after the eval baseline is stable.
This is a training extension layered on top of the eval profile, not a standalone starting point.

## Where This Profile Sits

- `scaffolds/README.md` owns the repo-wide split between templates, profiles, experiments, and skills.
- `scaffolds/single_file_experiment/naming.md` owns promoted helper names once a pattern is shared.
- `skills/new-experiment` owns baseline bootstrap.
- `skills/new-experiment-plus` owns research-grade logging and long-run harness guidance.
- Optional `analysis_manifest.json` + stdout `ANALYSIS_MANIFEST path=...` is not part of the minimal profile sketch until promoted repo-wide.

## What To Add

Replace the `raise NotImplementedError` block in `main_async` with a GRPO training path. The typical shape mirrors the SFT profile's directory-driven same-run resume (`scaffolds/profiles/sft.md`), just with RL's own loop-state shape (`completed_steps`, `next_step`) and RL-flavored append-only streams:

```python
# ---- GRPO training ----
# Automatic same-run resume: scan the current run directory's
# train/checkpoints.jsonl for the latest row that carries a resumable
# state_path. Compute this BEFORE opening console.log so the console can
# be opened in append mode, and before creating the training client so
# the restored state_path can be passed in.
resume_checkpoint = get_last_resumable_checkpoint(run_dir)
validate_resume_contract(run_dir, args, resume_checkpoint=resume_checkpoint)
resume_state_path = (
    str(resume_checkpoint.get("state_path") or "").strip()
    if resume_checkpoint is not None
    else ""
)

training_client = await create_training_client(
    service_client,
    args,
    resume_state_path=resume_state_path or None,
)
tokenizer = get_tokenizer(training_client, args.base_model)

# Truncate RL run-scoped append-only JSONL on fresh runs, preserve them
# on automatic same-run resume. Keep the checkpoint semantics split:
# - automatic same-run resume: a resumable checkpoint in the current run
#   directory takes priority, preserving the existing append-only logs
# - --load-checkpoint-path: load weights into a fresh run (no resumable
#   checkpoint present), so truncate the fresh run's append-only logs
reset_rl_append_streams(run_dir, resume_checkpoint=resume_checkpoint)

rl_metrics, periodic_eval_history, cached_final_eval, final_sampler_path = await grpo_train_loop(
    service_client, training_client, tokenizer, train_rows, eval_rows, args, run_dir,
    resume_checkpoint=resume_checkpoint,
)

if cached_final_eval is not None:
    eval_metrics = cached_final_eval
    predictions = None  # already written by background eval
    eval_examples = None
else:
    sampler = await save_weights_for_sampling(training_client)
    eval_examples, predictions, eval_metrics = await run_eval(
        sampler, tokenizer, eval_rows, args
    )

write_outputs(
    run_dir,
    args=args,
    eval_examples=eval_examples,
    predictions=predictions,
    metrics=eval_metrics,
    extra_payload={"sampler_path": final_sampler_path},
)
emit_metric_lines({**rl_metrics, **eval_metrics})
```

When a `resume_checkpoint` is found, open `console.log` in append mode so the resumed run continues the same file; a fresh run opens it in write mode and replaces the previous content:

```python
console_log_mode = "a" if resume_checkpoint is not None else "w"
console_log_handle = (run_dir / "console.log").open(
    console_log_mode,
    encoding="utf-8",
    buffering=1,
)
```

The canonical scaffold exposes the run directory via `--log-path`; historical experiments may still call the same knob `--output-dir`, depending on the experiment's run-dir flag. The auto-resume trigger rule is the same: rerun the same training command with the same run directory.

## GRPO Concepts

| Term | Meaning | Origin |
|------|---------|--------|
| `groups_per_batch` | Number of prompt groups sampled per training step | dapo-aime / tinker `Config` |
| `group_size` | Number of trajectories (responses) generated per prompt | dapo-aime / tinker `Config` |
| `trajectory` | One complete model response to a prompt | tinker `TrajectoryGroup` |
| `rollout` | A prompt's full set of trajectories (= one prompt group result) | dapo `PromptSamplingResult` |
| `advantage` | GRPO group-normalized reward: `(r_i - mean(r)) / std(r)` | GRPO paper |
| `datum` | Tinker SDK training unit: tokens + weights/mask + advantages | `tinker.types.Datum` |
| `sampler` / `sampling_client` | Inference client holding current trained weights | tinker SDK |
| `sampler_step` | Training step at which the sampler was created; tracks staleness | dapo-aime / tinker |
| `stream_minibatches_per_step` | How many forward_backward calls overlap with rollout collection within one step | dapo / tinker `StreamMinibatchConfig` |
| `tail_grace_seconds` | Grace period for slow rollouts before early-stopping the step | dapo-aime |

## Functions To Implement

### `get_last_resumable_checkpoint`, `validate_resume_contract`, `reset_rl_append_streams` — directory-driven same-run resume

These helpers share the exact same shape as the SFT profile (see `scaffolds/profiles/sft.md`) and are the directory-driven entry points for automatic same-run resume. Only `reset_rl_append_streams` is RL-specific because the append-only stream set is different.

`get_last_resumable_checkpoint(run_dir)` — scan `train/checkpoints.jsonl` for the latest row with a non-empty `state_path`; return the row (so callers can read `completed_steps`, `next_step`, `state_path`) or `None`. Shared implementation with SFT.

`validate_resume_contract(run_dir, args, *, resume_checkpoint)` — when `resume_checkpoint is not None`, reject mismatched run-defining args recorded in `run_dir/run.json` before append-only logs continue. Missing `run.json` is tolerated. Run-scoped append-only logs sliding past the last checkpoint are not by themselves a resume failure; this profile does not gate resume on `train/metrics.jsonl`, `train/rollouts.jsonl`, `train/failures.jsonl`, or `eval/metrics.jsonl` being exactly aligned to the last checkpoint. The `RUN_DEFINING_ARGS` list is experiment-local; for GRPO it typically covers `base_model`, `train_data`, `eval_data`, `seed`, `rank`, `grpo_steps`, `groups_per_batch`, `group_size`, `rl_learning_rate`, `rl_loss`, `eval_every_steps`, `save_every_steps`, plus any RL-specific dataset or scorer knobs the experiment uses.

`reset_rl_append_streams(run_dir, *, resume_checkpoint=None)` — truncate the RL run-scoped append-only streams on fresh runs; preserve them on automatic same-run resume:

```python
def reset_rl_append_streams(
    run_dir: Path,
    *,
    resume_checkpoint: dict[str, Any] | None = None,
) -> None:
    if resume_checkpoint is not None:
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    for path in (
        run_dir / "train" / "metrics.jsonl",
        run_dir / "train" / "rollouts.jsonl",
        run_dir / "train" / "failures.jsonl",
        run_dir / "train" / "failures.log",
        run_dir / "train" / "checkpoints.jsonl",
        run_dir / "eval" / "metrics.jsonl",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
```

Call it from `main_async` before `grpo_train_loop`, not inside the loop. Use the same name because the shape mirrors SFT's `reset_supervised_append_streams`; the RL-specific stream list is the only intentional divergence.

### `grpo_train_loop` — core RL loop

The loop follows this per-step pattern: sample → score → build datums → train → log → eval → checkpoint.

```python
async def grpo_train_loop(
    service_client, training_client, tokenizer, train_rows, eval_rows,
    args, output_dir, *, resume_checkpoint: dict[str, Any] | None = None,
) -> tuple[dict[str, float], list[dict[str, Any]], dict[str, float] | None, str | None]:
    # Derive RL loop state from the matched checkpoint row. The row is None
    # on fresh runs and on --load-checkpoint-path runs.
    start_step = int(resume_checkpoint["completed_steps"]) if resume_checkpoint else 0

    # --- Artifact paths ---
    metrics_jsonl_path = output_dir / "train" / "metrics.jsonl"
    train_rollouts_jsonl_path = output_dir / "train" / "rollouts.jsonl"
    train_failures_jsonl_path = output_dir / "train" / "failures.jsonl"
    train_failures_log_path = output_dir / "train" / "failures.log"
    checkpoints_jsonl_path = output_dir / "train" / "checkpoints.jsonl"
    periodic_eval_metrics_path = output_dir / "eval" / "metrics.jsonl"

    # Fresh-run truncation is handled by reset_rl_append_streams(run_dir,
    # resume_checkpoint=...) in main_async before grpo_train_loop starts;
    # do not repeat that logic here.

    sampler = await save_sampling_client(training_client, name=f"grpo-step-{start_step}")
    sampler_step = start_step

    for step in range(start_step + 1, args.grpo_steps + 1):
        step_started_at = time.time()

        # 1. Sample rollouts: one prompt group per prompt, group_size trajectories each
        prompt_group_jobs = make_prompt_group_batch(train_rows, step, args.groups_per_batch)
        rollout_results = await sample_all_prompt_groups(
            sampler, tokenizer, prompt_group_jobs, args,
        )

        # 2. Score trajectories and build GRPO datums
        datums, step_metrics, rollout_records = build_grpo_datums_from_rollout_groups(
            rollout_results, tokenizer, args,
        )

        # 3. Forward-backward + optimizer step
        if datums:
            fwd_bwd_future = await training_client.forward_backward_async(
                datums, loss_fn=args.rl_loss,  # "importance_sampling" or "ppo"
            )
            optim_future = await training_client.optim_step_async(
                types.AdamParams(learning_rate=args.rl_learning_rate),
            )
            await resolve_api_result_async(fwd_bwd_future)
            await resolve_api_result_async(optim_future)

        # 4. Save new sampler for next step
        sampler = await save_sampling_client(training_client, name=f"grpo-step-{step}")
        sampler_step = step

        # 5. Log metrics
        step_elapsed = time.time() - step_started_at
        metrics_record = {
            "step": step, "total_steps": args.grpo_steps,
            "learning_rate": args.rl_learning_rate,
            "progress": step / args.grpo_steps,
            "reward_mean": step_metrics["rl_reward_mean"],
            "accuracy": step_metrics["rl_group_accuracy"],
            "format": step_metrics["rl_final_answer_format_success"],
            "datums": int(step_metrics["rl_datums_per_step"]),
            "num_trajectories": int(step_metrics["rl_samples_per_step"]),
            "time/step": step_elapsed,
            ...
        }
        append_jsonl(metrics_jsonl_path, metrics_record)

        # 6. Log rollouts + failures
        for record in rollout_records:
            append_jsonl(train_rollouts_jsonl_path, record)
            if is_train_rollout_failure(record):
                append_jsonl(train_failures_jsonl_path, record)

        state_name_prefix = build_state_save_name(args.base_model, output_dir)

        # 7. Periodic checkpoint
        if args.save_every_steps > 0 and step % args.save_every_steps == 0:
            checkpoint_name = f"{state_name_prefix}-step-{step:06d}"
            state_path, sampler_path = await save_recovery_checkpoint(training_client, step)
            append_jsonl(checkpoints_jsonl_path, {
                "name": checkpoint_name,
                "step": step, "completed_steps": step,
                "state_path": state_path, "sampler_path": sampler_path,
            })

        # 8. Periodic eval (background)
        if args.eval_every_steps > 0 and step % args.eval_every_steps == 0:
            eval_sampler = await save_weights_for_sampling(training_client)
            eval_output_dir = output_dir / "eval" / "steps" / f"step-{step:06d}"
            eval_examples, predictions, eval_metrics = await run_eval(
                eval_sampler, tokenizer, eval_rows, args
            )
            write_jsonl(eval_output_dir / "examples.jsonl", eval_examples)
            write_jsonl(eval_output_dir / "predictions.jsonl", predictions)
            write_json(eval_output_dir / "metrics.json", eval_metrics)
            append_periodic_eval_record(periodic_eval_metrics_path,
                step=step, total_steps=args.grpo_steps,
                eval_output_dir=eval_output_dir, status="completed",
                eval_metrics=eval_metrics)

    return last_metrics, periodic_eval_history, cached_final_eval, final_sampler_path
```

### `reward_from_response` — per-response reward

```python
def reward_from_response(
    response: str, gold_answer: str,
) -> tuple[float, str | None, bool, bool]:
    """Score one trajectory response.

    Returns: (reward, extracted_answer, correct, format_valid)
    """
    extracted = parse_final_answer(response)
    format_valid = extracted.answer is not None
    correct = grade_math_answer(extracted.answer, gold_answer) if format_valid else False
    reward = 1.0 if correct else 0.0
    return reward, extracted.answer, correct, format_valid
```

### `build_grpo_datums_from_rollout_groups` — datum construction

For each prompt group with G trajectories and rewards `r_1, ..., r_G`:

1. Compute group advantage: `advantage_i = (r_i - mean(r)) / (std(r) + eps)`
2. Skip groups where all rewards are identical (no gradient signal)
3. Build one `types.Datum` per trajectory with importance_sampling loss:
   - `model_input`: prompt + response tokens (shifted by 1)
   - `loss_fn_inputs`: `target_tokens`, `logprobs` (from sampling), `mask` (response-only), `advantages`

```python
def build_grpo_datums_from_rollout_groups(
    rollout_groups: list[PromptSamplingResult],
    tokenizer: Any,
    args: argparse.Namespace,
) -> tuple[list[types.Datum], dict[str, float], list[dict[str, Any]]]:
    datums = []
    rollout_records = []
    for rollout in rollout_groups:
        rewards, trajectory_data = score_all_trajectories(rollout, tokenizer, args)
        # GRPO advantage normalization
        mean_r, std_r = mean(rewards), std(rewards)
        if std_r < 1e-8:  # constant reward → no gradient signal
            rollout_records.append(make_rollout_record(rollout, status="no_signal"))
            continue
        advantages = [(r - mean_r) / (std_r + 1e-8) for r in rewards]
        for traj, adv in zip(trajectory_data, advantages):
            datums.append(build_importance_sampling_datum(traj, adv))
        rollout_records.append(make_rollout_record(rollout, status="ok"))
    metrics = aggregate_step_metrics(rollout_groups, rollout_records)
    return datums, metrics, rollout_records
```

### Shared functions (same as SFT)

These are shared with the SFT profile and the scaffold template:

- `create_training_client(service_client, args, *, resume_state_path)` — create a LoRA training client; when `resume_state_path` is not `None`, restore optimizer plus state via `load_state_with_optimizer(...)`; when it is `None`, optionally fall back to weights-only `--load-checkpoint-path`
- `load_training_state(training_client, state_path)` — restore weights only for `--load-checkpoint-path`
- `load_training_state_with_optimizer(training_client, state_path)` — restore weights plus optimizer state for automatic same-run resume
- `save_training_state(training_client, save_name)` — save resumable checkpoint
- `save_sampler_checkpoint(training_client, save_name)` — save a durable sampler path for later offline eval
- `build_state_save_name(base_model, log_path)` — shared with SFT; generate one logical checkpoint base name for the run, with the GRPO implementation using a `-grpo` suffix
- `checkpoint_save_names(base_name)` — shared with SFT; derive distinct runtime save names for the resumable state export and the durable sampler export behind one logical checkpoint `name`
- `save_weights_for_sampling(training_client)` — export weights to a live sampler for eval

Do not re-sketch these shared helpers with GRPO-specific edits under the same
name. Reuse the exact implementations from `scaffolds/profiles/sft.md`; that
includes the shared checkpoint semantics:

- automatic same-run resume (scaffold default for GRPO, SFT, and DPO): rerun the same training command with the same run directory; `train.py` reads the latest row with a non-empty `state_path` from `train/checkpoints.jsonl`, restores optimizer plus loop state through a fresh LoRA training client and `load_state_with_optimizer(...)`, and continues the same append-only registries plus `console.log`. No dedicated `--resume-from` flag is reserved for this default.
- `--load-checkpoint-path` means load weights into a fresh run with fresh optimizer state and fresh append-only logs. A resumable checkpoint in the current run directory always takes priority over `--load-checkpoint-path`.

For MinT-style resume, keep the shared flow uniform across SFT, DPO, and GRPO:

- create a fresh LoRA training client with `create_lora_training_client(...)`
- then call `load_state_with_optimizer(...)` using the `resume_state_path` derived from `get_last_resumable_checkpoint(run_dir)`

Do not make checkpoint path text the scaffold's loop-state oracle. The matched
`train/checkpoints.jsonl` row is the restore contract. Numeric checkpoint-name
fallbacks are experiment-local compatibility shims, not promoted scaffold behavior.

If the RL path needs extra behavior, introduce a GRPO-specific helper with a
different name such as `save_sampling_client(...)` or
`save_recovery_checkpoint(...)`.
The same rule applies to training-client orchestration: keep
`create_training_client(...)` on the shared SFT contract, and put GRPO-only
extras such as timeout wrapping, create-time logging, or RL-specific
retry policies in a differently named wrapper such as
`provision_training_client(service_client, args, *, resume_state_path=None)`
that still forwards `resume_state_path` to the shared helper rather than
inventing a GRPO-local `resume_from` keyword.
For example, `save_recovery_checkpoint(...)` can be a thin GRPO wrapper that
calls `save_training_state(...)` plus `save_sampler_checkpoint(...)` and then
logs both returned paths together.

## Section Placement

GRPO experiments follow the same `# ===== Section =====` layout as SFT (see `naming.md`). This table shows where each function category lands:

| Section | What goes here (GRPO) |
|---------|----------------------|
| **Infrastructure helpers** | Shared scaffold functions: `load_local_env`, `create_service_client`, `create_sampling_client`, `resolve_api_result_async`, `cached_tokenizer_dir`, `get_tokenizer`, `load_jsonl`, `append_jsonl`, `write_json`, `write_jsonl`, `emit_metric_lines`, `build_generation_prompt_tokens`, `TeeStream`, `prepare_run_dir`, `write_run_metadata` |
| **Training helpers** | All RL machinery, define-before-use order. Shared with SFT: `create_training_client`, `save_training_state`, `save_sampler_checkpoint`, `save_weights_for_sampling`, `get_last_resumable_checkpoint`, `validate_resume_contract`, `reset_rl_append_streams`. GRPO-specific: `provision_training_client`, `save_sampling_client`, rollout dataclasses (`PromptSamplingJob`, `PromptSamplingResult`), `save_recovery_checkpoint`, `make_prompt_group_batch`, `reward_from_response`, `score_response_tokens`, `group_lacks_reward_signal`, `validate_grpo_datum_inputs`, `build_grpo_datums_from_rollout_groups`, SDK wrappers (`enqueue_forward_backward_async`, `enqueue_optim_step_async`), rollout sampling (`sample_one_async`, `sample_many_async`), `grpo_train_loop` (last — calls everything above). `experiments/dapo-aime` maps the resumable checkpoint **row** into an internal `ResumeLoopState` for `grpo_train_loop`; new GRPO experiments should pass the raw row dict as `resume_checkpoint` instead. |
| **Task-specific helpers** | Benchmark-specific logic: answer parsing (`ParsedFinalAnswer`, `parse_final_answer`), text normalization (`normalize_answer`), grading (`grade_math_answer`), reward shaping (`compute_soft_overlong_penalty`, `classify_overlong_response`, `shape_sequence_reward`), prompt construction (`build_math_rl_prompt`, `build_math_rl_messages`) |
| **Task-specific adapters** | Standard 5: `normalize_train_rows`, `normalize_eval_rows`, `build_eval_messages`, `grade_assistant_text`, `compute_eval_metrics` |
| **Runtime entrypoints** | `evaluate_with_sampler` / `run_eval`, `run_dry_run`, `main_async`, `main` |
| **Artifact writing** | `write_outputs`, `append_periodic_eval_record`, `is_train_rollout_failure`, `write_train_failure_record`, formatting helpers (`format_eval_metric_summary`, `format_log_block`) |

**Key ordering rule**: within `Training helpers`, place callees above callers. Typical order: dataclasses → shared training clients → reward/scoring → datum construction → sampling primitives → `grpo_train_loop`.

## CLI Flags To Add

These are sketch defaults; see `experiments/dapo-aime/train.py` for the current live values:

```python
# --- Algorithm knobs ---
parser.add_argument("--grpo-steps", type=int, default=100)
parser.add_argument("--groups-per-batch", type=int, default=8,
    help="Number of prompt groups per training step.")
parser.add_argument("--group-size", type=int, default=8,
    help="Number of trajectories (responses) per prompt group.")
parser.add_argument("--rl-learning-rate", type=float, default=1e-4)
parser.add_argument("--rl-temperature", type=float, default=0.7,
    help="Sampling temperature for rollout generation.")
parser.add_argument("--rl-max-tokens", type=int, default=4096,
    help="Max tokens per trajectory.")
parser.add_argument("--rl-loss", choices=("importance_sampling", "ppo"), default="importance_sampling")
parser.add_argument("--rank", type=int, default=16)

# --- Eval and checkpoint cadence ---
parser.add_argument("--eval-every-steps", type=int, default=1,
    help="Run eval every N completed training steps; 0 disables periodic eval.")
parser.add_argument("--save-every-steps", type=int, default=0,
    help="Save checkpoint every N steps; 0 = only at end.")
parser.add_argument("--load-checkpoint-path", default="",
    help="Start a fresh run from saved weights only; ignored when the current run directory already has a resumable state_path.")

# --- Throughput knobs (separate from algorithm) ---
parser.add_argument("--max-concurrent-requests", type=int, default=32,
    help="Fan-out for async rollout sampling.")
parser.add_argument("--stream-minibatches-per-step", type=int, default=4,
    help="Submit forward_backward as soon as N-th fraction of rollouts arrives.")
parser.add_argument("--min-accepted-groups", type=int, default=0,
    help="Minimum groups needed before tail grace starts; 0 = auto.")
parser.add_argument("--min-started-minibatches", type=int, default=0,
    help="Minimum minibatches submitted before tail grace starts; 0 = auto.")
parser.add_argument("--tail-grace-seconds", type=float, default=90.0,
    help="Seconds to wait for remaining slow rollouts once min thresholds are met.")

# --- Dynamic sampling (optional) ---
parser.add_argument("--dynamic-sampling-type", choices=("none", "filter"), default="none",
    help="'filter' drops constant-reward groups and re-samples replacements.")
parser.add_argument("--dynamic-sampling-max-rollout-waves", type=int, default=0,
    help="Max re-sampling waves when filtering; 0 = unlimited.")
```

Same-run resume does **not** need a dedicated CLI flag in the scaffold default: rerun the same training command with the same run directory (the canonical scaffold exposes it as `--log-path`; historical experiments may still call the same knob `--output-dir`, depending on the experiment's run-dir flag) and `train.py` auto-detects the latest resumable `state_path` in `train/checkpoints.jsonl`. `--load-checkpoint-path` is training-only and must be rejected when the current run is `--eval-only` or `--dry-run`.

## GRPO Artifacts

```
artifacts/runs/{run-name}/
├── run.json                     # two-phase metadata (baseline)
├── console.log                  # TeeStream stdout+stderr (baseline)
├── train/
│   ├── metrics.jsonl            # per-step RL metrics
│   ├── rollouts.jsonl           # per-group rollout records
│   ├── failures.jsonl           # rollout failures (structured)
│   ├── failures.log             # rollout failures (human-readable)
│   └── checkpoints.jsonl        # checkpoint registry
└── eval/
    ├── metrics.jsonl            # periodic eval metrics
    ├── steps/                   # per-step eval details
    │   └── step-NNNN/
    │       ├── examples.jsonl
    │       ├── predictions.jsonl
    │       ├── metrics.json
    │       ├── failures.log
    │       └── failures.jsonl
    ├── examples.jsonl           # final eval input snapshot
    ├── predictions.jsonl        # final eval predictions
    └── metrics.json             # final eval aggregate metrics
```

### `train/metrics.jsonl` schema (one row per completed step)

```json
{
  "step": 1,
  "total_steps": 180,
  "learning_rate": 1e-4,
  "progress": 0.0056,
  "reward_mean": 0.25,
  "accuracy": 0.30,
  "format": 0.98,
  "datums": 384,
  "num_trajectories": 384,
  "prompt_groups": 32,
  "accepted_groups": 32,
  "trained_groups": 32,
  "dropped_groups": 0,
  "early_stop": 0,
  "group_size": 8,
  "groups_per_batch": 32,
  "samples_per_sec": 12.5,
  "time/step": 25.6,
  "time/total": 25.6,
  "time/eta": 4582.4,
  "checkpoint_name": null,
  "checkpoint_state_path": null,
  "checkpoint_sampler_path": null
}
```

### `train/rollouts.jsonl` schema (one row per prompt group)

```json
{
  "step": 1,
  "row_index": 1,
  "sampler_step": 0,
  "id": "dapo-math-0042",
  "question": "Find the sum of all positive integers...",
  "gold_answer": "42",
  "num_trajectories": 8,
  "num_correct": 3,
  "num_formatted": 8,
  "num_datums": 8,
  "reward_mean": 0.375,
  "reward_min": 0.0,
  "reward_max": 1.0,
  "status": "ok",
  "trajectory_answers": ["42", null, "42", "7", "42", "7", "7", "7"],
  "trajectory_correct": [true, false, true, false, true, false, false, false],
  "trajectory_rewards": [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
  "trajectory_advantages": [1.83, -0.61, 1.83, -0.61, 1.83, -0.61, -0.61, -0.61],
  "trajectory_token_counts": [512, 1024, 480, 900, 550, 800, 750, 680],
  "first_trajectory_text": "Let me think step by step..."
}
```

### `train/checkpoints.jsonl` schema

```json
{
  "name": "step-000005",
  "step": 5,
  "completed_steps": 5,
  "next_step": 6,
  "state_path": "tinker://...",
  "sampler_path": "tinker://..."
}
```

Checkpoint registry rules:

- automatic same-run resume reads the latest row with a non-empty
  `state_path` from the current run directory's `train/checkpoints.jsonl`
  via `get_last_resumable_checkpoint(run_dir)`; no CLI flag is required
  to select that row
- that matched row is the loop-state source of truth: `completed_steps`
  for `start_step`, `next_step` for the next unstarted step
- one logical row `name` may back both exports, but the runtime save names for
  `state_path` and `sampler_path` should stay distinct, for example
  `<name>-state` and `<name>-sampler`
- `sampler_path` is for later eval, not for same-run resume
- parsing step numbers from checkpoint names is not part of the promoted
  scaffold contract
- run-scoped append-only logs (`train/metrics.jsonl`, `train/rollouts.jsonl`,
  `train/failures.jsonl`, `eval/metrics.jsonl`) may slide past the last
  checkpoint row; that is not by itself a resume failure, and the scaffold
  does not gate resume on those streams being exactly aligned to the last
  checkpoint

## Throughput Layers

GRPO throughput has multiple independent layers. Separate algorithm knobs from throughput knobs to avoid confusing algorithm changes with system tuning:

| Layer | Knob | What it controls |
|-------|------|------------------|
| Sampling shape | `group_size` | Trajectories per prompt (algorithm) |
| Batch shape | `groups_per_batch` | Prompts per step (algorithm) |
| Request fan-out | `max_concurrent_requests` | In-flight sampling requests (throughput) |
| In-step overlap | `stream_minibatches_per_step` | Submit forward_backward before all rollouts arrive (throughput) |
| Tail control | `min_accepted_groups`, `min_started_minibatches`, `tail_grace_seconds` | When to stop waiting for slow rollouts (throughput) |
| Dynamic sampling | `dynamic_sampling_type`, `max_rollout_waves` | Re-sample to replace constant-reward groups (algorithm/throughput hybrid) |
| Train-eval overlap | Background eval tasks | Eval runs while next step samples (throughput) |

**Prefer throughput fixes in this order:**
1. Stream minibatches within a step (immediate win, no algorithm change)
2. Tail early-stop for slow/invalid groups
3. Bounded async rollout fan-out
4. Dynamic sampling to replace flat-reward groups

## Merging Periodic Eval

When GRPO runs periodic eval, append both to `eval/metrics.jsonl` (sidecar) and optionally merge key eval scalars into the `train/metrics.jsonl` step record:

```python
if eval_metrics is not None:
    for k, v in scalar_metric_items(eval_metrics).items():
        metrics_record[f"eval/{k}"] = v
```

The `eval/` prefix (rather than SFT's `test/`) distinguishes held-out eval from training reward metrics in the same row. Readers that want pure eval history use the sidecar; readers that want a single flat metrics stream use the merged row.

## Resume And Advanced Checkpoint Support

Basic periodic checkpointing (`--save-every-steps`) can follow the same post-step pattern as SFT. The checkpoint record in `train/checkpoints.jsonl` must include a logical `name`, `state_path` (resumable), `sampler_path` (eval-ready), and the loop-state fields shown above. If one logical checkpoint exports both artifacts, keep their runtime save names distinct.

Automatic same-run resume and fresh-run weight loading (`--load-checkpoint-path`) are research-grade GRPO features and belong to `$new-experiment-plus`. When an experiment adopts them, follow the checkpoint-row contract above and keep `train/checkpoints.jsonl` as the restore source of truth:

- Same-run resume is triggered by rerunning the same training command with
  the same run directory. The canonical scaffold exposes that directory via
  `--log-path`; historical experiments may still call the same knob
  `--output-dir`, depending on the experiment's run-dir flag. `train.py`
  reads the last resumable `state_path` in that directory's
  `train/checkpoints.jsonl`, restores optimizer plus loop state through a
  fresh LoRA training client and `load_state_with_optimizer(...)`, and
  continues the existing append-only registries and `console.log`.
- Fresh run from saved weights is triggered by `--load-checkpoint-path`; it
  takes effect only when no resumable checkpoint is present in the current
  run directory, does not reuse optimizer state, and truncates the
  run-scoped JSONL plus `console.log` at the fresh-run entry point.

See `skills/new-experiment-plus/references/upgrade_playbook.md` for guidance.

## GRPO ↔ tinker-cookbook Mapping

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

## Current Repo Example

- `experiments/dapo-aime` — research-grade GRPO with streaming minibatches, dynamic sampling, background eval, failure logging, and the same directory-driven same-run resume plus `--load-checkpoint-path` pattern as LawBench/FinGPT/Chat-DPO; `--eval-only` passes a `sampler_path` as `--base-model`. The internal `ResumeLoopState` dataclass is experiment-local glue, not the promoted checkpoint-row contract for new experiments.

## README Result Reporting

When a GRPO experiment `README.md` reports measured runs, keep the result section compact but complete:

- start `Current results` with `Status: \`placeholder\`` until a checked run exists, and switch to `Status: \`checked\`` only when the reported run is actually checked
- record the eval config with the relevant `max_concurrent_requests`
- record the smallest set of RL training knobs needed to interpret the run, such as `grpo_steps`, `groups_per_batch`, `group_size`, `rl_learning_rate`, and `rank`
- report wall-clock timing, not a mixed timing convention
- if eval timing is batch wall-clock or depends on parallel vs sequential execution, say so explicitly
- if periodic eval uses pre-step or post-step sampler snapshots and that affects interpretation, say so explicitly
- treat throughput as supplementary, not as a replacement for wall-clock timing
