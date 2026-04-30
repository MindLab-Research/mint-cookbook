# SFT Profile

Add supervised fine-tuning after the eval baseline is stable.
This is a training extension layered on top of the eval profile, not a standalone starting point.

## Where This Profile Sits

- `scaffolds/README.md` owns the repo-wide split between templates, profiles, experiments, and skills.
- `scaffolds/single_file_experiment/naming.md` owns promoted helper names once a pattern is shared.
- `skills/new-experiment` owns baseline bootstrap and points at the scaffold-owned artifact contract.
- `skills/new-experiment-plus` owns long-run operability guidance, not the canonical baseline artifact names.
- Optional `analysis_manifest.json` + stdout `ANALYSIS_MANIFEST path=...` is not part of the minimal profile sketch until promoted repo-wide.

## What To Add

The canonical template now already carries the shared SFT CLI flags
(`--rank`, `--learning-rate`, `--num-epochs`, `--max-steps`,
`--batch-size`, `--lr-schedule`, `--eval-every`, `--save-every`,
`--train-metrics-every`, `--train-print-every`, `--load-checkpoint-path`)
plus the shared helper family (`load_existing_jsonl`,
`build_supervised_datum`, `compute_total_train_steps`,
`compute_lr_multiplier`, `get_last_resumable_checkpoint`,
`validate_resume_contract`, `resolve_resume_state`,
`create_training_client`, `save_training_state`,
`save_sampler_checkpoint`, `save_weights_for_sampling`, and friends).
It also now wires a generic SFT baseline into `main_async` plus a
generic `run_train`, `build_supervised_row_datum`, and
`build_supervised_batch_trace_record`.
This profile is about replacing or specializing those generic pieces for
benchmark-specific row rendering, periodic-eval routing, and richer
logging policy.

The canonical template already wires a minimal SFT training path. When a
benchmark needs different periodic-eval routing, row rendering, or richer
logging, keep the same shape:

```python
# ---- Training ----
# Automatic same-run resume: scan the current --log-path/train/checkpoints.jsonl
# for the latest row that carries a resumable state_path. Compute this BEFORE
# opening console.log so the console can be opened in append mode and before
# creating the training client so the restored state_path can be passed in.
# Gate on `not args.eval_only` so eval-only reruns in the same run directory
# never reopen training-side state (eval-only should use --base-model with a
# saved sampler_path, not pick up the state_path from an earlier SFT run).
resume_checkpoint = get_last_resumable_checkpoint(run_dir) if not args.eval_only else None
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

# Optional append-stream logging only: if the train path uses
# eval/periodic.jsonl / train/checkpoints.jsonl / streamed train/metrics.jsonl
# and optional prompt-lineage streams such as train/batches.jsonl, truncate
# those streams on fresh runs. Keep the checkpoint semantics split:
# - automatic same-run resume: a resumable checkpoint in the current run directory takes priority,
#   preserving the existing append-only logs and continuing the same run
# - --load-checkpoint-path: load weights into a fresh run (no resumable checkpoint present),
#   so truncate the fresh run's append-only logs
reset_supervised_append_streams(
    run_dir,
    resume_checkpoint=resume_checkpoint,
    include_periodic_eval=int(args.eval_every) > 0,
    include_checkpoints=bool(resume_checkpoint is not None or int(args.save_every) > 0),
    include_train_metrics=int(args.train_metrics_every) > 0,
    include_batch_trace=int(args.train_metrics_every) > 0,
)

train_metrics = await run_train(training_client, tokenizer, train_rows, eval_rows, args, run_dir)

sampler = await save_weights_for_sampling(training_client)
eval_examples, predictions, eval_metrics = await run_eval(sampler, tokenizer, eval_rows, args)

write_outputs(
    run_dir,
    args=args,
    eval_examples=eval_examples,
    predictions=predictions,
    metrics=eval_metrics,
)
emit_metric_lines({**train_metrics, **eval_metrics})
```

The canonical scaffold exposes the run directory via `--log-path`; some
concrete experiments expose the same directory under a different flag name
(historical experiments may still use `--output-dir`). Wherever this profile uses
`run_dir`, the underlying knob is whichever run-dir flag the experiment
keeps, and the automatic same-run resume rule is the same: rerun the same
training command with the same run directory.

When a `resume_checkpoint` is found, open `console.log` in append mode so the
resumed run continues the same file; a fresh run opens it in write mode and
replaces the previous content:

```python
console_log_mode = "a" if resume_checkpoint is not None else "w"
console_log_handle = (run_dir / "console.log").open(
    console_log_mode,
    encoding="utf-8",
    buffering=1,
)
```

## Functions To Wire Or Customize

Checkpoint-restore semantics follow the `tinker-cookbook` split, but same-run
resume is **automatic** rather than flag-driven. The current endpoint uses this
repo's supported API surface:

- Automatic same-run resume means: if the current run directory's
  `train/checkpoints.jsonl` already holds a row with a non-empty `state_path`,
  rerunning the same training command continues that run. `train.py` restores
  optimizer plus loop state from the matched row and continues the same
  append-only registries. A resumable checkpoint in the current run directory
  always takes priority over `--load-checkpoint-path`.
- `--load-checkpoint-path` means fresh run from saved weights only. It runs only
  when no resumable checkpoint is present in the current run directory, does not
  reuse optimizer state, and does not reuse the previous run's registries.
- Because same-run resume is directory-driven, this profile does not reserve a
  dedicated `--resume-from` flag. The DPO profile (`scaffolds/profiles/dpo.md`)
  and the GRPO profile (`scaffolds/profiles/grpo.md`) use the same
  directory-driven default. `experiments/dapo-aime` matches
  that contract; `--eval-only` uses `--base-model` with a `sampler_path`, not
  a separate resume flag.

Use the tinker-cookbook restore split directly:

- same-run resume reads loop state from the matched checkpoint row
- fresh checkpoint loading restores weights only
- checkpoint path text is not the scaffold's loop-state oracle

For MinT-style same-run resume, keep the control flow uniform across
experiments: create a fresh LoRA training client with
`create_lora_training_client(...)`, then call `load_state_with_optimizer(...)`
on it. Do not rely on a one-shot
`create_training_client_from_state_with_optimizer_async(...)` entrypoint as
the default resume path — that shape is not guaranteed across the supported
runtimes.

`get_last_resumable_checkpoint(run_dir)` - scan
`train/checkpoints.jsonl` for the latest row that records a non-empty
`state_path`; return the row (so callers can read `step`, `epoch`, `batch`,
`state_path`) or `None` when no resumable checkpoint exists. Missing or
empty files are expected on fresh runs, not an error:

```python
def get_last_resumable_checkpoint(run_dir: Path) -> dict[str, Any] | None:
    checkpoints_path = run_dir / "train" / "checkpoints.jsonl"
    checkpoint_rows = load_existing_jsonl(checkpoints_path)
    resumable_rows = [row for row in checkpoint_rows if str(row.get("state_path") or "").strip()]
    if not resumable_rows:
        return None
    return resumable_rows[-1]
```

`step_to_loop_position(step, *, n_batches, num_epochs)` - derive the next
`(epoch_idx, batch_idx)` loop position from a completed optim step count:

```python
def step_to_loop_position(step: int, *, n_batches: int, num_epochs: int) -> tuple[int, int]:
    if step < 0:
        raise RuntimeError(f"step must be non-negative, got {step}")
    if n_batches <= 0:
        raise RuntimeError(f"n_batches must be positive, got {n_batches}")
    full_schedule_steps = n_batches * max(num_epochs, 0)
    if step >= full_schedule_steps:
        return num_epochs, 0
    return step // n_batches, step % n_batches
```

`validate_resume_contract(run_dir, args, *, resume_checkpoint)` - when a
resumable row is present, reject mismatched run-defining args recorded in
`run.json` so a continued run cannot silently change batch size, seed,
dataset paths, etc. Missing `run.json` (older resume targets) is tolerated:

```python
def validate_resume_contract(
    run_dir: Path,
    args: argparse.Namespace,
    *,
    resume_checkpoint: dict[str, Any] | None,
) -> None:
    if resume_checkpoint is None:
        return
    run_json_path = run_dir / "run.json"
    if not run_json_path.is_file():
        return
    prior_args = json.loads(run_json_path.read_text(encoding="utf-8")).get("args")
    if not isinstance(prior_args, dict):
        return
    mismatches = [
        f"{key}: expected {prior_args[key]!r}, got {current!r}"
        for key, current in RUN_DEFINING_ARGS(args).items()
        if key in prior_args and prior_args[key] != current
    ]
    if mismatches:
        raise RuntimeError(
            "Automatic same-run resume requires the same run-defining args recorded in "
            f"{run_json_path}: " + "; ".join(mismatches)
        )
```

`RUN_DEFINING_ARGS(args)` is the experiment-local list of args that define
a run (for SFT usually `base_model`, `train_data`, `eval_data`, `seed`,
`rank`, `learning_rate`, `num_epochs`, `max_steps`, `batch_size`,
`lr_schedule`, `eval_every`, `save_every`, plus any extra dataset path the
experiment adds). Run-scoped append-only logs sliding past the last
checkpoint row are not by themselves a resume failure; this profile does
not gate resume on `train/metrics.jsonl`, `train/batches.jsonl`, or
`eval/periodic.jsonl` being exactly aligned to the last checkpoint.

`resolve_resume_state(run_dir, resume_checkpoint, *, total_steps, n_batches, num_epochs)` -
return `(start_step, start_epoch_idx, start_batch_idx)` from the matched
checkpoint row, consistency-checked against `step_to_loop_position(...)`:

```python
def resolve_resume_state(
    run_dir: Path,
    resume_checkpoint: dict[str, Any] | None,
    *,
    total_steps: int,
    n_batches: int,
    num_epochs: int,
) -> tuple[int, int, int]:
    if resume_checkpoint is None:
        return 0, 0, 0

    parsed: dict[str, int] = {}
    for key in ("step", "epoch", "batch"):
        try:
            parsed[key] = int(resume_checkpoint[key])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"Resume checkpoint row is missing integer `{key}`") from exc
        if parsed[key] < 0:
            raise RuntimeError(f"Resume checkpoint row contains negative `{key}`={parsed[key]}")

    expected_epoch, expected_batch = step_to_loop_position(
        parsed["step"], n_batches=n_batches, num_epochs=num_epochs
    )
    if (parsed["epoch"], parsed["batch"]) != (expected_epoch, expected_batch):
        raise RuntimeError(
            "Resume checkpoint row has inconsistent loop state: "
            f"step={parsed['step']} implies epoch={expected_epoch}, batch={expected_batch}, "
            f"but row stores epoch={parsed['epoch']}, batch={parsed['batch']}"
        )
    if parsed["step"] >= total_steps:
        raise RuntimeError(
            "The latest resumable checkpoint in this run directory already reached this run's configured total_steps"
        )
    return parsed["step"], parsed["epoch"], parsed["batch"]
```

`reset_supervised_append_streams(run_dir, *, resume_checkpoint, include_periodic_eval, include_checkpoints, include_train_metrics, include_batch_trace)` - reset
run-scoped append-only JSONL on fresh runs, preserve them on automatic
same-run resume:

```python
def reset_supervised_append_streams(
    run_dir: Path,
    *,
    resume_checkpoint: dict[str, Any] | None,
    include_periodic_eval: bool = True,
    include_checkpoints: bool = True,
    include_train_metrics: bool = True,
    include_batch_trace: bool = True,
) -> None:
    if resume_checkpoint is not None:
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    for path, enabled in (
        (run_dir / "eval" / "periodic.jsonl", include_periodic_eval),
        (run_dir / "train" / "checkpoints.jsonl", include_checkpoints),
        (run_dir / "train" / "metrics.jsonl", include_train_metrics),
        (run_dir / "train" / "batches.jsonl", include_batch_trace),
    ):
        if enabled:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("", encoding="utf-8")
        elif path.exists():
            path.unlink()
```

`load_training_state(training_client, state_path)` - shared weights-only helper
used by `create_training_client(...)` for `--load-checkpoint-path`:

```python
async def load_training_state(
    training_client: Any,
    state_path: str,
) -> None:
    load_fn = getattr(training_client, "load_state_async", None)
    if callable(load_fn):
        await resolve_api_result_async(load_fn(state_path))
        return
    load_fn = getattr(training_client, "load_state", None)
    if not callable(load_fn):
        raise RuntimeError("Training client must expose load_state")
    resolve_api_result(load_fn(state_path))
```

`load_training_state_with_optimizer(training_client, state_path)` - shared
resume helper used by `create_training_client(...)`:

```python
async def load_training_state_with_optimizer(
    training_client: Any,
    state_path: str,
) -> None:
    load_fn = getattr(training_client, "load_state_with_optimizer_async", None)
    if callable(load_fn):
        await resolve_api_result_async(load_fn(state_path))
        return
    load_fn = getattr(training_client, "load_state_with_optimizer", None)
    if not callable(load_fn):
        raise RuntimeError("Training client must expose load_state_with_optimizer")
    resolve_api_result(load_fn(state_path))
```

`create_training_client(service_client, args, *, resume_state_path)` - create a
LoRA training client, then optionally restore either optimizer state for
automatic same-run resume (`resume_state_path` passed in by the caller after it
ran `get_last_resumable_checkpoint(...)`) or weights only for
`--load-checkpoint-path`. Keeping the resume state as an explicit keyword
argument rather than reading `args` lets the caller own the auto-resume
priority rule (a resumable checkpoint in the current `--log-path` always
beats `--load-checkpoint-path`):

```python
async def create_training_client(
    service_client: Any,
    args: argparse.Namespace,
    *,
    resume_state_path: str | None,
) -> Any:
    load_checkpoint_path = str(getattr(args, "load_checkpoint_path", "") or "").strip()
    client_kwargs = {
        "base_model": args.base_model,
        "rank": int(args.rank),
        "train_mlp": True,
        "train_attn": True,
        "train_unembed": True,
    }
    create_fn = getattr(service_client, "create_lora_training_client_async", None)
    if callable(create_fn):
        training_client = await resolve_api_result_async(create_fn(**client_kwargs))
    else:
        create_fn = getattr(service_client, "create_lora_training_client", None)
        if not callable(create_fn):
            raise RuntimeError("Service client must expose create_lora_training_client")
        training_client = resolve_api_result(create_fn(**client_kwargs))

    if resume_state_path:
        print(f"Resuming from saved state: {resume_state_path}")
        await load_training_state_with_optimizer(training_client, resume_state_path)
    elif load_checkpoint_path:
        print(f"Loading weights from checkpoint: {load_checkpoint_path}")
        await load_training_state(training_client, load_checkpoint_path)
    return training_client
```

`build_supervised_datum(tokenizer, prompt_tokens, assistant_text)` - convert a prepared prompt-token prefix plus assistant target text into a response-only-loss datum. Keep benchmark-specific row/message rendering in a differently named local wrapper (for example `build_supervised_row_datum(...)`):

```python
def build_supervised_datum(
    tokenizer: Any,
    prompt_tokens: list[int],
    assistant_text: str,
) -> Any:
    completion_tokens = tokenizer.encode(assistant_text, add_special_tokens=False)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None:
        completion_tokens.append(int(eos_token_id))
    all_tokens = list(prompt_tokens) + completion_tokens
    if len(all_tokens) < 2:
        raise RuntimeError("Need at least two tokens to build supervised datum")
    weights = [0.0] * len(prompt_tokens) + [1.0] * len(completion_tokens)
    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens=all_tokens[:-1]),
        loss_fn_inputs={"target_tokens": all_tokens[1:], "weights": weights[1:]},
    )
```

Typical row/message wrappers:

```python
def build_supervised_row_datum(tokenizer: Any, row: dict[str, Any]) -> Any:
    prompt_tokens = tokenizer.encode(str(row["prompt"]), add_special_tokens=True)
    return build_supervised_datum(tokenizer, prompt_tokens, str(row["assistant_text"]))
```

```python
def build_supervised_row_datum(
    tokenizer: Any,
    row: dict[str, Any],
    task_type: str,
) -> Any:
    prompt_tokens = build_generation_prompt_tokens(
        tokenizer, build_eval_messages(row, task_type)
    )
    return build_supervised_datum(tokenizer, prompt_tokens, str(row["output"]))
```

`compute_lr_multiplier(schedule, step, total_steps)` - LR scheduling:

```python
def compute_lr_multiplier(schedule: str, step: int, total_steps: int) -> float:
    if schedule == "linear":
        return 1 - step / total_steps
    if schedule == "cosine":
        return 0.5 * (1 + math.cos(math.pi * step / total_steps))
    raise RuntimeError(f"Unknown lr_schedule: {schedule}")
```

`run_train(training_client, tokenizer, train_rows, eval_rows, args, output_dir)` - minibatch training loop with LR scheduling, periodic eval, and periodic checkpoint.

When you add run-scoped JSONL (`eval/periodic.jsonl`, `train/checkpoints.jsonl`, streamed `train/metrics.jsonl`, optional `train/batches.jsonl`), implement `reset_supervised_append_streams(run_dir, *, resume_checkpoint=None, include_periodic_eval=True, include_checkpoints=True, include_train_metrics=True, include_batch_trace=True)` beside `append_jsonl`: on fresh runs it truncates/creates the enabled files under `run_dir` and removes disabled stream files left behind by an older run in the same directory, but preserves everything when `resume_checkpoint` is not `None` so an automatic same-run resume keeps its history. Call it from `main_async` before `run_train`, not inside the loop, and set the booleans from the enabled cadence flags (for example `--eval-every` alone should prepare only `eval/periodic.jsonl`, `--save-every` alone should prepare only `train/checkpoints.jsonl`, while `--train-metrics-every` enables both `train/metrics.jsonl` and `train/batches.jsonl`). Baseline SFT that only returns a small summary dict and overwrites one metrics blob does not need this helper. Pair that reset step with explicit `validate_resume_contract(...)` and `resolve_resume_state(...)` helpers so automatic same-run resume restores loop state from the matched checkpoint row in `train/checkpoints.jsonl` instead of restarting step numbering from zero.

When the train loop shuffles rows each epoch, make the shuffle order a pure
function of `(seed, epoch_idx)` so resuming from `(epoch, batch)` does not
change the remaining batch order.

Checkpoint row contract for SFT:

- `name` is the logical checkpoint label recorded in `train/checkpoints.jsonl`
- `step` means completed optimizer steps
- `epoch` and `batch` identify the next loop position to execute
- `state_path` is the same-run resume handle
- `sampler_path` is the durable eval handle
- when one logical checkpoint exports both `state_path` and `sampler_path`,
  use distinct runtime save names such as `<name>-state` and `<name>-sampler`
- `final: true` may be present on the terminal row, but it does not replace the
  loop-state fields above

```python
async def run_train(training_client, tokenizer, train_rows, eval_rows, args, output_dir) -> dict[str, float]:
    n_batches = len(train_rows) // args.batch_size

    total_steps = n_batches * args.num_epochs
    if args.max_steps > 0:
        total_steps = min(total_steps, args.max_steps)

    # Automatic same-run resume: read the matched checkpoint row from
    # train/checkpoints.jsonl (step + next epoch/batch). The matched row is
    # None when the current output_dir has no resumable state_path yet, i.e.
    # either a fresh run or a --load-checkpoint-path run.
    resume_checkpoint = get_last_resumable_checkpoint(output_dir)
    start_step, start_epoch_idx, start_batch_idx = resolve_resume_state(
        output_dir,
        resume_checkpoint,
        total_steps=total_steps,
        n_batches=n_batches,
        num_epochs=args.num_epochs,
    )

    final_loss = float("inf")
    step = start_step
    for epoch_idx in range(start_epoch_idx, args.num_epochs):
        shuffled = build_epoch_rows(train_rows, seed=int(args.seed), epoch_idx=epoch_idx)
        batch_start_idx = start_batch_idx if epoch_idx == start_epoch_idx else 0
        for batch_idx in range(batch_start_idx, n_batches):
            if args.max_steps > 0 and step >= args.max_steps:
                break

            learning_rate = args.learning_rate * compute_lr_multiplier(args.lr_schedule, step, total_steps)
            batch = shuffled[batch_idx * args.batch_size : (batch_idx + 1) * args.batch_size]
            datums = [build_supervised_row_datum(tokenizer, row) for row in batch]

            fwd_bwd_result = await resolve_api_result_async(
                training_client.forward_backward_async(datums, loss_fn="cross_entropy"),
            )
            final_loss = ...
            await resolve_api_result_async(
                training_client.optim_step_async(types.AdamParams(learning_rate=learning_rate)),
            )
            step += 1

            if args.save_every > 0 and step % args.save_every == 0:
                checkpoint_name = f"step-{step:06d}"
                state_save_name, sampler_save_name = checkpoint_save_names(checkpoint_name)
                state_path = await save_training_state(training_client, state_save_name)
                sampler_path = await save_sampler_checkpoint(training_client, sampler_save_name)
                append_jsonl(output_dir / "train" / "checkpoints.jsonl", {
                    "name": checkpoint_name,
                    "step": step,
                    "epoch": epoch_idx,
                    "batch": batch_idx + 1,
                    "state_path": state_path,
                    "sampler_path": sampler_path,
                })
            if args.eval_every > 0 and step % args.eval_every == 0:
                sampler = await save_weights_for_sampling(training_client)
                preds, metrics = await run_eval(sampler, tokenizer, eval_rows, args)
                sample_preds = [
                    {k: v for k, v in p.items()
                     if k in ("id", "prediction", "correct", "expected", "assistant_text")}
                    for p in preds[:PERIODIC_EVAL_SAMPLE_COUNT]
                ]
                append_jsonl(output_dir / "eval" / "periodic.jsonl", {"step": step, **metrics, "samples": sample_preds})
        if args.max_steps > 0 and step >= args.max_steps:
            break

    return {"train_mean_nll": final_loss}
```

Eval timing convention:

- `--eval-every N` triggers after completing step `N` (post-step weights).
- `--save-every` also triggers post-step.
- Tinker-cookbook may use pre-step eval for pipelining, but post-step eval is the default in this repo unless an experiment documents otherwise.

`save_training_state(training_client, save_name)` - save a resumable checkpoint with async-first API:

```python
async def save_training_state(training_client: Any, save_name: str) -> str | None:
    save_fn = getattr(training_client, "save_state_async", None)
    if callable(save_fn):
        return extract_api_path(await resolve_api_result_async(save_fn(name=save_name)))
    save_fn = getattr(training_client, "save_state", None)
    if callable(save_fn):
        return extract_api_path(resolve_api_result(save_fn(name=save_name)))
    print("warning: training client does not expose save_state(); skipping checkpoint")
    return None
```

`save_sampler_checkpoint(training_client, save_name)` - export a durable sampler path for later offline eval:

```python
async def save_sampler_checkpoint(training_client: Any, save_name: str) -> str | None:
    fn = getattr(training_client, "save_weights_for_sampler_async", None)
    if callable(fn):
        return extract_api_path(await resolve_api_result_async(fn(name=save_name)))
    fn = getattr(training_client, "save_weights_for_sampler", None)
    if callable(fn):
        return extract_api_path(resolve_api_result(fn(name=save_name)))
    print("warning: training client does not expose save_weights_for_sampler(); skipping sampler checkpoint")
    return None
```

`checkpoint_save_names(base_name)` - derive distinct runtime save names for the
resumable state export and the durable sampler export while keeping one logical
checkpoint `name` in `train/checkpoints.jsonl`:

```python
def checkpoint_save_names(base_name: str) -> tuple[str, str]:
    return f"{base_name}-state", f"{base_name}-sampler"
```

`save_weights_for_sampling(training_client)` - export trained weights to a live sampler client for immediate eval:

```python
async def save_weights_for_sampling(training_client: Any) -> Any:
    fn = getattr(training_client, "save_weights_and_get_sampling_client_async", None)
    if callable(fn):
        return await resolve_api_result_async(fn(name="eval"))
    fn = getattr(training_client, "save_weights_and_get_sampling_client", None)
    if callable(fn):
        return resolve_api_result(fn(name="eval"))
    raise RuntimeError(
        "Training client must expose save_weights_and_get_sampling_client"
    )
```

These shared helpers are the canonical implementations for SFT, DPO, and GRPO
profiles when the function name matches:

- `create_training_client`
- `save_training_state`
- `save_sampler_checkpoint`
- `save_weights_for_sampling`

If a GRPO sketch needs different behavior, give it a different helper name
instead of silently forking one of these shared functions under the same name.

If you also need a durable checkpoint path for later offline eval, log it separately with `save_sampler_checkpoint(...)` and store the returned `sampler_path` beside any resumable `state_path`. Keep the row's logical `name` stable in `train/checkpoints.jsonl`, but do not reuse that same runtime save name for both exports.

## CLI Flags To Add

```python
parser.add_argument("--rank", type=int, default=16)
parser.add_argument("--learning-rate", type=float, default=1e-4)
parser.add_argument("--num-epochs", type=int, default=1)
parser.add_argument("--max-steps", type=int, default=0, help="0 = train full epochs")
parser.add_argument("--batch-size", type=int, default=8)
parser.add_argument("--lr-schedule", choices=("linear", "cosine"), default="linear")
parser.add_argument("--eval-every", type=int, default=0, help="Eval every N completed steps; 0 = only after training")
parser.add_argument("--save-every", type=int, default=0, help="Checkpoint every N steps; 0 = only after training")
parser.add_argument("--train-metrics-every", type=int, default=0, help="Append train/metrics.jsonl every N optim steps; 0 disables the streamed train/metrics.jsonl + train/batches.jsonl path")
parser.add_argument("--train-print-every", type=int, default=1, help="Print Step ... every N steps; 0 disables")
parser.add_argument("--load-checkpoint-path", default="", help="Start a fresh run from saved weights only; ignored when the current --log-path already has a resumable state_path")
```

Same-run resume does **not** need a dedicated CLI flag: rerun the same
training command with the same `--log-path` and `train.py` detects the
latest resumable `state_path` in `train/checkpoints.jsonl` automatically.
`--load-checkpoint-path` is training-only and must be rejected when the
current run is `--eval-only` or `--dry-run`.

Prefer structured rows in `train/metrics.jsonl` and optional stdout throttled by `--train-print-every`. When prompt visibility matters, add a parallel `train/batches.jsonl` stream with one row per completed step and bounded prompt/assistant-text previews (see `build_supervised_batch_trace_record` in `naming.md`). This row is **optional per experiment**: start without it, add it when prompt lineage becomes necessary (current examples: lawbench and fingpt both implement this stream).

## Merging Periodic Eval Into The Step Record

When `--eval-every N > 0`, run the periodic eval **before** assembling the
`train/metrics.jsonl` row so the held-out metrics can be folded into the same
step **when that stream is enabled**. Use
`merge_eval_metrics_into_step_record(step_record, eval_metrics)` to prefix the
eval scalars with `test/`, then gate the append with
`train_metrics_enabled = args.train_metrics_every > 0`. Once the streamed
`train/metrics.jsonl` path is enabled, keep merged-eval rows via
`should_record_train_metrics_row(..., has_merged_eval=True)` so eval steps are
not dropped by the scalar cadence filter. If the metrics stream is disabled,
periodic eval should still append to `eval/periodic.jsonl` only.

```python
step_eval_metrics: dict[str, Any] | None = None
if args.eval_every > 0 and step % args.eval_every == 0:
    sampler = await save_weights_for_sampling(training_client)
    _, step_eval_metrics = await run_eval(sampler, tokenizer, eval_rows, args)
    append_jsonl(output_dir / "eval" / "periodic.jsonl", {"step": step, **step_eval_metrics})

step_record = {"step": step, "train_mean_nll": final_loss, ...}
optim_metrics = getattr(optim_step_result, "metrics", None)
if isinstance(optim_metrics, dict):
    merge_optimizer_metrics_into_step_record(step_record, optim_metrics)
if step_eval_metrics is not None:
    merge_eval_metrics_into_step_record(step_record, step_eval_metrics)
train_metrics_enabled = args.train_metrics_every > 0
if train_metrics_enabled and should_record_train_metrics_row(
    step,
    total_steps,
    args.train_metrics_every,
    has_merged_eval=step_eval_metrics is not None,
):
    append_jsonl(output_dir / "train" / "metrics.jsonl", step_record)
```

Keep the cookbook-managed loop fields authoritative inside `step_record`.
If backend optimizer metrics carry colliding keys such as `step`, do not let
those values overwrite the outer training loop's logical step bookkeeping;
merge them through `merge_optimizer_metrics_into_step_record(...)` so
conflicts land under `optim/` names instead.

Current example: lawbench and fingpt both merge periodic eval into the step
record using the `test/` prefix, and both keep colliding optimizer metrics
namespaced under `optim/`.

## Section Placement

SFT experiments follow the `# ===== Section =====` layout in `naming.md`. This table shows where SFT functions land in the `Training helpers` section:

| Function | Role |
|----------|------|
| `create_training_client` | Create LoRA training client (shared with GRPO) |
| `load_training_state` | Restore weights only for `--load-checkpoint-path` |
| `load_training_state_with_optimizer` | Restore weights plus optimizer state for automatic same-run resume |
| `get_last_resumable_checkpoint` | Find the latest row in `train/checkpoints.jsonl` with a non-empty `state_path` |
| `step_to_loop_position` | Map completed optim step to next `(epoch, batch)` |
| `validate_resume_contract` | Reject mismatched run-defining args when resuming the same run |
| `resolve_resume_state` | Return `(start_step, start_epoch, start_batch)` from the matched checkpoint row |
| `build_supervised_datum` | Convert prepared prompt tokens + assistant text into a response-only-loss datum |
| `compute_lr_multiplier` | LR schedule: linear / cosine decay |
| `compute_total_train_steps` | Derive total steps from rows × epochs × batch_size |
| `compute_mean_nll` | Weighted NLL reduction from `forward_backward` result |
| `save_training_state` | Save resumable checkpoint (shared with GRPO) |
| `save_weights_for_sampling` | Export weights for eval sampler (shared with GRPO) |
| `save_sampler_checkpoint` | Export durable sampler path for offline eval |
| `should_record_train_metrics_row` | Cadence gate for `train/metrics.jsonl` append |
| `should_print_train_step` | Cadence gate for stdout step printing |
| `reset_supervised_append_streams` | Truncate run-scoped JSONL on fresh starts; preserve on same-run resume |
| `merge_eval_metrics_into_step_record` | Fold periodic eval scalars into train step dict |
| `run_train` | SFT training loop — **last in section**, calls everything above |

**Key ordering rule**: within `Training helpers`, place callees above callers. `run_train` comes last because it calls `build_supervised_datum`, `compute_lr_multiplier`, `save_*`, etc.

Other sections follow the same layout as eval-only experiments (see `naming.md`): benchmark-specific graders and prompt builders go in `Task-specific helpers`, the standard 5 adapters go in `Task-specific adapters`, and `write_outputs` goes in `Artifact writing`.

## Resume Support

Automatic same-run resume and fresh-run checkpoint loading
(`--load-checkpoint-path`) are research-grade SFT features. When you add them,
follow the checkpoint-row contract above and keep `train/checkpoints.jsonl`
as the restore source of truth:

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

See `skills/new-experiment-plus/references/upgrade_playbook.md` for
longer operability guidance.

## Current Repo Examples

- `experiments/lawbench` - research-grade SFT logging with prompt lineage (`train/batches.jsonl`), git provenance, and full resume support
- `experiments/fingpt` - multi-benchmark SFT with merged periodic-eval rows, cadence flags, `train/batches.jsonl` prompt lineage, and git provenance

## README Result Reporting

When an SFT experiment `README.md` reports measured runs, keep the result section compact but complete:

- start `Current results` with `Status: \`placeholder\`` until a checked run exists, and switch to `Status: \`checked\`` only when the reported run is actually checked
- record train config with at least `num_epochs`, `batch_size`, `learning_rate`, and `rank`
- record eval config with the relevant `max_concurrent_requests`
- report wall-clock timing, not a mixed timing convention
- if eval timing is batch wall-clock or depends on parallel vs sequential execution, say so explicitly
- treat throughput as supplementary, not as a replacement for wall-clock timing

## Current Shared SFT Core

As of the current repo alignment pass, these SFT helpers are treated as shared
reusable code fragments and should stay semantically identical when they appear
under the same name:

- `create_training_client`
- `load_training_state`
- `load_training_state_with_optimizer`
- `save_training_state`
- `save_sampler_checkpoint`
- `save_weights_for_sampling`
- `create_sampling_client`
- `resolve_api_result`
- `resolve_api_result_async`
- `extract_api_path`
- `build_supervised_datum`
- `compute_total_train_steps`
- `should_record_train_metrics_row`
- `reset_supervised_append_streams`
- `get_last_resumable_checkpoint`
- `step_to_loop_position`
- `validate_resume_contract`
- `resolve_resume_state`
- `append_jsonl`
- `load_jsonl`
- `scalar_metric_items`
- `optional_git_provenance`
- `build_generation_prompt_tokens`
- `extract_row_id`
- `get_tokenizer`
- `cached_tokenizer_dir`
- `is_sampler_model_path`
- `write_json`
- `write_jsonl`
- `write_outputs`

For `build_supervised_datum`, the shared contract is now the prompt-token-level
core:

```python
build_supervised_datum(tokenizer, prompt_tokens, assistant_text)
```

Benchmark-specific row/message rendering should stay outside that helper in a
local wrapper such as `build_supervised_row_datum(...)`.

## Intentional SFT Variants

These helpers may still differ across experiments even after a healthy
alignment pass, because they carry benchmark-specific training/eval behavior:

- `build_supervised_row_datum`
- `build_supervised_batch_trace_record`
- `build_eval_messages`
- `grade_assistant_text`
- `compute_eval_metrics`
- `normalize_train_rows`
- `normalize_eval_rows`
- `run_dry_run`
- `run_eval`
- `run_train`
- `main_async`
- `parse_args`

Additional runtime/environment helpers may also remain intentionally different
when an experiment needs deployment-specific behavior:

- `create_service_client`
- `prepare_run_dir`
- `write_run_metadata`
- `cached_tokenizer_dir`
- `get_tokenizer`
- `extract_row_id`
- `build_generation_prompt_tokens`

Decision rule:

- same name + shared infrastructure meaning -> keep implementations aligned
- benchmark-specific row rendering / scoring / loop orchestration -> keep local
- if behavior must diverge, move the local behavior behind a differently named
  wrapper instead of changing the shared helper's meaning
