# DPO Profile

Add pairwise preference optimization after the eval baseline is stable.
This is a training extension layered on top of the eval profile, not a standalone starting point.

## Where This Profile Sits

- `scaffolds/README.md` owns the repo-wide split between templates, profiles, experiments, and skills.
- `scaffolds/single_file_experiment/naming.md` owns promoted helper names once a pattern is shared or intentionally promoted repo-wide.
- `skills/new-experiment` owns baseline bootstrap and points at the scaffold-owned artifact contract.
- `skills/new-experiment-plus` owns long-run operability guidance, not the canonical baseline artifact names.
- Optional `analysis_manifest.json` + stdout `ANALYSIS_MANIFEST path=...` is not part of the minimal profile sketch until promoted repo-wide.

## What To Add

The canonical template does **not** ship a generic DPO loop the way it now ships a generic SFT path.
Keep the shared directory-driven resume / checkpoint helper family from the SFT profile where the behavior is identical, but replace the row-level supervised loop with pairwise batching, a frozen reference model, and DPO loss.

Typical shape:

```python
# ---- DPO training ----
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

reference_model = args.reference_model.strip() or args.base_model
reference_client = await create_sampling_client(service_client, reference_model)
reference_tokenizer = (
    tokenizer
    if reference_model == args.base_model
    else get_tokenizer(reference_client, reference_model)
)

reset_dpo_append_streams(
    run_dir,
    resume_checkpoint=resume_checkpoint,
    include_periodic_eval=int(args.eval_every) > 0,
    include_checkpoints=bool(resume_checkpoint is not None or int(args.save_every) > 0),
    include_train_metrics=int(args.train_metrics_every) > 0,
    include_batch_trace=int(args.train_metrics_every) > 0,
)

train_metrics = await run_train(
    training_client,
    reference_client,
    tokenizer,
    reference_tokenizer,
    train_rows,
    eval_rows,
    args,
    run_dir,
    resume_checkpoint=resume_checkpoint,
)

state_name = build_state_save_name(args.base_model, run_dir)
final_state_name, final_sampler_name = checkpoint_save_names(state_name)
final_state_path = await save_training_state(training_client, final_state_name)
final_sampler_path = await save_sampler_checkpoint(training_client, final_sampler_name)

eval_client = (
    await create_sampling_client(service_client, final_sampler_path)
    if final_sampler_path
    else await save_weights_for_sampling(training_client)
)
eval_examples, predictions, eval_metrics = await run_eval(
    eval_client,
    tokenizer,
    eval_rows,
    args,
)

write_outputs(
    run_dir,
    args=args,
    eval_examples=eval_examples,
    predictions=predictions,
    metrics=eval_metrics,
    extra_payload={
        "reference_model": reference_model,
        "state_path": final_state_path,
        "sampler_path": final_sampler_path,
        "train_metrics": train_metrics,
    },
)
emit_metric_lines({**train_metrics, **eval_metrics})
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

The canonical scaffold exposes the run directory via `--log-path`; historical experiments may still call the same knob `--output-dir`. The auto-resume trigger rule is the same: rerun the same training command with the same run directory.

## DPO Concepts

| Term | Meaning | Current reference |
|------|---------|-------------------|
| `pair` | One prompt with a `chosen` and `rejected` continuation | `experiments/chat-dpo` preference rows |
| `reference_model` | Frozen scoring baseline used in the DPO loss | `--reference-model` (defaults to `--base-model`) |
| `margin` | Chosen score minus rejected score on held-out eval | `eval_pair_accuracy` companion metric |
| `batch_group_key` | Optional row key used to spread repeated prompts across batches | `chat-dpo` prompt-group batching |

## CLI Additions

Add DPO-specific flags on top of the eval profile and the shared SFT-style training cadence flags:

- `--reference-model` — optional frozen reference model; defaults to `--base-model`
- `--dpo-beta` — DPO temperature / preference sharpness
- `--max-length` — truncate full prompt+completion token spans before datum creation
- `--batch-group-key` — dotted key used to spread related preference pairs across batches
- `--allow-partial-batch` / `--drop-partial-batch` — whether to keep the last short batch

The current promoted DPO reference is `experiments/chat-dpo/train.py`.

## Functions To Wire Or Customize

Shared resume / save helpers follow the same meaning as the SFT profile:

- `get_last_resumable_checkpoint(run_dir)` — scan `train/checkpoints.jsonl` for the latest row with a non-empty `state_path`
- `validate_resume_contract(run_dir, args, *, resume_checkpoint)` — reject mismatched run-defining args before same-run resume continues
- `resolve_resume_state(run_dir, resume_checkpoint, *, total_steps, n_batches, num_epochs)` — derive `(start_step, start_epoch, start_batch)` from the matched checkpoint row
- `create_training_client(service_client, args, *, resume_state_path)` — restore optimizer state for same-run resume or weights only for `--load-checkpoint-path`
- `save_weights_for_sampling(training_client)`, `save_training_state(training_client, save_name)`, `save_sampler_checkpoint(training_client, save_name)` — same save/export split as SFT
- `build_state_save_name(base_model, run_dir)` + `checkpoint_save_names(base_name)` — keep one logical checkpoint name while using distinct runtime save names for state vs sampler exports

`reset_dpo_append_streams(run_dir, *, resume_checkpoint=None, include_periodic_eval=True, include_checkpoints=True, include_train_metrics=True, include_batch_trace=True)` — DPO equivalent of `reset_supervised_append_streams(...)`; on fresh runs, truncate/create the enabled run-scoped streams and remove disabled stream files left behind by an older run in the same directory:

```python
def reset_dpo_append_streams(
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
    for relative_path, enabled in (
        (Path("eval/periodic.jsonl"), include_periodic_eval),
        (Path("train/checkpoints.jsonl"), include_checkpoints),
        (Path("train/metrics.jsonl"), include_train_metrics),
        (Path("train/batches.jsonl"), include_batch_trace),
    ):
        path = run_dir / relative_path
        if enabled:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("", encoding="utf-8")
        elif path.exists():
            path.unlink()
```

`build_epoch_batches(rows, *, batch_size, allow_partial_batch, batch_group_key, seed)` — pairwise batch constructor. When `batch_group_key` is present, spread related prompt groups across different batches when possible so repeated prompts do not cluster into one step.

`build_preference_datum(tokenizer, messages, completion, *, max_length)` — isolate the completion span for one chosen or rejected continuation and build a response-only-loss datum with prompt tokens masked out.

`build_pair_payload(tokenizer, row, *, max_length)` — package the chosen and rejected datums plus full model inputs and token counts for one pair.

`score_reference_pair_payloads(ref_logprob_seqs, pair_payloads)` — collapse reference-model token logprobs into one chosen score and one rejected score per pair.

`compute_dpo_loss(chosen_logprobs, rejected_logprobs, chosen_ref_logprobs, rejected_ref_logprobs, dpo_beta)` — return `(loss, metrics)` where metrics usually include `dpo_loss`, `accuracy`, `margin`, `chosen_reward`, and `rejected_reward`.

`build_dpo_batch_trace_record(batch_rows, *, limit=DEFAULT_BATCH_PAIR_RECORD_LIMIT)` — optional bounded pair preview record for `train/batches.jsonl`; prefer prompt and chosen/rejected previews for the first few pairs rather than logging full batches.

## Held-Out Pairwise Eval

Unlike the template's default generated-text eval path, a DPO experiment may define eval as **chosen should score above rejected on the same prompt**. In that shape:

- `run_eval(...)` scores both continuations for each pair rather than sampling a fresh assistant message
- `compute_eval_metrics(predictions)` usually aggregates held-out pair metrics such as `eval_pair_accuracy`, `eval_margin`, `eval_chosen_score`, `eval_rejected_score`, and `eval_num_pairs`
- `write_outputs(...)` still writes the standard baseline eval snapshot under `eval/`

`experiments/chat-dpo` is the concrete reference for this pairwise-eval shape.

## README Result Reporting

When a DPO experiment `README.md` reports measured runs, keep the result section compact but complete:

- start `Current results` with `Status: \`placeholder\`` until a checked run exists, and switch to `Status: \`checked\`` only when the reported run is actually checked
- record train config with at least `num_epochs`, `batch_size`, `learning_rate`, `dpo_beta`, and reference model when applicable
- record eval config with the relevant `max_concurrent_requests`
- report wall-clock timing, not a mixed timing convention
- if eval timing is batch wall-clock or depends on parallel vs sequential execution, say so explicitly
- treat throughput as supplementary, not as a replacement for wall-clock timing
