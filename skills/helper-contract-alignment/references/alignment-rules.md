# Alignment Rules

Use this file when running a repo-wide helper-alignment pass.

## Classification

### 1. Shared infra/helper drift

These helpers should usually share one contract across live experiments and the
single-file scaffold template:

- `create_training_client`
- `create_sampling_client`
- `resolve_api_result`
- `resolve_api_result_async`
- `extract_api_path`
- `save_training_state`
- `save_sampler_checkpoint`
- `save_weights_for_sampling`
- `append_jsonl`
- `load_jsonl`
- `scalar_metric_items`
- `optional_git_provenance`

If the same helper name appears in multiple experiments and belongs to this
group, default to aligning it unless the user explicitly says otherwise.

### 2. Internal contract drift

These may be shared inside one profile family, but often need judgment before
forcing strict identity:

- `build_supervised_datum`
- `compute_total_train_steps`
- `should_record_train_metrics_row`
- `run_train`
- GRPO wrappers such as `save_recovery_checkpoint`

Decision rule:

- If semantics are really shared, align the helper.
- If one experiment needs extra orchestration, keep the shared helper stable
  and add a wrapper with a different name.

Current repo decisions:

- `build_supervised_datum` is the shared core helper that accepts prepared
  `prompt_tokens` plus `assistant_text`.
- Benchmark-specific row/message rendering belongs in a differently named local
  wrapper such as `build_supervised_row_datum(...)`.
- `run_train` should stay on the canonical
  `run_train(training_client, tokenizer, train_rows, eval_rows, args, output_dir)`
  shape unless an experiment really needs an approved variant like fingpt's
  explicit periodic-eval dataset routing.

### 3. Task-specific helpers

These often share names but usually carry benchmark-local logic:

- `build_eval_messages`
- `grade_assistant_text`
- `compute_eval_metrics`
- `normalize_train_rows`
- `normalize_eval_rows`
- `write_outputs`

Do not force strict identity here unless the user explicitly wants to promote a
new shared contract.

## Known intentional variants

These names may still drift after a healthy alignment pass:

- `run_train` - fingpt keeps the documented periodic-eval dataset-routing
  variant; lawbench stays on the canonical SFT signature.
- `build_supervised_row_datum` - wrapper names are intentionally local because
  they own row/message rendering for each benchmark.
- `build_supervised_batch_trace_record` - batch-trace metadata may stay local
  when experiments log different benchmark-specific summary fields.
- `extract_row_id` - benchmark-local key aliases are allowed when a dataset
  really uses them (for example DAPO's uppercase `ID` field).
- `create_service_client` - runtime environment keys can differ between tinker
  and mint experiments.
- `parse_args` / `run_dry_run` / `run_eval` / `main_async` - entrypoint helpers
  stay benchmark-local unless the task is explicitly to promote a broader
  scaffold contract.
- `cached_tokenizer_dir` / `get_tokenizer` - runtime or cluster-specific
  tokenizer staging fallbacks may stay local.
- `prepare_run_dir` / `write_run_metadata` - run-directory layout and metadata
  fields can vary when an experiment intentionally owns a different artifact
  lifecycle.
- `build_generation_prompt_tokens` - stricter prompt-template requirements may
  stay local when an experiment depends on them.
- `preview_text` - formatting details may vary when one experiment needs
  placeholder rendering or ASCII-only output.

When scanning, prefer filtering these with `scan_helper_drift.py --ignore ...`
unless the task is explicitly to revisit that boundary.

## Owner map

When a shared helper contract changes, sync in this order:

1. live experiment code
2. `scaffolds/single_file_experiment/train.py.tpl`
3. `scaffolds/single_file_experiment/naming.md`
4. `scaffolds/profiles/*.md`
5. relevant skills and references
6. `AGENTS.md` / `experiments/README.md` only if a hard/shared policy changed

## Validation

Minimum validation for each alignment phase:

- AST parse touched `.py` files
- Parse `train.py.tpl` when possible
- Run targeted tests or a small behavior check
- Compare helper hashes across the intended files

## Scope control

Prefer explicit include/exclude rules. For this repo, a common scope is:

- include `experiments/dapo-aime`, `experiments/fingpt`, `experiments/lawbench`
