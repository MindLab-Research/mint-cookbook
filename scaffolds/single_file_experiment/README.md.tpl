# {{EXPERIMENT_NAME}}

Before committing a concrete experiment README, delete unused optional rows and blocks instead of leaving placeholders.

This directory is a self-contained {{RUNTIME_NAME}} experiment for {{TARGET_METHOD_OR_QUESTION}}.
It evaluates {{BENCHMARK_NAME}} under one fixed local benchmark contract.

Current runnable scope:

- benchmark: {{BENCHMARK_SCOPE}}
- base model: `{{BASE_MODEL}}`
- training route: {{TRAINING_ROUTE_OR_EVAL_ONLY}}
- primary metric: `METRIC {{PRIMARY_METRIC}}=...`
- reportable data: `{{REPORTABLE_EVAL_PATH}}`

This experiment does not claim {{OUT_OF_SCOPE_CLAIM}}.
If the local baseline differs from the paper, reference implementation, model backbone, split, or scorer, state that difference here before any commands.

## Quickstart

The fastest way to reproduce the current repo-local results is:

1. sync the environment
2. build or download the local data artifacts
3. run the smoke or dry-run validation path
4. rerun the frozen eval-only benchmark
5. launch the canonical train-and-eval wrapper, if present (or run the raw train command)
6. read the saved `sampler_path` / `state_path` from `run.json`, or from `train/checkpoints.jsonl` if periodic checkpointing was enabled
7. rerun the frozen eval-only benchmark against that checkpoint
8. inspect the run artifacts

Set up the environment and local credentials:

```bash
cd experiments/{{EXPERIMENT_NAME}}
uv sync
cp .env.example .env  # if needed
# fill in {{RUNTIME_ENV_KEYS}} in .env, or export them in the shell
```

Prepare local data:

```bash
{{DATA_PREP_COMMANDS}}
```

Validate local data and prompt or scorer wiring without remote credentials when possible:

```bash
uv run train.py --dry-run \
  --eval-data {{SMOKE_EVAL_PATH}}
```

Run the frozen eval-only benchmark:

```bash
uv run train.py --eval-only \
  --base-model {{BASE_MODEL}} \
  --eval-data {{FULL_EVAL_PATH}} \
  --eval-limit 0 \
  --eval-max-tokens {{EVAL_MAX_TOKENS}} \
  --max-concurrent-requests {{MAX_CONCURRENT_REQUESTS}} \
  --tinker-timeout {{TINKER_TIMEOUT}} \
  --seed {{SEED}} \
  --log-path artifacts/runs/eval-{{RUN_LABEL}}-$(date +%Y%m%d-%H%M%S)
```

Run training plus final eval, if training is enabled:

```bash
uv run train.py \
  --base-model {{BASE_MODEL}} \
  --train-data {{FULL_TRAIN_PATH}} \
  --eval-data {{FULL_EVAL_PATH}} \
  --log-path artifacts/runs/{{TRAIN_RUN_LABEL}}-$(date +%Y%m%d-%H%M%S) \
  {{TRAINING_FLAGS}}
```

The canonical scaffold keeps training-side append streams opt-in:
`--save-every` enables `train/checkpoints.jsonl`,
`--eval-every` enables `eval/periodic.jsonl`, and
`--train-metrics-every` enables streamed `train/metrics.jsonl` plus
`train/batches.jsonl`.

Run the canonical wrapper, if this experiment ships one:

```bash
bash autoresearch.sh
```

`bash autoresearch.sh` is the canonical automation wrapper for the current practical line. `uv run train.py --eval-only` remains the bare benchmark confirmation entrypoint. If the wrapper later becomes a train-and-eval recipe, document in both `README.md` and `autoresearch.md` which eval path is used for periodic train-time checks, which eval path is used for the final benchmark confirmation, and how a saved checkpoint should be rerun cleanly.

After a training run, inspect `run.json` for the saved `sampler_path` / `state_path`.
If checkpointing is enabled (for example `--save-every N`), `train/checkpoints.jsonl` also records periodic and final checkpoint rows.

Evaluate a saved sampler checkpoint, if training records a `sampler_path` in `run.json` or `train/checkpoints.jsonl`:

```bash
uv run train.py --eval-only \
  --base-model '<sampler_path>' \
  --eval-data {{FULL_EVAL_PATH}} \
  --tinker-timeout {{TINKER_TIMEOUT}} \
  --log-path artifacts/runs/eval-checkpoint-{{RUN_LABEL}}-$(date +%Y%m%d-%H%M%S)
```

For each reportable run, keep the evidence bundle together: `run.json`, `console.log`, `eval/metrics.json`, `eval/predictions.jsonl`, and `train/checkpoints.jsonl` when checkpoints are produced.

### Run modes and knobs

| Mode | Command | Purpose |
| --- | --- | --- |
| dry run | `uv run train.py --dry-run --eval-data {{SMOKE_EVAL_PATH}}` | validate data and prompt/scorer wiring |
| eval only | `uv run train.py --eval-only --eval-data {{FULL_EVAL_PATH}}` | frozen benchmark baseline or checkpoint eval |
| train and eval | `uv run train.py --train-data {{FULL_TRAIN_PATH}} --eval-data {{FULL_EVAL_PATH}}` | train, then run the same final eval path |
| wrapper | `bash autoresearch.sh` | canonical automation wrapper for the current practical line, if present |

Important knobs for interpreting reported runs:

| Flag | Reported value | Meaning |
| --- | --- | --- |
| `--base-model` | `{{BASE_MODEL}}` | model or checkpoint evaluated |
| `--max-concurrent-requests` | `{{MAX_CONCURRENT_REQUESTS}}` | eval fan-out, or rollout request cap when this experiment documents that usage |
| `--eval-max-tokens` | `{{EVAL_MAX_TOKENS}}` | completion cap during eval |
| `{{TRAIN_FLAG_1}}` | `{{TRAIN_FLAG_VALUE_1}}` | {{TRAIN_FLAG_MEANING_1}} |
| `{{TRAIN_FLAG_2}}` | `{{TRAIN_FLAG_VALUE_2}}` | {{TRAIN_FLAG_MEANING_2}} |
| `{{TRAIN_FLAG_3}}` | `{{TRAIN_FLAG_VALUE_3}}` | {{TRAIN_FLAG_MEANING_3}} |

### Checkpoints and resume

- `state_path`: the saved training-state export (runtime save name typically ends with `-state`). The generic scaffold records the final one in `run.json`; periodic checkpoint rows also carry it when checkpointing is enabled. {{STATE_PATH_MEANING}}
- `sampler_path`: the durable sampler export for later `--eval-only --base-model <sampler_path>` reruns (runtime save name typically ends with `-sampler`). The generic scaffold records the final one in `run.json`; periodic checkpoint rows also carry it when checkpointing is enabled. {{SAMPLER_PATH_MEANING}}
- same-run resume (automatic, directory-driven): rerun the same training command with the same run directory (`--log-path`). When that directory's `train/checkpoints.jsonl` already contains a resumable `state_path`, `train.py` restores optimizer plus loop state through a fresh LoRA training client and `load_state_with_optimizer(...)`, then continues the same append-only registries and `console.log`. No dedicated `--resume-from` flag is used.
- `--load-checkpoint-path`: fresh run from saved weights only. Does not reuse optimizer state or the previous run's append-only logs; it is ignored when the current run directory already has a resumable `state_path`. {{LOAD_CHECKPOINT_PATH_MEANING}}

## Live smoke tests

Run the live smoke suite against the real backend:

```bash
python -m unittest experiments.{{EXPERIMENT_NAME}}.tests.test_train  # (Finished in xxmin)
```

This suite should stay on the smallest meaningful local slice and cover the user-facing entrypoint families for this experiment, usually:

- `--eval-only`
- smoke train
- interrupted same-run resume by rerunning the same training command in the same `--log-path`

If this experiment needs an additional fresh-run restore path, document it with `--load-checkpoint-path`, not a dedicated `--resume-from` flag.

## Data

If this experiment does not yet ship a local data pipeline, leave this section present but keep unknown fields blank (or replace with `TODO`) until the first reportable run is recorded.

| Split | Path | Rows | Source | Build command | Reportable use |
| --- | --- | ---: | --- | --- | --- |
| eval smoke | `{{SMOKE_EVAL_PATH}}` | `{{SMOKE_EVAL_ROWS}}` | {{SMOKE_EVAL_SOURCE}} | `{{SMOKE_EVAL_BUILD_COMMAND}}` | validation only |
| eval full | `{{FULL_EVAL_PATH}}` | `{{FULL_EVAL_ROWS}}` | {{FULL_EVAL_SOURCE}} | `{{FULL_EVAL_BUILD_COMMAND}}` | final benchmark |
| train smoke | `{{SMOKE_TRAIN_PATH}}` | `{{SMOKE_TRAIN_ROWS}}` | {{SMOKE_TRAIN_SOURCE}} | `{{SMOKE_TRAIN_BUILD_COMMAND}}` | validation only |
| train full | `{{FULL_TRAIN_PATH}}` | `{{FULL_TRAIN_ROWS}}` | {{FULL_TRAIN_SOURCE}} | `{{FULL_TRAIN_BUILD_COMMAND}}` | training |
| train-time eval | `{{TRAIN_EVAL_PATH}}` | `{{TRAIN_EVAL_ROWS}}` | {{TRAIN_EVAL_SOURCE}} | `{{TRAIN_EVAL_BUILD_COMMAND}}` | periodic eval only |

Eval row contract:

```json
{{EVAL_ROW_EXAMPLE}}
```

Train row contract, if training is enabled:

```json
{{TRAIN_ROW_EXAMPLE}}
```

Provenance and split rules:

- eval and train artifacts are managed as separate local workflows unless the benchmark requires a native layout
- `data/sources.yaml` records upstream sources, raw snapshot paths, split rules, filtering, and deduplication
- train/eval overlap policy: {{OVERLAP_POLICY}}
- smoke or fallback rows are not reportable benchmark results
- benchmark-native layout exception, if any: {{DATA_LAYOUT_EXCEPTION}}

### Benchmark contract

Primary metric:

```text
METRIC {{PRIMARY_METRIC}}=...
```

Companion metrics:

- `METRIC {{COMPANION_METRIC_1}}=...`
- `METRIC {{COMPANION_METRIC_2}}=...`

Prompt, parser, grader, or scorer contract:

- prompt shape: {{PROMPT_CONTRACT}}
- parser: {{PARSER_CONTRACT}}
- scorer: {{SCORER_CONTRACT}}
- aggregation: {{AGGREGATION_CONTRACT}}
- eval ordering or batching constraint: {{EVAL_ORDERING_CONSTRAINT}}
- official scorer runtime requirement: {{SCORER_RUNTIME_REQUIREMENT}}

## Current results

If there are no checked runs yet, keep this section but leave it empty (or a single `TODO`) rather than inventing numbers.

Result-format rules:

- record eval config, primary metric, artifact path, and wall-clock timing for every eval-only baseline
- for SFT, record `num_epochs`, `batch_size`, `learning_rate`, and `rank`
- for GRPO or other RL, record the smallest set of algorithm knobs needed to interpret the run, such as steps, group size, groups per batch, learning rate, and rank
- for DPO or pairwise training, record `num_epochs`, `batch_size`, `learning_rate`, `dpo_beta`, and reference model when applicable
- when reporting eval timing, record `max_concurrent_requests` and say whether timing is per-run or batch wall-clock
- do not report smoke or fallback data as benchmark results

### {{RESULT_SECTION_NAME}}

Run context:

- data: `{{RESULT_EVAL_PATH}}` (`{{RESULT_EVAL_ROWS}}` rows)
- base model: `{{BASE_MODEL}}`
- train config: {{TRAIN_CONFIG_SUMMARY}}
- eval config: {{EVAL_CONFIG_SUMMARY}}

| Run | Model or checkpoint | {{PRIMARY_METRIC}} | Wall time | Artifacts |
| --- | --- | ---: | --- | --- |
| base eval | `{{BASE_MODEL}}` | `{{BASE_METRIC_VALUE}}` | {{BASE_EVAL_WALL_TIME}} | `{{BASE_ARTIFACT_PATH}}` |
| trained eval | `{{TRAINED_CHECKPOINT_REF}}` | `{{TRAINED_METRIC_VALUE}}` | {{TRAINED_EVAL_WALL_TIME}} | `{{TRAINED_ARTIFACT_PATH}}` |

Delta vs base: `{{METRIC_DELTA}}`.

If the experiment reports multiple benchmark slices or datasets, use one row per slice:

| Dataset | Official reference | Repo base | Trained checkpoint | Delta |
| --- | ---: | ---: | ---: | ---: |
| `{{DATASET_1}}` | `{{OFFICIAL_1}}` | `{{BASE_1}}` | `{{TRAINED_1}}` | `{{DELTA_1}}` |
| `{{DATASET_2}}` | `{{OFFICIAL_2}}` | `{{BASE_2}}` | `{{TRAINED_2}}` | `{{DELTA_2}}` |

Timing notes:

- eval-only wall time: {{EVAL_ONLY_WALL_TIME}}
- train-and-eval total wall time: {{TRAIN_EVAL_WALL_TIME}}
- mean train step wall time: {{MEAN_STEP_WALL_TIME}}
- throughput, if useful: {{THROUGHPUT_NOTE}}
- parallel vs sequential eval note: {{PARALLELISM_NOTE}}

Figures, if generated:

![{{FIGURE_ALT_TEXT}}]({{FIGURE_PATH}})

## References

- Requirement: `{{REQUIREMENT_DOC_PATH}}`
- Paper: `{{PAPER_URL}}`
- Benchmark or dataset: `{{BENCHMARK_OR_DATASET_URL}}`
- Official repository: `{{OFFICIAL_REPOSITORY_URL}}`
- Additional deviation notes, if the experiment needs them: {{DEVIATION_NOTE_1}} / {{DEVIATION_NOTE_2}}
