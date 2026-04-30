# Autoresearch: {{EXPERIMENT_NAME}}

`autoresearch.sh` is the canonical automation wrapper for this experiment's current runnable line.
It is not automatically the final report or confirmation command.
This file is the wrapper protocol for search work. Keep environment setup, data preparation, and full reproduction commands in `README.md`.

## Objective

Improve `{{PRIMARY_METRIC}}` on `{{BENCHMARK_NAME}}` without redefining the benchmark contract.

## Current wrapper

The baseline scaffold starts eval-first. The default wrapper recipe is:

```bash
LOG_PATH="artifacts/runs/eval-only-$(date +%Y%m%d-%H%M%S)"

exec uv run train.py \
  --eval-only \
  --eval-data {{FULL_EVAL_PATH}} \
  --log-path "$LOG_PATH" \
  "$@" 2>&1
```

Run it with:

```bash
bash autoresearch.sh
```

Intentional variants should still go through CLI overrides to this wrapper.

If this experiment later promotes a train-and-eval recipe into the wrapper, replace this section with the concrete default train data, eval data, cadence, and log-path naming instead of leaving generic prose behind.
When you do that, list the concrete wrapper fields in one stable order: base model, train data, periodic train-time eval data if any, final benchmark eval data, train config, eval config, cadence, and log path.

## Search signals

- Primary search signal: `METRIC {{PRIMARY_METRIC}}=...` from the wrapper run
- Secondary diagnostics: the companion benchmark metrics, failure modes, and wall-clock timing recorded for that run
- Once training exists, add the train-side continuation signals that matter for this experiment, for example `train_mean_nll`, reward metrics, or periodic eval rows

## Mutable surface

- wrapper defaults and training recipe knobs once those defaults are intentionally documented here
- bounded `train.py` implementation details that preserve the benchmark contract
- prompt formatting, parsing, and data normalization that keep the reported benchmark meaning stable
- side-specific data materialization helpers under `data/eval/` and `data/train/`, or documented dataset-family helpers under `data/`, when provenance and split rules stay explicit
- the concrete files that normally move together are `train.py`, `data/`, `data/sources.yaml`, `README.md`, and `autoresearch.sh`

## Frozen contract

- the meaning of `{{PRIMARY_METRIC}}`
- the frozen benchmark scorer, aggregation rule, and reportable eval manifest without an explicit contract change
- the role of `uv run train.py --eval-only --eval-data {{FULL_EVAL_PATH}}` as the benchmark confirmation entrypoint
- the semantic split between automatic same-run resume (rerunning the same training command with the same run directory via `--log-path`) and fresh-run `--load-checkpoint-path` once training and checkpoints exist

## Recovery and confirmation

- Checkpoint cadence: in the eval-first phase there are no train checkpoints yet; once training exists, say explicitly whether checkpointing is off by default or part of the default wrapper path.
- Same-run resume: once training exists, rerunning the same training command in the same run directory (`--log-path`) should continue from the latest resumable `state_path`.
- Fresh restart: `--load-checkpoint-path <state_path>` starts a new run from saved weights only and does not replace same-run resume.
- Clean confirmation: `sampler_path` stays the later eval-only handle for `uv run train.py --eval-only --eval-data {{FULL_EVAL_PATH}} --base-model <sampler_path>` reruns, not for same-run resume.
- When this experiment adopts the repo checkpoint contract, document in this file and in `README.md` which path is used for periodic train-time eval, which path is used for the final benchmark confirmation, and how to rerun a saved checkpoint cleanly. Keep the wrapper recipe, this file, and `README.md` aligned.

## Run budget and stopping

- In the eval-first phase, stop when the benchmark path is stable, the outputs are interpretable, and the remaining variance is understood
- Once training exists, use the train-side metric trend as the main continuation signal and treat periodic eval as a slower confirmation signal
- Promote promising checkpoints to clean `--eval-only` confirmation reruns instead of extending one candidate indefinitely

Keep `autoresearch.sh`, `autoresearch.md`, and `README.md` aligned whenever the wrapper recipe changes.
