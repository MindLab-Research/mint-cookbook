# Autoresearch: lawbench

`autoresearch.sh` is the canonical automation wrapper for this experiment's current runnable line.
It is not automatically the final report or confirmation command.
This file is the wrapper protocol for search work. Keep environment setup, data preparation, and full reproduction commands in `README.md`.

## Objective

Improve `eval_lawbench_avg` on LawBench without redefining the benchmark contract.

## Current wrapper

- base model: `Qwen/Qwen3-4B-Instruct-2507`
- train data: `data/train/full.jsonl`
- periodic train-time eval: `data/eval/train_eval_200.jsonl`
- final benchmark eval: `data/eval/full.jsonl`
- train config: `num_epochs=1`, `rank=16`, `batch_size=256`, `learning_rate=1e-4`, `lr_schedule=cosine`
- eval config: `eval_max_tokens=1024`, `max_concurrent_requests=128`, `mint_timeout=600`
- cadence: `eval_every=10`, `save_every=10`, `train_metrics_every=1`, `train_print_every=1`
- log path: `artifacts/runs/sft-1epoch-qwen3-4b-<timestamp>/`

Run it with:

```bash
bash autoresearch.sh
```

Intentional variants should still go through CLI overrides to this wrapper, for example:

```bash
bash autoresearch.sh --learning-rate 5e-5
bash autoresearch.sh --num-epochs 1 --batch-size 16
bash autoresearch.sh --log-path artifacts/manual-runs/lawbench-ablation
```

## Search signals

- Primary search signal: `eval_lawbench_avg` from the wrapper run
- Secondary diagnostics: `eval_mem_avg`, `eval_understanding_avg`, `eval_application_avg`, `eval_abstention_rate`, task-level metrics, and wall-clock timing recorded for that run
- Train-side continuation signals: `train_mean_nll` in `train/metrics.jsonl` plus periodic rows in `eval/periodic.jsonl`

`data/eval/train_eval_200.jsonl` is only the maintained periodic train-time proxy slice inside the wrapper run. The wrapper's final score still comes from `data/eval/full.jsonl`.

## Mutable surface

- `train.py` implementation details that preserve the benchmark contract
- prompt formatting, parsing, and data normalization that keep the reported benchmark meaning stable
- wrapper defaults and training recipe knobs that are already intentionally documented here
- side-specific data materialization helpers under `data/eval/` and `data/train/` when provenance and split rules stay explicit
- the concrete files that normally move together are `train.py`, `data/eval/`, `data/train/`, `data/sources.yaml`, `README.md`, and `autoresearch.sh`

## Frozen contract

- the meaning of `eval_lawbench_avg`
- the official LawBench scorer contract, aggregation rule, and reportable full eval manifest without an explicit contract change
- the task-id-first eval order
- the role of `uv run train.py --eval-only --eval-data data/eval/full.jsonl` as the benchmark confirmation entrypoint
- the semantic split between automatic same-run resume in the current run directory and fresh-run `--load-checkpoint-path`

## Recovery and confirmation

- Checkpoint cadence: the wrapper saves periodically with `save_every=10`, so `train/checkpoints.jsonl` is part of the default search path.
- Same-run resume: rerunning the same training command in the same `--log-path` continues the interrupted candidate from the latest resumable `state_path` in `--log-path/train/checkpoints.jsonl`.
- Fresh restart: `--load-checkpoint-path <state_path>` is for fresh-run weight loading only and should be used intentionally, not as the default search strategy.
- Clean confirmation: `state_path` stays the resumable restore handle and `sampler_path` stays the later eval-only handle. For a clean confirmation rerun or later comparison, read the latest `sampler_path` from `train/checkpoints.jsonl` and rerun:

```bash
uv run train.py --eval-only \
  --eval-data data/eval/full.jsonl \
  --base-model '<sampler_path>'
```

## Run budget and stopping

- do not stop a candidate only because early periodic eval is flat; the periodic proxy slice is sparse and can lag behind train-side improvement
- use `train_mean_nll` as the main continuation signal during the early and middle part of the run
- treat `eval/periodic.jsonl` as a slower confirmation signal, not as a requirement that every periodic eval must immediately improve
- keep running while train loss is still making meaningful downward progress and the run is still within the current wall-clock or step budget
- stop when train loss has clearly flattened for multiple windows, when periodic eval shows repeated degradation, or when the current run budget is exhausted
- promote promising checkpoints to clean `--eval-only` confirmation reruns instead of extending one candidate indefinitely

Keep `autoresearch.sh`, `autoresearch.md`, and `README.md` aligned whenever the wrapper recipe changes.
