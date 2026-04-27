# dapo-aime24

This directory is a self-contained MinT experiment that trains direct GRPO on a local materialization of `BytedTsinghua-SIA/DAPO-Math-17k` and evaluates on the full frozen AIME 2024 set under one fixed local contract.
It does not do an SFT warm start.

Current runnable scope:

- benchmark: full frozen AIME 2024 from `data/eval.jsonl`
- base model: `Qwen/Qwen3-4B-Instruct-2507`
- training route: direct GRPO on `data/train.jsonl`
- primary metrics: `METRIC eval_accuracy=...`, `METRIC eval_greedy_accuracy=...`, `METRIC eval_pass_at_k=...`
- reportable data: `data/eval.jsonl`

## Quickstart

The fastest way to reproduce the current repo-local workflow is:

1. sync the environment and source the shared cache helper
2. run `--dry-run` for credential-free local validation
3. rerun the frozen eval-only benchmark
4. run the live smoke suite or the canonical GRPO wrapper
5. pick a saved `sampler_path` from `train/checkpoints.jsonl` and rerun `--eval-only`

Set up the environment and local credentials:

```bash
cd experiments/dapo-aime24
source ./cache-env.sh
uv sync
# export MINT_API_KEY (and optional MINT_BASE_URL) in the shell or put them in .env
```

Validate local data and prompt wiring without remote calls:

```bash
uv run train.py --dry-run
```

Run the frozen eval-only benchmark:

```bash
uv run train.py --eval-only \
  --base-model Qwen/Qwen3-4B-Instruct-2507 \
  --eval-data data/eval.jsonl
```

Run the canonical long GRPO line:

```bash
bash autoresearch.sh --save-state-name dapo-aime24
```

Evaluate a saved checkpoint export:

```bash
uv run train.py --eval-only \
  --base-model '<sampler_path>' \
  --eval-data data/eval.jsonl
```

For each reportable run, keep the evidence bundle together: `run.json`, `console.log`, `eval/metrics.json`, `eval/predictions.jsonl`, and `train/checkpoints.jsonl` when checkpoints are produced.

### Run modes and restore

- `uv run train.py --dry-run` validates data shape and prompt construction without remote calls.
- `uv run train.py --eval-only --base-model <hf_id_or_sampler_path>` is the frozen benchmark confirmation path.
- `bash autoresearch.sh` is the canonical automation wrapper for the current practical GRPO line.
- same-run resume is directory-driven: rerun the same training command with the same `--log-path` and `train.py` restores from the latest resumable `state_path` row in `train/checkpoints.jsonl`.
- `--load-checkpoint-path` is the fresh weight-only start path; it is ignored when the current `--log-path` already has a resumable `state_path`.

### Artifact pointers

In the chosen `--log-path` (default `artifacts/latest`), the most useful files are:

- `run.json`, `console.log`
- `eval/metrics.json`, `eval/predictions.jsonl`, `eval/metrics.jsonl`
- `train/checkpoints.jsonl`, `train/metrics.jsonl`
- `train/rollouts.jsonl`, `train/failures.jsonl`, `train/failures.log`

## Live smoke tests

Default train-flow validation uses the real MinT backend:

```bash
uv run python -m unittest tests.test_train tests.test_async_compat tests.test_rl_logprobs
```

This live validation covers the main user-facing happy paths on minimal local data:

- `--eval-only`
- smoke train
- interrupted same-run resume by rerunning the same command in the same `--log-path`
- fresh `--eval-only --base-model <sampler_path>` from a saved sampler checkpoint
- async compatibility and RL logprobs support checks

## Data

- train: `data/train.jsonl` (materialized locally from `BytedTsinghua-SIA/DAPO-Math-17k`)
- eval: `data/eval.jsonl` (all 30 AIME 2024 problems)
- provenance: `data/sources.yaml` plus local raw snapshots `data/raw_aime_2024.jsonl` and `data/raw_dapo_math_17k.parquet`

Train row contract:

```json
{"id": "...", "question": "...", "answer": "34", "source": "BytedTsinghua-SIA/DAPO-Math-17k"}
```

Eval row contract:

```json
{"ID": "2024-I-1", "Problem": "...", "Answer": 204}
```

### Benchmark contract

- prompt: one `user` message rendered with `tokenizer.apply_chat_template(..., add_generation_prompt=True)`
- answer parsing: grade the last parseable `Answer: ...` line
- grading: normalized exact match

Primary metrics:

```text
METRIC eval_accuracy=...
METRIC eval_greedy_accuracy=...
METRIC eval_pass_at_k=...
```

## References

- Upstream model and method line: `BytedTsinghua-SIA/DAPO`
- Upstream train corpus: `BytedTsinghua-SIA/DAPO-Math-17k`
- Local provenance: `data/sources.yaml`
