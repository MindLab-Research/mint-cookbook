# dapo-aime

This directory is a self-contained MinT experiment that trains direct GRPO on a local materialization of `BytedTsinghua-SIA/DAPO-Math-17k`, keeps AIME 2024 as the fixed reportable benchmark, and also ships local AIME 2025 and AIME 2026 eval manifests under the same row contract.
It does not do an SFT warm start.

Current runnable scope:

- benchmark anchor: AIME 2024 from `data/eval/aime2024.jsonl`
- auxiliary eval manifests: `data/eval/aime2025.jsonl`, `data/eval/aime2026.jsonl`
- base model: `Qwen/Qwen3-4B-Instruct-2507`
- training route: direct GRPO on `data/train/full.jsonl`
- primary metrics: `METRIC eval_accuracy=...`, `METRIC eval_greedy_accuracy=...`, `METRIC eval_pass_at_k=...`
- multi-eval metrics: when `--eval-data` names multiple manifests, the first dataset remains the unprefixed primary metric stream and additional datasets emit prefixed metrics such as `METRIC aime2025_eval_accuracy=...`
- reportable benchmark data: `data/eval/aime2024.jsonl`

## Quickstart

The fastest way to reproduce the current repo-local workflow is:

1. sync the environment and source the shared cache helper
2. use the checked-in eval-side split files for explicit smoke validation and AIME 2024 benchmark confirmation, then materialize `data/train/full.jsonl` before training
3. run `--dry-run --eval-data data/eval/smoke.jsonl` for credential-free local validation, or pass all three year manifests to validate the multi-eval path
4. run a bounded live eval-only smoke check before the slower AIME 2024 benchmark rerun, and only then decide whether you need the full benchmark or the combined multi-year final eval
5. run the live smoke suite or the canonical GRPO wrapper
6. pick a saved `sampler_path` from `train/checkpoints.jsonl` and rerun `--eval-only --eval-data data/eval/aime2024.jsonl --base-model <sampler_path>`

Set up the environment and local credentials:

```bash
cd experiments/dapo-aime
source ./cache-env.sh
uv sync
cp ../../.env.example .env  # if needed
# then fill in MINT_API_KEY and optional MINT_BASE_URL in .env, or export them in the shell
```

If `data/train/full.jsonl` is missing, materialize the split local artifacts first:

```bash
python data/download_and_prepare.py
```

Validate local data and prompt wiring without remote calls:

```bash
uv run train.py --dry-run \
  --eval-data data/eval/smoke.jsonl
```

Validate the multi-eval path and inspect the fixed periodic-eval mix:

```bash
uv run train.py --dry-run \
  --eval-data data/eval/aime2024.jsonl,data/eval/aime2025.jsonl,data/eval/aime2026.jsonl
```

For the cheapest real live eval-only confirmation, run the dedicated live smoke test first:

```bash
uv run python -m unittest tests.test_train.LiveDAPOAIMEFlowTest.test_eval_only_live_smoke
```

Run the frozen eval-only benchmark:

```bash
uv run train.py --eval-only \
  --eval-data data/eval/aime2024.jsonl \
  --base-model Qwen/Qwen3-4B-Instruct-2507
```

Run one final eval sweep that writes all three year manifests in one run:

```bash
uv run train.py --eval-only \
  --eval-data data/eval/aime2024.jsonl,data/eval/aime2025.jsonl,data/eval/aime2026.jsonl \
  --base-model Qwen/Qwen3-4B-Instruct-2507
```

Run the canonical long GRPO line:

```bash
bash autoresearch.sh
```

If you want an extra named final export appended to `train/checkpoints.jsonl` on top of the per-step checkpoints, pass `--save-state-name <label>`.

Evaluate a saved checkpoint export:

```bash
uv run train.py --eval-only \
  --eval-data data/eval/aime2024.jsonl \
  --base-model '<sampler_path>'
```

For each reportable run, keep the evidence bundle together: `run.json`, `console.log`, `eval/metrics.json`, the per-dataset final eval directories under `eval/` when multi-eval is enabled, and `train/checkpoints.jsonl` when checkpoints are produced.

### Run modes and restore

- `uv run train.py --dry-run --eval-data data/eval/smoke.jsonl` validates data shape and prompt construction without remote calls.
- `uv run train.py --eval-only --eval-data data/eval/aime2024.jsonl --base-model <hf_id_or_sampler_path>` is the frozen benchmark confirmation path.
- `--eval-data` also accepts comma-separated `name:path` or plain-path entries. Final eval writes one full snapshot per named dataset under `eval/<name>/...`; training-time periodic eval uses one stable random third from each named dataset to form a single mixed eval set.
- `bash autoresearch.sh` is the canonical automation wrapper for the current practical GRPO line.
- same-run resume is directory-driven: rerun the same training command with the same `--log-path` and `train.py` restores from the latest resumable `state_path` row in `train/checkpoints.jsonl`.
- `--load-checkpoint-path` is the fresh weight-only start path; it is ignored when the current `--log-path` already has a resumable `state_path`.

In the chosen `--log-path` (default `artifacts/latest`), the most useful files are:

- `run.json`, `console.log`
- `eval/metrics.json`, `eval/predictions.jsonl`, `eval/metrics.jsonl`
- `eval/<dataset_name>/metrics.json`, `eval/<dataset_name>/predictions.jsonl` when `--eval-data` names multiple eval manifests
- `train/checkpoints.jsonl`, `train/metrics.jsonl`
- `train/rollouts.jsonl`, `train/failures.jsonl`, `train/failures.log`

## Live smoke tests

Default train-flow validation uses the real MinT backend:

```bash
uv run python -m unittest tests.test_train
```

This live validation covers the main user-facing happy paths on minimal local data:

- `--eval-only`
- smoke train
- interrupted same-run resume by rerunning the same command in the same `--log-path`
- fresh `--eval-only --eval-data data/eval/aime2024.jsonl --base-model <sampler_path>` from a saved sampler checkpoint

If you only need the bounded remote eval-only check, run the single test method instead of the whole suite:

```bash
uv run python -m unittest tests.test_train.LiveDAPOAIMEFlowTest.test_eval_only_live_smoke
```

## Data

The local dataset layout is split by train side and eval side:

- default train: `data/train/full.jsonl` (materialized locally from `BytedTsinghua-SIA/DAPO-Math-17k`)
- train smoke: `data/train/smoke.jsonl` (checked in for tiny local validation)
- train raw snapshot: `data/train/raw_dapo_math_17k.parquet`
- default eval: `data/eval/aime2024.jsonl` (the frozen AIME 2024 benchmark file; checked in for explicit eval-only validation)
- named eval manifests: `data/eval/aime2024.jsonl`, `data/eval/aime2025.jsonl`, `data/eval/aime2026.jsonl` (AIME 2024 is the reportable benchmark; AIME 2025/2026 are additional eval manifests under the same schema)
- eval smoke: `data/eval/smoke.jsonl` (built from one row each from AIME 2024, 2025, and 2026 for explicit dry-run validation)
- eval raw snapshots: `data/eval/raw_aime_2024.jsonl`, `data/eval/raw_aime_2025.parquet`, `data/eval/raw_aime_2026.parquet`
- provenance: `data/sources.yaml`

`python data/download_and_prepare.py` maintains the raw snapshots plus the year-named eval manifests under `data/eval/`, and rebuilds `data/eval/smoke.jsonl` from one prefix row per AIME year.
When multiple eval manifests are passed to `--eval-data`, final eval runs every manifest in full while training-time periodic eval builds one fixed mixed set by sampling one third from each named manifest.

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

## Current results

Status: `placeholder`

No maintained reportable run is checked in yet for this experiment.

When you add one, report it against the frozen `data/eval/aime2024.jsonl` AIME 2024 benchmark and keep the evidence bundle together: train config, eval config, `eval_accuracy`, `eval_greedy_accuracy`, `eval_pass_at_k`, wall-clock timing, and the artifact directory path.

## References

- Upstream model and method line: `BytedTsinghua-SIA/DAPO`
- Upstream train corpus: `BytedTsinghua-SIA/DAPO-Math-17k`
- Local provenance: `data/sources.yaml`
