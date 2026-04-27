# chat-dpo

This directory is a self-contained, eval-first DPO experiment for chat-quality preference pairs.
The benchmark is a held-out pairwise preference set rather than a generation benchmark with an external grader.

Current runnable scope:

- benchmark: held-out pairwise preference eval on `data/eval.jsonl`
- base model: `Qwen/Qwen3-4B-Instruct-2507`
- training route: local DPO on `data/train.jsonl`, then final held-out eval
- primary metric: `METRIC eval_pair_accuracy=...`
- reportable data: `data/eval.jsonl`

## Quickstart

The fastest way to reproduce the current repo-local workflow is:

1. sync the environment and MinT credentials
2. prepare local `data/train.jsonl` and `data/eval.jsonl`
3. run `--dry-run` to validate schema and overlap without credentials
4. run `--eval-only` on a base model
5. run training plus final eval, or the canonical wrapper
6. pick a saved `sampler_path` from `train/checkpoints.jsonl` and rerun `--eval-only`

Set up the environment and local credentials:

```bash
cd experiments/chat-dpo
uv sync
cp .env.example .env  # if needed
# then fill in MINT_API_KEY (and optional MINT_BASE_URL) in .env
```

Validate local data without MinT credentials:

```bash
uv run train.py --dry-run
```

Evaluate a base model:

```bash
uv run train.py --eval-only \
  --base-model Qwen/Qwen3-4B-Instruct-2507 \
  --eval-data data/eval.jsonl
```

Train plus final eval:

```bash
uv run train.py \
  --train-data data/train.jsonl \
  --eval-data data/eval.jsonl \
  --base-model Qwen/Qwen3-4B-Instruct-2507 \
  --batch-size 8 \
  --num-epochs 1 \
  --dpo-beta 0.1 \
  --learning-rate 1e-5
```

Run the canonical wrapper:

```bash
bash autoresearch.sh
```

After training, inspect `train/checkpoints.jsonl` to find a saved `sampler_path`, then rerun eval-only:

```bash
uv run train.py --eval-only \
  --base-model '<sampler_path>' \
  --eval-data data/eval.jsonl
```

For each reportable run, keep the evidence bundle together: `run.json`, `console.log`, `eval/metrics.json`, `eval/predictions.jsonl`, and `train/checkpoints.jsonl` when checkpoints are produced.

### Run modes and restore

- `uv run train.py --dry-run` validates local pair data and overlap without remote calls.
- `uv run train.py --eval-only` is the frozen benchmark confirmation path.
- `bash autoresearch.sh` is the canonical automation wrapper for the current practical DPO line.
- same-run resume is directory-driven: rerun the same training command with the same `--log-path` and `train.py` restores from the latest resumable `state_path` in `train/checkpoints.jsonl`.
- `--load-checkpoint-path` is the fresh weight-only start path when you want a new run directory instead of continuing an existing one.

## Live smoke tests

Default train-flow validation now uses the real MinT backend instead of mocked helpers:

```bash
uv run python -m unittest tests.test_train
```

This live suite covers the main user-facing happy paths on the tiny local pair files:

- `--eval-only`
- smoke train
- interrupted same-run automatic resume by rerunning the same `--log-path`
- fresh `--eval-only --base-model <sampler_path>` from a saved sampler checkpoint

Use `--load-checkpoint-path` as a manual smoke path when you want a fresh weight-only start from a recorded checkpoint.

## Data

Default local train and eval paths are expected to contain tiny scaffold or real preference rows so local contract validation works immediately. These JSONL files are gitignored in this repo, so create or sync them locally before running `--dry-run`, eval, or training. Replace scaffold rows with real preference data before treating any run as reportable.

- train: `data/train.jsonl`
- eval: `data/eval.jsonl`
- provenance and migration notes: `data/README.md` and `data/sources.yaml`

Eval row contract:

```json
{
  "pair_id": "pair-0001",
  "messages": [{"role": "user", "content": "..."}],
  "chosen": [{"role": "assistant", "content": "..."}],
  "rejected": [{"role": "assistant", "content": "..."}],
  "group_id": "prompt-42",
  "source": "dataset-name",
  "metadata": {}
}
```

The experiment also accepts legacy compatibility inputs so old `chat_dpo` data can migrate gradually:

- `prompt` / `chosen` / `rejected`
- `prompt_conversation` / `completion_A` / `completion_B` + `label`
- `comparison` + `label`

### Benchmark contract

Primary metric:

```text
METRIC eval_pair_accuracy=...
```

Companion eval metrics:

- `METRIC eval_margin=...`
- `METRIC eval_chosen_score=...`
- `METRIC eval_rejected_score=...`
- `METRIC eval_num_pairs=...`

Interpretation:

- `eval_pair_accuracy`: fraction of eval pairs where the model assigns a higher weighted completion score to `chosen` than `rejected`
- `eval_margin`: average `chosen_score - rejected_score`
- `eval_chosen_score` / `eval_rejected_score`: average weighted completion logprob sums

## References

- Local data guide: `data/README.md`
- Local provenance: `data/sources.yaml`
