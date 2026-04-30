# chat-dpo agent notes

## Read order

1. `PROMPT.md`, when present
2. `README.md`
3. `data/README.md`
4. `train.py`

## Experiment-local truth

- The benchmark anchor is held-out pairwise preference eval on `data/eval/full.jsonl`; do not treat placeholder smoke rows as reportable results.
- Keep `--eval-data` explicit: use `data/eval/smoke.jsonl` for local `--dry-run` validation and `data/eval/full.jsonl` for held-out eval or training final eval.
- `autoresearch.sh` is the canonical automation wrapper for the current DPO line, but it intentionally leaves `--save-every 0` and `--eval-every 0`; enable those explicitly when you want checkpoints or periodic eval inside search runs.
- Same-run resume is directory-driven only when checkpoint rows exist in `train/checkpoints.jsonl`; rerun the same training command with the same `--log-path`.
- `--load-checkpoint-path` is the fresh weight-only start path for a new run directory, not the default same-run resume path.
- Use `--eval-only --eval-data data/eval/full.jsonl --base-model <sampler_path>` for clean checkpoint confirmation reruns.
- Keep the runtime single-file and keep pairwise scoring explicit inside this experiment; do not import helpers from sibling experiments.

## Canonical commands

```bash
cd experiments/chat-dpo
uv sync
uv run train.py --dry-run --eval-data data/eval/smoke.jsonl
uv run train.py --eval-only --eval-data data/eval/full.jsonl --base-model Qwen/Qwen3-4B-Instruct-2507
uv run python -m unittest tests.test_train
bash autoresearch.sh
```

## Sharp edges

- The checked-in `data/train/full.jsonl` and `data/eval/full.jsonl` are tiny scaffold placeholders; replace them with local real pair data before treating runs as reportable.
- Keep the split data layout literal: do not add root-level `data/*.jsonl` aliases back into this experiment.
- Keep the held-out pairwise scoring meaning fixed: `chosen` must outscore `rejected` on the same prompt.
- Do not rewrite the experiment into a generation-style benchmark with an external grader; this directory is intentionally pairwise-first.
