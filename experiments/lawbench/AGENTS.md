# lawbench agent notes

## Read order

1. `PROMPT.md`, when present
2. `README.md`
3. `data/README.md`
4. `train.py`

## Experiment-local truth

- The benchmark anchor is the full official LawBench manifest at `data/eval/full.jsonl`; smoke files and `train_eval_200.jsonl` are validation aids, not final benchmark numbers.
- Keep `--eval-data` explicit: use `data/eval/smoke.jsonl` for local `--dry-run` validation and `data/eval/full.jsonl` for held-out eval or training final eval.
- `--train-data` defaults to `data/train/full.jsonl`; point it somewhere else only when you are intentionally validating a different local train manifest.
- `autoresearch.sh` is the canonical automation wrapper for the maintained SFT line and should use a fresh timestamped run directory under `artifacts/runs/`.
- Same-run resume is directory-driven through the latest resumable `state_path` row in `train/checkpoints.jsonl`; `--load-checkpoint-path` starts a fresh weight-only run.
- Use `--eval-only --eval-data data/eval/full.jsonl --base-model <sampler_path>` for clean checkpoint confirmation reruns.
- The official scorer lives under `third_party/lawbench_official/evaluation/`; keep task-id-first eval ordering and official aggregation intact.
- The scorer can emit temporary `tmp_*.para` and `tmp_*.para.m2` files under `third_party/.../utils/`; they are local scratch and should stay ignored by git and your sync tool.

## Canonical commands

```bash
cd experiments/lawbench
uv sync
uv run train.py --dry-run --eval-data data/eval/smoke.jsonl
uv run train.py --eval-only --eval-data data/eval/full.jsonl --base-model Qwen/Qwen3-4B-Instruct-2507
uv run python -m unittest tests.test_train
bash autoresearch.sh
```

## Sharp edges

- Keep `data/train/full.jsonl` and `data/eval/full.jsonl` as separate local pipelines with explicit overlap checks; do not merge their provenance.
- `data/train/full.jsonl` intentionally preserves source-level duplicates from `DISC-Law-SFT`; do not treat repeated prompts as a resume bug unless the run bookkeeping proves it.
- The official scorer writes scratch files during eval, so sync tooling must ignore them or they will churn the working tree.
