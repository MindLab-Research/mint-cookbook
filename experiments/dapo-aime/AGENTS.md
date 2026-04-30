# dapo-aime agent notes

## Read order

1. `PROMPT.md`, when present
2. `README.md`
3. `data/README.md`
4. `train.py`

## Experiment-local truth

- The reportable benchmark anchor is the frozen AIME 2024 manifest at `data/eval/aime2024.jsonl`; keep that eval contract fixed unless you are repairing a broken measurement path.
- Keep `--eval-data` explicit: use `data/eval/smoke.jsonl` for local `--dry-run` validation, `data/eval/aime2024.jsonl` for held-out benchmark eval, and comma-separated specs such as `data/eval/aime2024.jsonl,data/eval/aime2025.jsonl,data/eval/aime2026.jsonl` when you intentionally want one run to produce multiple final eval snapshots.
- When `--eval-data` names multiple eval manifests, final eval runs every named dataset in full under `eval/<name>/...`, while training-time periodic eval uses one stable random third from each dataset to build a single mixed eval set.
- This experiment is direct GRPO only; do not silently reintroduce an SFT warm start into the canonical path.
- `cache-env.sh` is part of the practical runtime path and should be sourced before live runs when you need the shared cache setup.
- Same-run resume is directory-driven via the latest resumable `state_path` row in `train/checkpoints.jsonl`; `--load-checkpoint-path` is only for fresh weight-only starts.
- Use `--eval-only --eval-data data/eval/aime2024.jsonl --base-model <sampler_path>` for clean checkpoint confirmation reruns; do not invent a parallel eval resume flag.

## Canonical commands

```bash
cd experiments/dapo-aime
source ./cache-env.sh
uv sync
uv run train.py --dry-run --eval-data data/eval/smoke.jsonl
uv run train.py --dry-run --eval-data data/eval/aime2024.jsonl,data/eval/aime2025.jsonl,data/eval/aime2026.jsonl
uv run train.py --eval-only --eval-data data/eval/aime2024.jsonl --base-model Qwen/Qwen3-4B-Instruct-2507
uv run python -m unittest tests.test_train
bash autoresearch.sh
```

## Sharp edges

- `data/train/full.jsonl` is materialized locally from `DAPO-Math-17k`; if it is missing, rebuild it instead of swapping in ad-hoc train data.
- Keep the maintained split layout under `data/train/` and `data/eval/`; legacy root-level aliases like `data/train.jsonl` or `data/eval.jsonl` are compatibility leftovers and should not drive new docs, automation, or sync rules.
- AIME eval is noisy enough that checkpoint comparisons should look at `eval_accuracy`, `eval_greedy_accuracy`, and `eval_pass_at_k` together.
- Keep GRPO-specific rollout, checkpoint, and failure logging local to this experiment instead of abstracting it into sibling imports.
