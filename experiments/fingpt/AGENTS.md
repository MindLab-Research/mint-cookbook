# fingpt agent notes

## Read order

1. `PROMPT.md`, when present
2. `README.md`
3. `data/README.md`
4. `train.py`

## Experiment-local truth

- The current M0 benchmark anchor is the official `FinGPT/fingpt-fineval` Hugging Face dataset mirrored locally as `data/fingpt-fineval/train.jsonl` and `data/fingpt-fineval/test.jsonl`.
- `autoresearch.sh` is intentionally a separate sentiment wrapper line; do not treat it as the Fineval benchmark anchor or the default Fineval reproduction command.
- Keep `--eval-data` explicit: use `smoke:data/smoke_eval.jsonl` for local `--dry-run` validation, `fineval:data/fingpt-fineval/test.jsonl` for Fineval eval, and the full held-out sentiment bundle for sentiment confirmation runs.
- `--train-data` defaults to `data/fingpt-fineval/train.jsonl` only when `--task-type fineval`; sentiment training still needs an explicit task-specific train manifest.
- Treat any smoke fallback result as harness validation, not a benchmark number worth reporting.
- The current grader assumes multiple-choice outputs where the gold answer is either `A`/`B`/`C`/`D` or a full option string like `C. 1730万元`.
- Use the intended eval-only path with `--base-model <sampler_path>` for clean checkpoint confirmation reruns: Fineval uses `--task-type fineval --eval-data fineval:data/fingpt-fineval/test.jsonl`, while sentiment uses the full held-out bundle in `--eval-data`.
- Keep the runtime single-file; do not import helpers from sibling experiments.
- The current SFT path is intentionally narrow: train only on `data/fingpt-fineval/train.jsonl`, evaluate only on `data/fingpt-fineval/test.jsonl`, and interpret gains only as Fineval-slice SFT.

## Canonical commands

```bash
cd experiments/fingpt
uv sync
uv run train.py --dry-run --task-type fineval --eval-data smoke:data/smoke_eval.jsonl
uv run python data/download_fingpt_fineval.py
uv run python data/download_fingpt_sentiment.py
uv run python data/download_fingpt_sentiment_benchmarks.py
uv run python data/make_sentiment_train_eval_subset.py
uv run train.py --eval-only --task-type fineval --base-model Qwen/Qwen3-4B-Instruct-2507 --eval-data fineval:data/fingpt-fineval/test.jsonl
uv run train.py --task-type fineval --train-data data/fingpt-fineval/train.jsonl --eval-data fineval:data/fingpt-fineval/test.jsonl --base-model Qwen/Qwen3-4B-Instruct-2507 --num-epochs 1 --batch-size 8 --learning-rate 1e-4 --rank 16
bash autoresearch.sh
```

## Sharp edges

- The target paper is broader than the current M0 slice; do not rewrite the README to imply the full multi-task reproduction already exists.
- If you expand from Fineval to the wider FinGPT benchmark suite, document the new task set, metrics, and local artifact layout in `README.md`, `PROMPT.md`, and `data/sources.yaml` together.
- Keep `run.json` honest about whether the run used `official_fingpt_fineval` or `smoke_fallback`.
- Do not describe the current SFT path as a full FinGPT-family training reproduction; it is only Fineval-slice LoRA SFT.
