# LawBench data guide

This experiment keeps train-side and eval-side artifacts separate under `data/train/` and `data/eval/`.
The local benchmark contract is frozen on JSONL manifests even though the upstream LawBench release ships one raw JSON file per task.

## Eval side

Managed eval artifacts:

- `data/eval/raw/lawbench-official/` - upstream raw `<task_id>.json` snapshot
- `data/eval/full.jsonl` - canonical full benchmark manifest
- `data/eval/smoke.jsonl` - fixed 5-task smoke slice for local validation
- `data/eval/train_eval_200.jsonl` - periodic train-time proxy slice only

Materialize them with:

```bash
cd experiments/lawbench
uv sync
uv run --no-sync data/eval/download_eval_raw.py --output-dir data/eval/raw/lawbench-official
uv run --no-sync data/eval/build_eval_manifest.py --official-data-dir data/eval/raw/lawbench-official
uv run --no-sync data/eval/build_eval_manifest.py \
  --official-data-dir data/eval/raw/lawbench-official \
  --smoke \
  --output-eval data/eval/smoke.jsonl \
  --output-meta data/eval/smoke.meta.json
uv run --no-sync data/eval/build_train_eval_manifest.py \
  --input-eval data/eval/full.jsonl \
  --output-eval data/eval/train_eval_200.jsonl \
  --output-meta data/eval/train_eval_200.meta.json
```

Canonical explicit eval-data commands:

- `uv run train.py --dry-run --eval-data data/eval/smoke.jsonl`
- `uv run train.py --eval-only --eval-data data/eval/full.jsonl`
- `data/eval/train_eval_200.jsonl` stays opt-in through `--train-eval-data`; it does not replace the final benchmark eval

## Train side

Managed train artifacts:

- `data/train/raw/disc-law-sft/` - raw `DISC-Law-SFT` snapshot
- `data/train/full.jsonl` - canonical train artifact
- `data/train/smoke.jsonl` - checked-in smoke SFT slice

Materialize the full train artifact with:

```bash
cd experiments/lawbench
uv run --no-sync data/train/download_train_raw.py --output-dir data/train/raw/disc-law-sft
uv run --no-sync data/train/build_train_manifest.py \
  --source data/train/raw/disc-law-sft/DISC-Law-SFT-Pair.jsonl \
  --source data/train/raw/disc-law-sft/DISC-Law-SFT-Pair-QA-released.jsonl \
  --source data/train/raw/disc-law-sft/DISC-Law-SFT-Triplet-released.jsonl \
  --source data/train/raw/disc-law-sft/DISC-Law-SFT-Triplet-QA-released.jsonl \
  --output-train data/train/full.jsonl \
  --output-meta data/train/full.meta.json
```

Reportable benchmark numbers come only from `data/eval/full.jsonl`.
Smoke artifacts and `data/eval/train_eval_200.jsonl` are validation-only.

See `data/sources.yaml` for provenance, split rules, and raw snapshot details.
