# Eval Profile

This is the default startup profile for every new experiment.
The canonical template already includes a runnable single-turn chat eval path.

## What To Customize

Start by replacing these five task adapter functions:

1. `normalize_eval_rows(path)` - load benchmark eval data
2. `build_eval_messages(row)` - render each row into chat messages
3. `grade_assistant_text(assistant_text, row)` - extract and score the sampled assistant text
4. `compute_eval_metrics(predictions)` - aggregate row-level results into metrics
5. `normalize_train_rows(path)` - optionally load training data if it already exists

The template defaults handle the simplest case: JSONL plus exact-match grading. Override only what the benchmark actually needs.

Accepted variant: pairwise DPO eval may replace the generation-oriented `build_eval_messages(...)` / `grade_assistant_text(...)` path with pairwise scoring inside `run_eval(...)`. In that shape, keep `normalize_eval_rows(...)`, `normalize_train_rows(...)`, and `compute_eval_metrics(...)` explicit, and use `scaffolds/profiles/dpo.md` as the training reference.

## Startup Checklist

1. Copy the template into `experiments/<name>/train.py`.
2. Set up the eval-side data workflow under `data/eval/`: preserve a raw snapshot when local materialization is needed, keep one download or snapshot script, and keep one build or adjustment script.
3. Materialize `data/eval/smoke.*` first when a cheap local slice is practical. Standard split-layout experiments should pass that artifact explicitly via `--eval-data`; benchmark-native layouts may keep using named specs.
4. Run `uv run train.py --dry-run --eval-data <smoke_eval_path>` to verify data loading and prompt shape.
5. Keep adjusting adapter functions until the dry-run output looks correct.
6. Materialize the full eval artifact under `data/eval/full.*`, then run `uv run train.py --eval-only --eval-data <full_eval_path>` to establish the benchmark baseline.
7. Confirm the baseline eval artifacts are populated under `eval/` (`examples.jsonl`, `predictions.jsonl`, `metrics.json`), then record the baseline in `README.md` under `Current results`: start that section at `Status: \`placeholder\`` until a checked run exists, switch it to `Status: \`checked\`` only when the baseline is actually checked, and keep the eval config, primary metric, and wall-clock timing together.

## When To Add Training

Only add training after `--eval-only` produces a stable, meaningful baseline.
Then choose exactly one training extension:

- `scaffolds/profiles/sft.md` for supervised fine-tuning
- `scaffolds/profiles/dpo.md` for pairwise preference optimization
- `scaffolds/profiles/grpo.md` for RL optimization
