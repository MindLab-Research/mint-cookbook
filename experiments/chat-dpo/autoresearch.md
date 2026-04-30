# Autoresearch: chat-dpo

`autoresearch.sh` is the canonical automation wrapper for this experiment's current runnable line.
It is not automatically the final report or confirmation command.
This file is the wrapper protocol for search work. Keep environment setup, data preparation, and full reproduction commands in `README.md`.

## Objective

Improve held-out pairwise preference results on `data/eval/full.jsonl`, not redefine the benchmark.

## Current wrapper

- base model: `Qwen/Qwen3-4B-Instruct-2507`
- train data: `data/train/full.jsonl`
- final eval data: `data/eval/full.jsonl`
- train config: `rank=16`, `learning_rate=1e-5`, `dpo_beta=0.1`, `batch_size=8`, `num_epochs=1`, `max_steps=0`
- cadence: `eval_every=0`, `save_every=0`
- log path: `artifacts/runs/dpo-rank16-qwen-qwen3-4b-instruct-2507-<timestamp>/`

Run it with:

```bash
bash autoresearch.sh
```

Intentional variants should still go through CLI overrides to this wrapper, for example:

```bash
bash autoresearch.sh --learning-rate 5e-6
bash autoresearch.sh --dpo-beta 0.2 --batch-size 4
bash autoresearch.sh --save-every 100 --eval-every 100
```

## Search signals

- Primary search signal: `eval_pair_accuracy`
- Secondary diagnostics: `eval_margin`, `eval_chosen_score`, `eval_rejected_score`
- Train-side diagnostics: `train_dpo_loss` and step timing once training is enabled for the current run

## Mutable surface

- DPO recipe knobs such as `--learning-rate`, `--dpo-beta`, `--batch-size`, `--num-epochs`, `--max-steps`, `--eval-every`, and `--save-every`
- prompt normalization and pair-shape compatibility logic in `train.py` when the held-out pairwise scoring meaning stays fixed
- wrapper defaults and local data preparation details once they are documented together with `README.md`
- the concrete files that normally move together are `train.py`, `data/train/full.jsonl`, `data/train/smoke.jsonl`, `data/eval/full.jsonl`, `data/eval/smoke.jsonl`, `data/sources.yaml`, `README.md`, and `autoresearch.sh`

## Frozen contract

- the meaning of `eval_pair_accuracy`
- the held-out pairwise scoring rule: chosen should score above rejected on the same prompt
- the separation between `--dry-run`, `--eval-only`, and train mode
- the requirement that reportable runs use real local train and eval pairs rather than scaffold placeholder rows

## Recovery and confirmation

- Checkpoint cadence: the default wrapper leaves `save_every=0`, so `train/checkpoints.jsonl` appears only when you intentionally enable checkpointing.
- Same-run resume: when checkpoint saves are enabled, rerun the same wrapper or training command with the same `--log-path` so the latest recorded `state_path` continues the interrupted run.
- Fresh restart: use `--load-checkpoint-path <state_path>` to start a new run from saved weights only.
- Clean confirmation: use a recorded `sampler_path` for later `--eval-only --eval-data data/eval/full.jsonl --base-model ...` reruns, for example:

```bash
uv run train.py --eval-only \
  --base-model '<sampler_path>' \
  --eval-data data/eval/full.jsonl
```

## Run budget and stopping

- Do not treat falling `train_dpo_loss` alone as success; held-out pair metrics are the real oracle here
- Keep a candidate running while held-out pair metrics are still improving or while the run has not yet produced enough evidence to separate noise from signal
- Stop when held-out pair metrics are repeatedly flat or worse across comparable reruns, or when the current wall-clock or step budget is exhausted
- If a checkpoint looks promising, promote it to a clean `--eval-only` rerun on `data/eval/full.jsonl` instead of extending one candidate indefinitely

Keep `autoresearch.sh`, `autoresearch.md`, and `README.md` aligned whenever the wrapper recipe changes.
