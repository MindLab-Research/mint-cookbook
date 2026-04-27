# Autoresearch: chat-dpo

`autoresearch.sh` is the canonical automation wrapper for this experiment's current runnable line.
It is not automatically the final report or confirmation command.
This file is the wrapper protocol for search work. Keep environment setup, data preparation, and full reproduction commands in `README.md`.

## Objective

Improve held-out pairwise preference results on `data/eval.jsonl`, not redefine the benchmark.

## Current wrapper

- base model: `Qwen/Qwen3-4B-Instruct-2507`
- train data: `data/train.jsonl`
- final eval data: `data/eval.jsonl`
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
- the concrete files that normally move together are `train.py`, `data/train.jsonl`, `data/eval.jsonl`, `data/sources.yaml`, `README.md`, and `autoresearch.sh`

## Frozen contract

- the meaning of `eval_pair_accuracy`
- the held-out pairwise scoring rule: chosen should score above rejected on the same prompt
- the separation between `--dry-run`, `--eval-only`, and train mode
- the requirement that reportable runs use real local train and eval pairs rather than scaffold placeholder rows

## Recovery and confirmation

- The current wrapper does not save periodic checkpoints unless you override `--save-every`
- When checkpoint saves are enabled:
  - same-run resume: rerun the same wrapper/training command with the same `--log-path` to continue from the latest recorded `state_path`
  - fresh restart: use `--load-checkpoint-path <state_path>` to start a new run from weights only
  - clean confirmation: use recorded `sampler_path` entries for later `--eval-only --base-model ...` reruns
- For a clean held-out confirmation rerun, prefer:

```bash
uv run train.py --eval-only \
  --base-model '<sampler_path>' \
  --eval-data data/eval.jsonl
```

## Run budget and stopping

- Do not treat falling `train_dpo_loss` alone as success; held-out pair metrics are the real oracle here
- Keep a candidate running while held-out pair metrics are still improving or while the run has not yet produced enough evidence to separate noise from signal
- Stop when held-out pair metrics are repeatedly flat or worse across comparable reruns, or when the current wall-clock or step budget is exhausted
- If a checkpoint looks promising, promote it to a clean `--eval-only` rerun on `data/eval.jsonl` instead of extending one candidate indefinitely

Keep `autoresearch.sh`, `autoresearch.md`, and `README.md` aligned whenever the wrapper recipe changes.
