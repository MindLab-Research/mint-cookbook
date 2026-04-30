# Autoresearch: dapo-aime

`autoresearch.sh` is the canonical automation wrapper for this experiment's current runnable line.
It is not automatically the final report or confirmation command.
This file is the wrapper protocol for search work. Keep environment setup, data preparation, and full reproduction commands in `README.md`.

## Objective

Improve AIME 2024 benchmark performance from direct GRPO on local `DAPO-Math-17k`, while keeping the same row contract available for AIME 2025 and AIME 2026 confirmation runs.

## Current wrapper

- base model: `Qwen/Qwen3-4B-Instruct-2507`
- train data: local `data/train/full.jsonl` derived from `DAPO-Math-17k`
- final eval data: `data/eval/aime2024.jsonl`, `data/eval/aime2025.jsonl`, `data/eval/aime2026.jsonl`, with AIME 2024 kept as the primary benchmark anchor
- train config: `rank=32`, `grpo_steps=180`, `groups_per_batch=32`, `group_size=16`, `rl_learning_rate=1e-4`, `rl_temperature=1.0`
- eval config inside the wrapper: `eval_num_samples=1`, `eval_temperature=1.0`, `eval_top_p=0.7`
- throughput and cadence defaults: `tail_grace_seconds=30`, `eval_every_steps=5`, `save_every_steps=1`, `max_concurrent_requests=128`, `mint_timeout=6000`
- training-time periodic eval: one stable random third from each named AIME eval manifest, combined into one mixed eval set
- shaping defaults: `overlong_buffer_len=4096`, `overlong_buffer_penalty_factor=1.0`, `dynamic_sampling_type=filter`, `dynamic_sampling_max_rollout_waves=30`
- log path: `artifacts/runs/grpo-rank32-qwen3-4b-instruct-2507-<timestamp>/`

Run it with:

```bash
bash autoresearch.sh
```

Intentional variants should still go through CLI overrides to this wrapper, for example:

```bash
bash autoresearch.sh --grpo-steps 360
bash autoresearch.sh --group-size 8 --groups-per-batch 48
bash autoresearch.sh --eval-only
```

## Search signals

- Primary search signal: `eval_accuracy`
- Secondary diagnostics: `eval_greedy_accuracy`, `eval_pass_at_k`
- Train-side diagnostics: reward stability, accepted vs trained groups, rollout-wave behavior, throughput, and client health

## Mutable surface

- RL budget and batch-shape knobs such as `--grpo-steps`, `--groups-per-batch`, `--group-size`, `--stream-minibatches-per-step`, `--tail-grace-seconds`, and `--max-concurrent-requests`
- RL optimization and shaping knobs such as `--rl-learning-rate`, `--rl-temperature`, `--dynamic-sampling-type`, `--dynamic-sampling-max-rollout-waves`, and overlong-penalty settings
- bounded implementation details in `train.py` that preserve the frozen AIME 2024 benchmark contract and the shared row schema used by AIME 2025/2026 manifests
- the concrete files that normally move together are `train.py`, `data/train/full.jsonl`, `data/train/smoke.jsonl`, `data/train/raw_dapo_math_17k.parquet`, `data/eval/aime2024.jsonl`, `data/eval/aime2025.jsonl`, `data/eval/aime2026.jsonl`, `data/eval/smoke.jsonl`, `data/eval/raw_aime_2024.jsonl`, `data/eval/raw_aime_2025.parquet`, `data/eval/raw_aime_2026.parquet`, `data/sources.yaml`, `README.md`, and `autoresearch.sh`

## Frozen contract

- the frozen `data/eval/aime2024.jsonl` AIME 2024 benchmark file
- the prompt and grading contract around the final `Answer: ...` line unless a change is required to fix a broken measurement path
- the meaning of `eval_accuracy`, `eval_greedy_accuracy`, or `eval_pass_at_k`
- the direct-GRPO scope of this experiment by silently reintroducing an SFT warm start into the canonical wrapper

## Recovery and confirmation

- Checkpoint cadence: the wrapper saves a checkpoint every step, so `train/checkpoints.jsonl` is part of the default search path rather than an optional extra.
- Same-run resume: rerun with the same `--log-path` so the latest `state_path` row in `train/checkpoints.jsonl` restores optimizer and loop state.
- Fresh restart: use `--load-checkpoint-path <state_path>` when you want saved weights in a new run directory without reusing the interrupted run's optimizer state or append-only logs.
- Clean confirmation: `--eval-only` uses `--base-model` with a recorded `sampler_path`, not a separate resume flag. For a clean benchmark confirmation rerun under the same frozen AIME file, prefer:

```bash
uv run train.py --eval-only \
  --eval-data data/eval/aime2024.jsonl \
  --base-model '<sampler_path_from_train/checkpoints.jsonl>'
```

- Compatibility note: raw `train.py` and `autoresearch.sh` now share the same default base model: `Qwen/Qwen3-4B-Instruct-2507`. Older checkpoints created under another base model still need that original `--base-model` passed explicitly when you resume or replay them.
- Saved `sampler_path` exports are also useful when you want to replay the exact sampling snapshot created for periodic background evals.

## Run budget and stopping

- Do not keep a candidate alive only because train-side reward wiggles upward; AIME eval remains the real oracle
- Do not kill a candidate only because one sampled eval is noisy; compare `eval_accuracy`, `eval_greedy_accuracy`, and `eval_pass_at_k` across multiple checkpoints under the same contract
- Keep running while checkpointed AIME metrics still improve or while the run is primarily exposing recovery or stability issues that need to be fixed first
- Stop on repeated flat or degrading AIME checkpoints, persistent backend instability without recovery progress, or budget exhaustion
- Promote promising checkpoints to explicit `--eval-only` confirmation reruns instead of extending one candidate indefinitely

Keep `autoresearch.sh`, `autoresearch.md`, and `README.md` aligned whenever the wrapper recipe changes.
