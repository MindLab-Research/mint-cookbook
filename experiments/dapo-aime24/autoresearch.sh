#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOG_PATH="artifacts/runs/grpo-rank32-qwen3-4b-instruct-2507-$(date +%Y%m%d-%H%M%S)"

source "$SCRIPT_DIR/cache-env.sh"

exec uv run train.py \
  --train-data data/train.jsonl \
  --eval-data data/eval.jsonl \
  --log-path "$LOG_PATH" \
  --base-model Qwen/Qwen3-4B-Instruct-2507 \
  --rank 32 \
  --grpo-steps 180 \
  --groups-per-batch 32 \
  --group-size 16 \
  --mint-timeout 6000 \
  --rl-learning-rate 1e-4 \
  --rl-temperature 1.0 \
  --rl-max-tokens 8192 \
  --overlong-buffer-len 4096 \
  --overlong-buffer-penalty-factor 1.0 \
  --dynamic-sampling-type filter \
  --dynamic-sampling-max-rollout-waves 30 \
  --eval-num-samples 1 \
  --eval-temperature 1.0 \
  --eval-top-p 0.7 \
  --eval-every-steps 5 \
  --tail-grace-seconds 30 \
  --save-every-steps 1 \
  --max-concurrent-requests 512 \
  --seed 42 \
  "$@" 2>&1
