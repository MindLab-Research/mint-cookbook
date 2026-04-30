#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOG_PATH="artifacts/runs/dpo-rank16-qwen-qwen3-4b-instruct-2507-$(date +%Y%m%d-%H%M%S)"

exec uv run train.py \
  --train-data data/train/full.jsonl \
  --eval-data data/eval/full.jsonl \
  --log-path "$LOG_PATH" \
  --base-model Qwen/Qwen3-4B-Instruct-2507 \
  --rank 16 \
  --learning-rate 1e-5 \
  --dpo-beta 0.1 \
  --batch-size 8 \
  --num-epochs 1 \
  --max-steps 0 \
  --eval-every 0 \
  --save-every 0 \
  --mint-timeout 600 \
  "$@" 2>&1
