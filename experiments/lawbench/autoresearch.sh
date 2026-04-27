#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# LOG_PATH="artifacts/runs/sft-1epoch-qwen3-4b-$(date +%Y%m%d-%H%M%S)"
LOG_PATH="artifacts/runs/sft-1epoch-qwen3-4b-20260423-221504"

exec uv run train.py \
  --base-model Qwen/Qwen3-4B-Instruct-2507 \
  --train-data data/train/full.jsonl \
  --eval-data data/eval/full.jsonl \
  --train-eval-data data/eval/train_eval_200.jsonl \
  --log-path "$LOG_PATH" \
  --rank 16 \
  --num-epochs 1 \
  --batch-size 256 \
  --learning-rate 1e-4 \
  --lr-schedule cosine \
  --eval-max-tokens 1024 \
  --max-concurrent-requests 128 \
  --eval-every 10 \
  --save-every 10 \
  --train-metrics-every 1 \
  --train-print-every 1 \
  --mint-timeout 600 \
  --seed 42 \
  "$@" 2>&1
