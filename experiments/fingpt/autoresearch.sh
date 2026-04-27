#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOG_PATH="artifacts/runs/sft-1epoch-sentiment-qwen3-4b-$(date +%Y%m%d-%H%M%S)"

exec uv run train.py \
  --task-type sentiment \
  --base-model Qwen/Qwen3-4B-Instruct-2507 \
  --log-path "$LOG_PATH" \
  --train-data "data/fingpt-sentiment-train/train.jsonl" \
  --eval-data "fpb:data/benchmarks/sentiment/fpb/test.jsonl,fiqa-sa:data/benchmarks/sentiment/fiqa-sa/test.jsonl,tfns:data/benchmarks/sentiment/tfns/test.jsonl,nwgi:data/benchmarks/sentiment/nwgi/test.jsonl" \
  --train-eval-data "train-eval-160:data/benchmarks/sentiment/train-eval-160/all/test.jsonl" \
  --rank 16 \
  --num-epochs 1 \
  --batch-size 32 \
  --learning-rate 1e-4 \
  --lr-schedule linear \
  --eval-max-tokens 256 \
  --eval-limit 100 \
  --max-concurrent-requests 128 \
  --eval-every 10 \
  --save-every 10 \
  --train-metrics-every 1 \
  --train-print-every 1 \
  --mint-timeout 600 \
  --seed 42 \
  "$@" 2>&1
