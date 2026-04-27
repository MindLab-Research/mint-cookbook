#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOG_PATH="artifacts/runs/eval-only-$(date +%Y%m%d-%H%M%S)"

exec uv run train.py \
  --eval-only \
  --log-path "$LOG_PATH" \
  "$@" 2>&1
