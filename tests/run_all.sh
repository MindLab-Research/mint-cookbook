#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

echo "Running sync smoke..."
uv run python -u sync_smoke.py "$@"

echo "================================================"

echo "Running async smoke..."
uv run python -u async_smoke.py "$@"

echo "================================================"

echo "Running checkpoint resume smoke..."
uv run python -u checkpoint_resume_smoke.py "$@"

echo "================================================"

echo "Running loss smoke..."
uv run python -u loss_smoke.py "$@"
