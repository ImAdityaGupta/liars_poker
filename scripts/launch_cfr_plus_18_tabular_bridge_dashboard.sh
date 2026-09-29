#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
RUN_ROOT=${RUN_ROOT:-artifacts/cfr_plus_18_tabular_bridge/main_20260929}
PORT=${PORT:-8766}
mkdir -p "$RUN_ROOT"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
exec .venv/bin/python -u scripts/monitor_cfr_plus_18_tabular_bridge.py \
  --output-root "$RUN_ROOT" --port "$PORT" \
  >> "$RUN_ROOT/dashboard.log" 2>&1
