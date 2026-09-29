#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

RUN_ROOT=${RUN_ROOT:-artifacts/cfr_plus_18_reach_weighted/main_20260929}
REFERENCE_ROOT=${REFERENCE_ROOT:-artifacts/cfr_plus_18_parallel_cpu/long_20260928}
PORT=${PORT:-8765}
mkdir -p "$RUN_ROOT"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
exec .venv/bin/python -u scripts/monitor_cfr_plus_18_parallel_cpu.py \
  "$RUN_ROOT" --reference-root "$REFERENCE_ROOT" --port "$PORT" \
  --eval-workers 2 >> "$RUN_ROOT/live_monitor.log" 2>&1
