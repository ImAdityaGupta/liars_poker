#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

RUN_ROOT=${RUN_ROOT:-artifacts/cfr_plus_18_cumulative_regret/main_20260929}
REFERENCE_ROOT=${REFERENCE_ROOT:-artifacts/cfr_plus_18_parallel_cpu/long_20260928}
TABULAR_FORK=${TABULAR_FORK:-artifacts/cfr_plus_18_tabular_regret_forks/oens_0300m}
PORT=${PORT:-8765}
mkdir -p "$RUN_ROOT"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
exec .venv/bin/python -u scripts/monitor_cfr_plus_18_parallel_cpu.py \
  "$RUN_ROOT" --reference-root "$REFERENCE_ROOT" --port "$PORT" \
  --tabular-fork "$TABULAR_FORK" --fork-start-min 300 \
  --eval-workers 2 >> "$RUN_ROOT/live_monitor.log" 2>&1
