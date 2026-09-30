#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
RUN=artifacts/cfr_plus_18_gpu_fit_forks/overnight_20260930
TABULAR=artifacts/cfr_plus_18_gpu_fit_forks/overnight_20260930/tabular_cumulative
mkdir -p "$RUN"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
exec .venv/bin/python -u scripts/monitor_cfr_plus_18_gpu_fit_forks.py "$RUN" \
  --cpu-root artifacts/cfr_plus_18_cumulative_regret/main_20260929 \
  --tabular-fork "$TABULAR" --tabular-target-min 180 --port 8767 \
  >> "$RUN/dashboard.log" 2>&1
