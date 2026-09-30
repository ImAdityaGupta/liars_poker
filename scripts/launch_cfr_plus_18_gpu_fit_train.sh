#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
RUN=artifacts/cfr_plus_18_gpu_fit_forks/overnight_20260930
mkdir -p "$RUN"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
exec .venv/bin/python -u scripts/run_cfr_plus_18_gpu_fit_forks.py \
  --source-checkpoint artifacts/cfr_plus_18_gpu_fit_forks/source/source_checkpoint.pt \
  --source-state artifacts/cfr_plus_18_gpu_fit_forks/source/source_state.json \
  --output-dir "$RUN" --hours-per-arm 3 --arms 96,384 --interval-minutes 15 \
  >> "$RUN/train.log" 2>&1
