#!/usr/bin/env bash
set -euo pipefail
cd /root/liars_poker_20261001
ROOT=/root/liars_poker/artifacts/cfr_plus_30_claim_first_run/main_20261003
ARM="${1:?arm or june required}"
DEPTH="${2:-0}"
mkdir -p "$ROOT"
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
ARGS=(--output-root "$ROOT" --minutes 1440 --timeout-s 3600)
if [[ "$ARM" == june ]]; then
  ARGS+=(--june-only)
else
  ARGS+=(--arm-filter "$ARM")
fi
if [[ "$DEPTH" != 0 ]]; then
  ARGS+=(--depth-filter "$DEPTH")
fi
exec .venv/bin/python -u scripts/evaluate_cfr_plus_30_claim_first_run.py worker \
  "${ARGS[@]}" >> "$ROOT/eval_${ARM}_d${DEPTH}.log" 2>&1
