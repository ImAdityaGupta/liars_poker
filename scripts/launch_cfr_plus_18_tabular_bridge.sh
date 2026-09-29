#!/usr/bin/env bash
# Run in tmux from /root/liars_poker. Set RESUME=1 after a host restart.
set -euo pipefail
cd "$(dirname "$0")/.."

RUN_ROOT=${RUN_ROOT:-artifacts/cfr_plus_18_tabular_bridge/main_20260929}
RESUME_ARGS=()
if [[ ${RESUME:-0} == 1 ]]; then
  RESUME_ARGS=(--resume)
fi
mkdir -p "$RUN_ROOT"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=""
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

exec .venv/bin/python -u scripts/run_cfr_plus_18_tabular_bridge.py \
  --output-root "$RUN_ROOT" \
  --seed 17 --roots 1024 --minutes 300 \
  --eval-minutes 15 --checkpoint-minutes 30 \
  --threads-per-arm 1 "${RESUME_ARGS[@]}" \
  >> "$RUN_ROOT/supervisor.log" 2>&1
