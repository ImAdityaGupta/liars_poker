#!/usr/bin/env bash
# Six 18-claim bridge arms: sample_both and conditional at K=128,256,512.
set -euo pipefail
cd "$(dirname "$0")/.."

RUN_ROOT=${RUN_ROOT:-artifacts/cfr_plus_18_tabular_bridge/main_20260929/low_roots}
MINUTES=${MINUTES:-180}
ROOTS=${ROOTS:-"128 256 512"}
mkdir -p "$RUN_ROOT"
export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

pids=()
for roots in $ROOTS; do
  dir="$RUN_ROOT/k$(printf '%04d' "$roots")"
  mkdir -p "$dir"
  args=(--output-root "$dir" --arms sample_both,conditional
        --seed 17 --roots "$roots" --minutes "$MINUTES"
        --eval-minutes 15 --checkpoint-minutes 30 --threads-per-arm 1)
  if [[ ${RESUME:-0} == 1 ]]; then args+=(--resume); fi
  .venv/bin/python -u scripts/run_cfr_plus_18_tabular_bridge.py "${args[@]}" \
    >> "$dir/supervisor.log" 2>&1 &
  pids+=("$!")
  echo "K=$roots supervisor pid=${pids[-1]} log=$dir/supervisor.log"
done

failure=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then failure=1; fi
done
exit "$failure"
