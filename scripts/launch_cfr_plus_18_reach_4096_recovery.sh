#!/usr/bin/env bash
# Recover the 4096-root neural arm with a buffer that retains every visit.
set -euo pipefail
cd "$(dirname "$0")/.."

BASE=${BASE:-artifacts/cfr_plus_18_reach_weighted/main_20260929}
ARM_ROOT="$BASE/trav4096__aggregate_then_clip__seed17"
ARM_DIR="$ARM_ROOT/aggregate_then_clip__seed_17"
mkdir -p "$ARM_DIR"
RESUME_ARGS=()
if [[ -f "$ARM_DIR/latest_checkpoint.pt" && -f "$ARM_DIR/state.json" ]]; then
  RESUME_ARGS=(--resume)
elif [[ -f "$ARM_DIR/training.jsonl" ]]; then
  stamp=$(date -u +%Y%m%d-%H%M%S)
  mv "$ARM_DIR/training.jsonl" "$ARM_DIR/training.failed_$stamp.jsonl"
  if [[ -f "$ARM_DIR/manifest.json" ]]; then
    mv "$ARM_DIR/manifest.json" "$ARM_DIR/manifest.failed_$stamp.json"
  fi
fi

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=""
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=1

exec .venv/bin/python -u scripts/run_cfr_plus_18_target_order_cpu_overnight.py \
  --output-root "$ARM_ROOT" --hours-per-arm 5.5 \
  --traversals 4096 --seeds 17 --modes aggregate_then_clip \
  --reach-mode visit_fraction --regret-buffer-capacity 4000000 \
  --torch-threads 8 --snapshot-minutes 15 --checkpoint-minutes 30 \
  "${RESUME_ARGS[@]}" >> "$ARM_ROOT/recovery.log" 2>&1
