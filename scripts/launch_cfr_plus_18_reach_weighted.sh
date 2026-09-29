#!/usr/bin/env bash
# Two seed-17 aggregate-first neural arms with N/K on only the fresh regret.
set -euo pipefail
cd "$(dirname "$0")/.."

RUN_ROOT=${RUN_ROOT:-artifacts/cfr_plus_18_reach_weighted/main_20260929}
RESUME_ARGS=()
if [[ ${RESUME:-0} == 1 ]]; then RESUME_ARGS=(--resume); fi
mkdir -p "$RUN_ROOT"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=""

exec .venv/bin/python -u scripts/run_cfr_plus_18_parallel_cpu.py \
  --output-root "$RUN_ROOT" \
  --minutes-per-arm 330 --traversals 1024,4096 --seeds 17 \
  --modes aggregate_then_clip --reach-mode visit_fraction \
  --threads-per-arm 8 --max-parallel 2 \
  --snapshot-minutes 15 --checkpoint-minutes 30 --train-only \
  "${RESUME_ARGS[@]}" >> "$RUN_ROOT/supervisor.log" 2>&1
