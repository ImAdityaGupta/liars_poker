#!/usr/bin/env bash
# Run from /root/liars_poker inside tmux on the rented CPU host.
set -euo pipefail

RUN_ROOT=artifacts/cfr_plus_18_parallel_cpu/long_20260928
THREADS_PER_ARM=${THREADS_PER_ARM:-8}
TARGET_MINUTES=${TARGET_MINUTES:-150}
RESUME_ARGS=()
if [[ ${RESUME:-0} == 1 ]]; then
  RESUME_ARGS=(--resume)
fi
mkdir -p "$RUN_ROOT"
export PYTHONUNBUFFERED=1
exec .venv/bin/python -u scripts/run_cfr_plus_18_parallel_cpu.py \
  --output-root "$RUN_ROOT" \
  --minutes-per-arm "$TARGET_MINUTES" \
  --traversals 1024,4096 \
  --seeds 17,23 \
  --threads-per-arm "$THREADS_PER_ARM" \
  --max-parallel 8 \
  --snapshot-minutes 15 \
  --checkpoint-minutes 30 \
  "${RESUME_ARGS[@]}" \
  >> "$RUN_ROOT/supervisor.log" 2>&1
