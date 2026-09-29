#!/usr/bin/env bash
# Run from /root/liars_poker in a separate tmux session from training.
set -euo pipefail
RUN_ROOT=artifacts/cfr_plus_18_parallel_cpu/long_20260928
mkdir -p "$RUN_ROOT"
exec .venv/bin/python -u scripts/monitor_cfr_plus_18_parallel_cpu.py \
  "$RUN_ROOT" --port 8765 --eval-workers 2 \
  >> "$RUN_ROOT/live_monitor.log" 2>&1
