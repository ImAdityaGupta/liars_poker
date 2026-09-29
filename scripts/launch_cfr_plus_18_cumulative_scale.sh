#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

RUN_ROOT=${RUN_ROOT:-artifacts/cfr_plus_18_cumulative_regret/main_20260929}
mkdir -p "$RUN_ROOT"
exec .venv/bin/python -u scripts/run_cfr_plus_18_cumulative_scale.py \
  --output-root "$RUN_ROOT" --minutes-per-arm 330 \
  --threads-per-arm 8 --snapshot-minutes 15 --checkpoint-minutes 15 \
  >> "$RUN_ROOT/supervisor.log" 2>&1
