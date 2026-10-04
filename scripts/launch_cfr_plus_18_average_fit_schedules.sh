#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/root/liars_poker}
PYTHON="$REPO/.venv/bin/python"
OUTPUT="$REPO/artifacts/cfr_plus_18_average_fit_schedules/main_20261001"
SESSION=cfr18_average_fit_schedules
MODE=--schedules
if [[ "${SMOKE:-0}" == 1 ]]; then
  OUTPUT="$REPO/artifacts/cfr_plus_18_average_fit_schedules/smoke_20261001"
  SESSION=cfr18_average_fit_smoke
  MODE=--smoke-schedules
fi
mkdir -p "$OUTPUT"
test -x "$PYTHON"

if ! tmux has-session -t "$SESSION" 2>/dev/null; then
  tmux new-session -d -s "$SESSION" \
    "cd '$REPO' && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 '$PYTHON' -u scripts/run_cfr_plus_18_average_fit_optimizer_experiment.py '$MODE' --output-root '$OUTPUT' >> '$OUTPUT/run.log' 2>&1"
fi
echo "results: $OUTPUT"
