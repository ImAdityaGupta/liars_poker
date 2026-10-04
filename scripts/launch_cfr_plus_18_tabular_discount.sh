#!/usr/bin/env bash
# Launch all six independent, resumable CPU discount arms and their dashboard.
set -euo pipefail

REPO=/root/liars_poker_discount
OUTPUT=/root/liars_poker/artifacts/cfr_plus_18_tabular_discount/main_20260930
BRIDGE=/root/liars_poker/artifacts/cfr_plus_18_tabular_bridge/main_20260929
PYTHON="$REPO/.venv/bin/python"

test -x "$PYTHON"
test -f "$REPO/liars_poker/algo/cfr_discount_tabular.py"
test -f "$REPO/scripts/run_cfr_plus_18_tabular_discount.py"
test -f "$BRIDGE/conditional/evaluations.jsonl"
command -v tmux >/dev/null
mkdir -p "$OUTPUT"

arms=(
  V_cfr_uniform A_cfr_plus_linear B_cfr_plus_quadratic
  C_dcfr_plus_quadratic D_dcfr_exact_quadratic E_dcfr_visited_quadratic
)

for arm in "${arms[@]}"; do
  session="cfr18_discount_${arm%%_*}"
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "$session already running"
    continue
  fi
  mkdir -p "$OUTPUT/$arm"
  tmux new-session -d -s "$session" \
    "cd '$REPO' && CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 '$PYTHON' -u scripts/run_cfr_plus_18_tabular_discount.py --output-root '$OUTPUT' --arm '$arm' --minutes-per-arm 180 --threads 8 >> '$OUTPUT/$arm/console.log' 2>&1"
  echo "started $session"
done

if ! tmux has-session -t cfr18_discount_dashboard 2>/dev/null; then
  tmux new-session -d -s cfr18_discount_dashboard \
    "cd '$REPO' && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 '$PYTHON' -u scripts/monitor_cfr_plus_18_tabular_discount.py --output-root '$OUTPUT' --bridge-root '$BRIDGE' --port 8768 >> '$OUTPUT/dashboard.log' 2>&1"
  echo "started cfr18_discount_dashboard on port 8768"
fi

echo "results: $OUTPUT"
