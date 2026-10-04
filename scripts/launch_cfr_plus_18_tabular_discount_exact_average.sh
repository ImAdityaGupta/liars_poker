#!/usr/bin/env bash
# Launch the five new exact-average discount arms in independent resumable tmux sessions.
set -euo pipefail

REPO=${REPO:-/root/liars_poker_20261001}
OUTPUT=${OUTPUT:-/root/liars_poker/artifacts/cfr_plus_18_tabular_discount_exact_average/main_20261002}
PYTHON="$REPO/.venv/bin/python"
THREADS=${THREADS:-4}
MINUTES=${MINUTES:-540}

test -x "$PYTHON"
test -f "$REPO/scripts/run_cfr_plus_18_tabular_discount_exact_average.py"
command -v tmux >/dev/null
mkdir -p "$OUTPUT"

arms=(V_cfr_uniform B_cfr_plus_quadratic C_dcfr_plus_quadratic D_dcfr_exact_quadratic E_dcfr_visited_quadratic)
for arm in "${arms[@]}"; do
  session="cfr18_exactavg_${arm%%_*}"
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "$session already running"
    continue
  fi
  mkdir -p "$OUTPUT/$arm"
  tmux new-session -d -s "$session" \
    "cd '$REPO' && CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 '$PYTHON' -u scripts/run_cfr_plus_18_tabular_discount_exact_average.py --output-root '$OUTPUT' --arm '$arm' --minutes '$MINUTES' --threads '$THREADS' >> '$OUTPUT/$arm/console.log' 2>&1"
  echo "started $session"
done

echo "exact-average discount results: $OUTPUT"
