#!/usr/bin/env bash
# Resume the independent CPU checkpoint evaluator; never starts a trainer.
set -euo pipefail

repo=/root/liars_poker_20261001
run=/root/liars_poker/artifacts/cfr_plus_30_claim_first_run/main_20261003
session=cfr30precise

test -x "$repo/.venv/bin/python"
test -d "$run"
mkdir -p "$run/precise_evaluations"
if tmux has-session -t "$session" 2>/dev/null; then
  echo "$session is already running"
  exit 0
fi

tmux new-session -d -s "$session" \
  "cd '$repo' && CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 .venv/bin/python -u scripts/evaluate_cfr_plus_30_precise.py queue --output-root '$run' --workers 30 --shards 30 --sweeps 58 --timeout-s 1800 >> '$run/precise_evaluations/queue.log' 2>&1"
echo "Started $session; log: $run/precise_evaluations/queue.log"
