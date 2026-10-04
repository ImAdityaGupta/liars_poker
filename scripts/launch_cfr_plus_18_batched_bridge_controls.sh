#!/usr/bin/env bash
set -euo pipefail
cd /root/liars_poker_discount
run_root=/root/liars_poker/artifacts/cfr_plus_18_batched_bridge_controls/main_20260930
mkdir -p "$run_root/exact4096" "$run_root/neural1024"
for control in exact4096 neural1024; do
  session="bridge_batched_$control"
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "already running: $session"
    continue
  fi
  cmd="cd /root/liars_poker_discount && CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 .venv/bin/python -u scripts/run_cfr_plus_18_batched_bridge_controls.py --output-root '$run_root' --control '$control' --minutes 540 --threads 8 >> '$run_root/$control/console.log' 2>&1"
  tmux new-session -d -s "$session" "$cmd"
  echo "started: $session"
done
if ! tmux has-session -t bridge_batched_eval 2>/dev/null; then
  cmd="cd /root/liars_poker_discount && CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 .venv/bin/python -u scripts/evaluate_cfr_plus_18_tabular_discount_queue.py --output-root '$run_root' --evaluator /root/liars_poker_discount/scripts/evaluate_cfr_plus_18_batched_bridge_snapshot.py --workers 2 --arms exact4096 neural1024 >> '$run_root/evaluator.log' 2>&1"
  tmux new-session -d -s bridge_batched_eval "$cmd"
  echo "started: bridge_batched_eval"
fi
