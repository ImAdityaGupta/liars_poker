#!/usr/bin/env bash
set -euo pipefail
cd /root/liars_poker_discount
run_root=/root/liars_poker/artifacts/cfr_plus_18_tabular_discount/main_20260930
arms=(V_cfr_uniform A_cfr_plus_linear B_cfr_plus_quadratic C_dcfr_plus_quadratic D_dcfr_exact_quadratic E_dcfr_visited_quadratic)
for arm in "${arms[@]}"; do
  session="discount_continue_${arm%%_*}"
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "already running: $session"
    continue
  fi
  cmd="cd /root/liars_poker_discount && CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 .venv/bin/python -u scripts/continue_cfr_plus_18_tabular_discount.py --output-root '$run_root' --arm '$arm' --minutes-per-arm 540 --threads 8 >> '$run_root/$arm/continuation.log' 2>&1"
  tmux new-session -d -s "$session" "$cmd"
  echo "started: $session"
done
if ! tmux has-session -t discount_eval_queue 2>/dev/null; then
  cmd="cd /root/liars_poker_discount && CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 .venv/bin/python -u scripts/evaluate_cfr_plus_18_tabular_discount_queue.py --output-root '$run_root' --evaluator /root/liars_poker/scripts/evaluate_cfr_plus_18_fit_snapshot.py --workers 2 >> '$run_root/evaluator.log' 2>&1"
  tmux new-session -d -s discount_eval_queue "$cmd"
  echo "started: discount_eval_queue"
fi
