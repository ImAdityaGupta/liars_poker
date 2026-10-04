#!/usr/bin/env bash
# Extend the tabular-regret fork displayed on port 8767 by six training hours.
set -euo pipefail
cd /root/liars_poker

RUN=artifacts/cfr_plus_18_gpu_fit_forks/overnight_20260930/tabular_cumulative
ROOT=artifacts/cfr_plus_18_gpu_fit_forks/overnight_20260930
SESSION=cfr18_tabular_fit_extend
test -f "$RUN/latest_checkpoint.pt"
test -f "$RUN/manifest.json"
test -x .venv/bin/python
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "$SESSION already running"
  exit 1
fi

# The previous target was 180 measured fork minutes; the new total is 540.
tmux new-session -d -s "$SESSION" \
  "cd /root/liars_poker && CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 .venv/bin/python -u scripts/run_cfr_plus_18_tabular_regret_fork.py --resume --output-dir '$RUN' --additional-minutes 540 --monitor-minutes 15 --checkpoint-minutes 15 --traversals 4096 --threads 8 >> '$RUN/console.log' 2>&1"
echo "started $SESSION: 180 -> 540 measured fork minutes"

# The dashboard's only change is its displayed target. Its exact GPU fork
# evaluations are already complete; restarting it leaves their files intact.
if tmux has-session -t cfr18_gpu_fit_dashboard 2>/dev/null; then
  tmux kill-session -t cfr18_gpu_fit_dashboard
fi
tmux new-session -d -s cfr18_gpu_fit_dashboard \
  "cd /root/liars_poker && CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 .venv/bin/python -u scripts/monitor_cfr_plus_18_gpu_fit_forks.py '$ROOT' --cpu-root artifacts/cfr_plus_18_cumulative_regret/main_20260929 --tabular-fork '$RUN' --tabular-target-min 540 --port 8767 >> '$ROOT/dashboard.log' 2>&1"
echo "port 8767 dashboard now displays a 540-minute target"
