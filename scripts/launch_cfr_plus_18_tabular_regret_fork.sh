#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

RUN=artifacts/cfr_plus_18_tabular_regret_forks/oens_0300m
SOURCE=artifacts/cfr_plus_18_tabular_regret_fork_sources/oens_normal_0300m.pt
SESSION=cfr18_tabular_fork
MODE=${1:-start}
if [[ "$MODE" != start && "$MODE" != resume ]]; then
  echo "usage: bash $0 [start|resume]" >&2
  exit 2
fi
test -x .venv/bin/python
test -f "$SOURCE"
command -v tmux >/dev/null
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Session $SESSION is already running" >&2
  exit 1
fi

free_kib=$(df -Pk . | awk 'NR==2 {print $4}')
if (( free_kib < 8 * 1024 * 1024 )); then
  echo "Need at least 8 GiB free disk to launch the fork" >&2
  exit 1
fi

if [[ "$MODE" == start ]]; then
  if [[ -e "$RUN/manifest.json" || -e "$RUN/latest_checkpoint.pt" ]]; then
    echo "Run already exists; use resume" >&2
    exit 1
  fi
  run_flags="--source-checkpoint $SOURCE"
else
  test -f "$RUN/manifest.json"
  test -f "$RUN/latest_checkpoint.pt"
  run_flags="--resume"
fi

mkdir -p "$RUN"
tmux new-session -d -s "$SESSION" "cd /root/liars_poker && CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 .venv/bin/python -u scripts/run_cfr_plus_18_tabular_regret_fork.py $run_flags --output-dir $RUN --additional-minutes 600 --monitor-minutes 15 --checkpoint-minutes 15 --traversals 4096 --threads 8 >> $RUN/console.log 2>&1"
echo "session: $SESSION"
echo "run: /root/liars_poker/$RUN"
echo "log: /root/liars_poker/$RUN/console.log"
