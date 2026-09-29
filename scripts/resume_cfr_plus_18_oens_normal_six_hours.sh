#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

RUN=artifacts/cfr_plus_18_oens_followups/main_20260929/normal
SESSION=oens_normal_extend

test -x .venv/bin/python
test -f "$RUN/manifest.json"
test -f "$RUN/state.json"
test -f "$RUN/checkpoints/0360m.pt"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "$SESSION is already running" >&2
  exit 1
fi

free_kib=$(df -Pk . | awk 'NR==2 {print $4}')
if (( free_kib < 8 * 1024 * 1024 )); then
  echo "Need at least 8 GiB free disk to resume OENS" >&2
  exit 1
fi

# The runner restores elapsed=360m from state.json. A 720m target therefore
# continues this same seed/checkpoint for roughly six additional training hours.
tmux new-session -d -s "$SESSION" \
  "cd /root/liars_poker && CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 .venv/bin/python -u scripts/run_cfr_plus_18_oens_followups.py --mode normal --output-dir $RUN --training-minutes 720 --monitor-minutes 15 --threads 8 --seed 31 --resume >> $RUN/console.log 2>&1"
echo "session: $SESSION"
echo "run: /root/liars_poker/$RUN"
echo "log: /root/liars_poker/$RUN/console.log"
echo "target: 720 total measured minutes (approximately 360 additional minutes)"
