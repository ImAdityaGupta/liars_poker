#!/usr/bin/env bash
set -euo pipefail
cd /root/liars_poker_discount
root=/root/liars_poker/artifacts/cfr_plus_18_offline_average_study
stage="${1:-0030m}"
mode="${2:-first}"
case "$stage" in
  0030m|0045m) ;;
  *) echo "unsupported stage: $stage" >&2; exit 2 ;;
esac
checkpoint="$root/exact4096_${stage}_checkpoint.pt"
test -f "$checkpoint"
case "$mode" in
  first) suffix="refit_${stage}"; options="" ;;
  extended) suffix="refit_${stage}_extended"; options="--variants warm --milestones 5000 10000 20000" ;;
  *) echo "unsupported mode: $mode" >&2; exit 2 ;;
esac
mkdir -p "$root/$suffix"
session="cfr18_offline_average_${stage}_${mode}"
if tmux has-session -t "$session" 2>/dev/null; then
  echo "already running: $session"
  exit 0
fi
command="cd /root/liars_poker_discount && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 .venv/bin/python -u scripts/run_cfr_plus_18_offline_average_fitting.py --checkpoint '$checkpoint' --output '$root/$suffix' $options >> '$root/$suffix.log' 2>&1"
tmux new-session -d -s "$session" "$command"
echo "started: $session; log: $root/$suffix.log"
