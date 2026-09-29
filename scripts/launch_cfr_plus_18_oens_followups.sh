#!/usr/bin/env bash
# Start or resume the two independent CPU jobs and the read-only audit dashboard.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=${ROOT:-artifacts/cfr_plus_18_oens_followups/main_20260929}
SOURCE=${SOURCE:-artifacts/cfr_plus_18_parallel_cpu/long_20260928/trav4096__aggregate_then_clip__seed17/aggregate_then_clip__seed_17/latest_checkpoint.pt}
MODE=${1:-start}
if [[ "$MODE" != start && "$MODE" != resume ]]; then
  echo "usage: bash $0 [start|resume]" >&2
  exit 2
fi
test -f "$SOURCE"
test -x .venv/bin/python
command -v tmux >/dev/null
free_kib=$(df -Pk . | awk 'NR==2 {print $4}')
if (( free_kib < 8 * 1024 * 1024 )); then
  echo "Need at least 8 GiB free disk before launching" >&2
  exit 1
fi
mkdir -p "$ROOT/normal" "$ROOT/exact_g"
resume_flag=""
if [[ "$MODE" == resume ]]; then
  test -f "$ROOT/normal/state.json"
  test -f "$ROOT/exact_g/state.json"
  resume_flag=" --resume"
else
  if [[ -e "$ROOT/normal/manifest.json" || -e "$ROOT/exact_g/manifest.json" ]]; then
    echo "A run already exists; use resume" >&2
    exit 1
  fi
fi
if tmux has-session -t oens_normal 2>/dev/null || tmux has-session -t oens_exact 2>/dev/null; then
  echo "OENS tmux session already exists; inspect it before relaunching" >&2
  exit 1
fi
env_prefix="CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1"
tmux new-session -d -s oens_normal "cd /root/liars_poker && $env_prefix .venv/bin/python -u scripts/run_cfr_plus_18_oens_followups.py --mode normal --output-dir $ROOT/normal --training-minutes 360 --monitor-minutes 15 --threads 8 --seed 31$resume_flag >> $ROOT/normal/console.log 2>&1"
tmux new-session -d -s oens_exact "cd /root/liars_poker && $env_prefix .venv/bin/python -u scripts/run_cfr_plus_18_oens_followups.py --mode exact_g --source-checkpoint $SOURCE --output-dir $ROOT/exact_g --training-minutes 120 --monitor-minutes 15 --threads 8$resume_flag >> $ROOT/exact_g/console.log 2>&1"
if ! tmux has-session -t oens_dashboard 2>/dev/null; then
  tmux new-session -d -s oens_dashboard "cd /root/liars_poker && CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -u scripts/monitor_cfr_plus_18_oens.py $ROOT/normal --port 8767 >> $ROOT/dashboard.log 2>&1"
fi
echo "normal: $ROOT/normal"
echo "exact_g: $ROOT/exact_g"
echo "fresh audit dashboard: port 8767 (SSH tunnel)"
