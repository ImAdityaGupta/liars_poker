#!/usr/bin/env bash
set -euo pipefail
cd /root/liars_poker_20261001
BASE=/root/liars_poker/artifacts
DIST="$BASE/cfr_plus_18_regret_table_distillation/smoke2_20261002"
NT="$BASE/cfr_plus_18_regret_bootstrap_teacher_forced/smoke3_20261002"
mkdir -p "$DIST" "$NT"
tmux kill-session -t regntN_smoke 2>/dev/null || true
tmux kill-session -t regntT_smoke 2>/dev/null || true
tmux new-session -d -s regdist_smoke2 "cd /root/liars_poker_20261001 && OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -u scripts/run_cfr_plus_18_regret_table_distillation.py --artifacts /root/liars_poker/artifacts --output-root $DIST --source 0030m --steps 8 --threads 2 --smoke > $DIST/smoke.log 2>&1"
tmux new-session -d -s regntN_smoke3 "cd /root/liars_poker_20261001 && OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -u scripts/run_cfr_plus_18_regret_bootstrap_teacher_forced.py --output-root $NT --arm N --snapshot-every 50 --threads 2 --smoke > $NT/N.log 2>&1"
tmux new-session -d -s regntT_smoke3 "cd /root/liars_poker_20261001 && OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -u scripts/run_cfr_plus_18_regret_bootstrap_teacher_forced.py --output-root $NT --arm T --snapshot-every 50 --threads 2 --smoke > $NT/T.log 2>&1"
tmux ls | grep -E 'reg(dist|nt)'
