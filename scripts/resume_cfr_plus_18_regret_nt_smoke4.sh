#!/usr/bin/env bash
set -euo pipefail
cd /root/liars_poker_20261001
NT=/root/liars_poker/artifacts/cfr_plus_18_regret_bootstrap_teacher_forced/smoke4_20261002
tmux new-session -d -s regntN_smoke8 "cd /root/liars_poker_20261001 && OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -u scripts/run_cfr_plus_18_regret_bootstrap_teacher_forced.py --output-root $NT --arm N --snapshot-every 50 --threads 2 --smoke > $NT/N_resume4.log 2>&1"
tmux new-session -d -s regntT_smoke8 "cd /root/liars_poker_20261001 && OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -u scripts/run_cfr_plus_18_regret_bootstrap_teacher_forced.py --output-root $NT --arm T --snapshot-every 50 --threads 2 --smoke > $NT/T_resume4.log 2>&1"
