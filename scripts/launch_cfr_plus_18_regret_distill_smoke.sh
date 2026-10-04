#!/usr/bin/env bash
set -euo pipefail
cd /root/liars_poker_20261001
OUT=/root/liars_poker/artifacts/cfr_plus_18_regret_table_distillation/smoke4_20261002
mkdir -p "$OUT"
tmux new-session -d -s regdist_smoke4 "cd /root/liars_poker_20261001 && CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -u scripts/run_cfr_plus_18_regret_table_distillation.py --artifacts /root/liars_poker/artifacts --output-root $OUT --source 0030m --steps 8 --threads 2 --smoke > $OUT/smoke.log 2>&1"
