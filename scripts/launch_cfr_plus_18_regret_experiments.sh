#!/usr/bin/env bash
set -euo pipefail
cd /root/liars_poker_20261001
ART=/root/liars_poker/artifacts
DIST="$ART/cfr_plus_18_regret_table_distillation/main_20261002"
NT="$ART/cfr_plus_18_regret_bootstrap_teacher_forced/main_20261002"
mkdir -p "$DIST" "$NT"
for source in 0030m 0045m 0120m 1080m; do
  tmux new-session -d -s "regdist_${source}" \
    "cd /root/liars_poker_20261001 && CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 .venv/bin/python -u scripts/run_cfr_plus_18_regret_table_distillation.py --artifacts /root/liars_poker/artifacts --output-root $DIST --source $source --threads 8 > $DIST/$source.log 2>&1"
done
for arm in N T; do
  tmux new-session -d -s "regnt_${arm}" \
    "cd /root/liars_poker_20261001 && OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 .venv/bin/python -u scripts/run_cfr_plus_18_regret_bootstrap_teacher_forced.py --output-root $NT --arm $arm --snapshot-every 250 --iterations 24000 --threads 2 > $NT/$arm.log 2>&1"
done
tmux new-session -d -s regret_diagnostics_20261002 \
  "cd /root/liars_poker_20261001 && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -u scripts/monitor_cfr_plus_18_regret_diagnostics.py --artifacts /root/liars_poker/artifacts --state-dir $ART/cfr_plus_18_regret_dashboard --port 8770"
tmux ls | grep -E 'regdist_|regnt_|regret_diagnostics'
