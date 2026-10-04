#!/usr/bin/env bash
set -euo pipefail

cd /root/liars_poker_20261001
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

artifacts=/root/liars_poker/artifacts
run_root="$artifacts/cfr_plus_18_regret_table_distillation/main_20261002"
for source in 0030m 1080m; do
  log="$run_root/$source/P-visit/gpu.log"
  echo "[start] $source $(date -u +%FT%TZ)" | tee -a "$log"
  .venv/bin/python -u scripts/run_cfr_plus_18_regret_table_distillation.py \
    --artifacts "$artifacts" --source "$source" --arm P-visit --device cuda --threads 2 \
    2>&1 | tee -a "$log"
  echo "[complete] $source $(date -u +%FT%TZ)" | tee -a "$log"
done
