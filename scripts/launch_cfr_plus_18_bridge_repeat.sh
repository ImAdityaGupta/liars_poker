#!/usr/bin/env bash
set -euo pipefail
cd /root/liars_poker
run_root=/root/liars_poker/artifacts/cfr_plus_18_tabular_bridge/repeat_conditional_20260930
mkdir -p "$run_root/conditional"
export CUDA_VISIBLE_DEVICES=
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
exec .venv/bin/python -u scripts/run_cfr_plus_18_tabular_bridge.py \
  --arm conditional --output-root "$run_root" --seed 17 --roots 1024 \
  --minutes 540 --eval-minutes 15 --checkpoint-minutes 30 \
  >> "$run_root/conditional/console.log" 2>&1
