#!/usr/bin/env bash
set -euo pipefail
cd /root/liars_poker_discount
run_root=/root/liars_poker/artifacts/cfr_plus_18_tabular_discount/main_20260930
export CUDA_VISIBLE_DEVICES=
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec .venv/bin/python -u scripts/monitor_cfr_plus_18_tabular_discount.py \
  --output-root "$run_root" \
  --bridge-root /root/liars_poker/artifacts/cfr_plus_18_tabular_bridge/main_20260929 \
  --controls-root /root/liars_poker/artifacts/cfr_plus_18_batched_bridge_controls/main_20260930 \
  --port 8768 >> "$run_root/dashboard.log" 2>&1
