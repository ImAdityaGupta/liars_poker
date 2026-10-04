#!/usr/bin/env bash
set -euo pipefail
REPO=${REPO:-/root/liars_poker_20261001}
PORT=${PORT:-8769}
cd "$REPO"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec "$REPO/.venv/bin/python" -u scripts/monitor_cfr_plus_18_tabular_discount.py \
  --output-root /root/liars_poker/artifacts/cfr_plus_18_tabular_discount/main_20260930 \
  --bridge-root /root/liars_poker/artifacts/cfr_plus_18_tabular_bridge/main_20260929 \
  --controls-root /root/liars_poker/artifacts/cfr_plus_18_batched_bridge_controls/main_20260930 \
  --neural-root /root/liars_poker/artifacts/cfr_plus_18_neural_o4_cpu/main_20261001 \
  --regret-root /root/liars_poker/artifacts/cfr_plus_18_regret_noise/main_20261001 \
  --schedule-root /root/liars_poker/artifacts/cfr_plus_18_root_schedules/main_20261001 \
  --exact-discount-root /root/liars_poker/artifacts/cfr_plus_18_tabular_discount_exact_average/main_20261002 \
  --port "$PORT" >> /root/liars_poker/artifacts/cfr_plus_18_tabular_discount/main_20260930/dashboard_20261001.log 2>&1
