#!/usr/bin/env bash
set -euo pipefail
cd /root/liars_poker_20261001
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
OUT=/root/liars_poker/artifacts/cfr_plus_18_approx_br_calibration/main_20261003
mkdir -p "$OUT"
exec .venv/bin/python -u scripts/run_cfr_plus_18_approx_br_calibration.py \
  --output-root "$OUT" --workers 4
