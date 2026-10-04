#!/usr/bin/env bash
set -euo pipefail
cd /root/liars_poker_20261001
OUT=/root/liars_poker/artifacts/cfr_plus_30_claim_first_run/regression18_20261003
mkdir -p "$OUT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec .venv/bin/python -u scripts/smoke_cfr_plus_18_regression_for_30.py \
  --output-root "$OUT" --iterations 1000 --fit-steps 5000 \
  >> "$OUT/run.log" 2>&1
