#!/usr/bin/env bash
set -euo pipefail
cd /root/liars_poker_20261001
POLICY=/root/liars_poker/artifacts/cfr_plus_18_neural_o4_cpu/main_20261001/neural_o4_k4096/policy_snapshots/0015m/average_policy
OUT=/root/liars_poker/artifacts/cfr_plus_30_claim_first_run/mc_check18_20261003
mkdir -p "$OUT"
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
.venv/bin/python -u scripts/evaluate_cfr_plus_30_claim_first_run.py one \
  --arm check18 --snapshot 0015m --kind o4 --depth 2 --policy "$POLICY" \
  --output "$OUT/exact.json" --exact >> "$OUT/run.log" 2>&1
.venv/bin/python -u scripts/evaluate_cfr_plus_30_claim_first_run.py one \
  --arm check18 --snapshot 0015m --kind o4 --depth 2 --policy "$POLICY" \
  --output "$OUT/mc.json" --episodes 3000 >> "$OUT/run.log" 2>&1
