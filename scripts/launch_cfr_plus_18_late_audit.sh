#!/usr/bin/env bash
# Audit the final 330-minute neural checkpoints one seed at a time.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=${ROOT:-artifacts/cfr_plus_18_late_update_audit}
SOURCE=${SOURCE:-artifacts/cfr_plus_18_parallel_cpu/long_20260928}
SEEDS=${SEEDS:-"17 23"}
mkdir -p "$ROOT"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
for seed in $SEEDS; do
  checkpoint="$SOURCE/trav4096__aggregate_then_clip__seed${seed}/aggregate_then_clip__seed_${seed}/latest_checkpoint.pt"
  out="$ROOT/seed${seed}"
  mkdir -p "$out"
  echo "auditing seed=$seed checkpoint=$checkpoint" | tee -a "$ROOT/launcher.log"
  .venv/bin/python -u scripts/audit_cfr_plus_18_late_update.py \
    --checkpoint "$checkpoint" --output-dir "$out" \
    --player 0 --roots 4096 --threads 2 \
    >> "$out/console.log" 2>&1
  echo "completed seed=$seed" | tee -a "$ROOT/launcher.log"
done
