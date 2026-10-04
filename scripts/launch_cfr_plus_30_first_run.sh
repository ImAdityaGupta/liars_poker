#!/usr/bin/env bash
set -euo pipefail
cd /root/liars_poker_20261001
ROOT="${1:-/root/liars_poker/artifacts/cfr_plus_30_claim_first_run/main_20261003}"
MINUTES="${2:-1440}"
SNAPSHOT="${3:-60}"
STEPS="${4:-5000}"
BATCH="${5:-16384}"
mkdir -p "$ROOT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
.venv/bin/python -u scripts/run_cfr_plus_30_claim_first_run.py controller \
  --output-root "$ROOT" --minutes "$MINUTES" --snapshot-minutes "$SNAPSHOT" \
  --fit-steps "$STEPS" --fit-batch "$BATCH" >> "$ROOT/controller.log" 2>&1
if [[ -f "$ROOT/ALL_DONE" && "$MINUTES" == 1440 ]]; then
  .venv/bin/python -u scripts/run_cfr_plus_30_final_brs.py \
    --output-root "$ROOT" >> "$ROOT/final_br.log" 2>&1
fi
