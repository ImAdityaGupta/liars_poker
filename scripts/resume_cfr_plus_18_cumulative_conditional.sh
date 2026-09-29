#!/usr/bin/env bash
# Resume only the cumulative conditional arm. Override TARGET_HOURS to extend it.
set -euo pipefail
cd "$(dirname "$0")/.."
RUN_ROOT=${RUN_ROOT:-artifacts/cfr_plus_18_cumulative_regret/main_20260929/conditional4096}
TARGET_HOURS=${TARGET_HOURS:-15.5}
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
exec .venv/bin/python -u scripts/run_cfr_plus_18_target_order_cpu_overnight.py \
  --output-root "$RUN_ROOT" --hours-per-arm "$TARGET_HOURS" --traversals 4096 \
  --seeds 17 --modes aggregate_then_clip --reach-mode none \
  --regret-accumulation-mode cumulative --regret-buffer-capacity 4000000 \
  --torch-threads 8 --snapshot-minutes 15 --checkpoint-minutes 15 --resume \
  >> "$RUN_ROOT/resume.log" 2>&1
