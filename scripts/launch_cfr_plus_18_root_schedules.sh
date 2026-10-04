#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/root/liars_poker}
PYTHON="$REPO/.venv/bin/python"
OUTPUT="$REPO/artifacts/cfr_plus_18_root_schedules/main_20261001"
mkdir -p "$OUTPUT"
test -x "$PYTHON"
if [[ "${SMOKE:-0}" == 1 ]]; then
  OUTPUT="$REPO/artifacts/cfr_plus_18_root_schedules/smoke_redesign_20261001"
  mkdir -p "$OUTPUT"
  CUDA_VISIBLE_DEVICES= "$PYTHON" -u "$REPO/scripts/run_cfr_plus_18_root_schedules.py" \
    --output-root "$OUTPUT" --arm k0512 --threads 8 --stop-after-iterations 2
  CUDA_VISIBLE_DEVICES= "$PYTHON" -u "$REPO/scripts/run_cfr_plus_18_root_schedules.py" \
    --output-root "$OUTPUT" --arm k0512 --threads 8 --stop-after-iterations 3
  echo "smoke resume: $OUTPUT/k0512/summary.json"
  exit 0
fi
free_kib=$(df -Pk "$OUTPUT" | awk 'NR==2 {print $4}')
if (( free_kib < 16 * 1024 * 1024 )); then
  echo "Need at least 16 GiB free before launching the root-count sweep" >&2
  exit 1
fi

ARMS=(k0256 k0512 k1024 k2048 k8192 k16384 ramp_up ramp_down step_late step_early)
for arm in "${ARMS[@]}"; do
  mkdir -p "$OUTPUT/$arm"
  if ! tmux has-session -t "cfr18_${arm}_train" 2>/dev/null; then
    tmux new-session -d -s "cfr18_${arm}_train" \
      "cd '$REPO' && CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 '$PYTHON' -u scripts/run_cfr_plus_18_root_schedules.py --output-root '$OUTPUT' --arm '$arm' --hours 9 --threads 8 >> '$OUTPUT/$arm/train.log' 2>&1"
  fi
done

if ! tmux has-session -t cfr18_root_eval 2>/dev/null; then
  tmux new-session -d -s cfr18_root_eval \
    "cd '$REPO' && CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 '$PYTHON' -u scripts/evaluate_cfr_plus_18_tabular_discount_queue.py --output-root '$OUTPUT' --evaluator scripts/evaluate_cfr_plus_18_batched_bridge_snapshot.py --workers 4 --arms k0256 k0512 k1024 k2048 k8192 k16384 ramp_up ramp_down step_late step_early --discard-policy-after-eval >> '$OUTPUT/eval.log' 2>&1"
fi
echo "results: $OUTPUT"
echo "pause: touch '$OUTPUT/PAUSE'"
