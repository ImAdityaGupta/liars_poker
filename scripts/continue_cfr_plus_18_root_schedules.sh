#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/root/liars_poker_20261001}
PYTHON="$REPO/.venv/bin/python"
OUTPUT="$REPO/artifacts/cfr_plus_18_root_schedules/main_20261001"
ARMS=(k0256 k0512 k1024 k2048 k8192 k16384 ramp_up)

cd "$REPO"
test -x "$PYTHON"
free_kib=$(df -Pk "$OUTPUT" | awk 'NR==2 {print $4}')
if (( free_kib < 8 * 1024 * 1024 )); then
  echo "Need at least 8 GiB free before resuming the seven arms" >&2
  exit 1
fi

for arm in "${ARMS[@]}"; do
  test -s "$OUTPUT/$arm/latest_checkpoint.pt"
  test -s "$OUTPUT/$arm/manifest.json"
  if tmux has-session -t "cfr18_${arm}_train" 2>/dev/null; then
    echo "Refusing to duplicate active tmux session cfr18_${arm}_train" >&2
    exit 1
  fi
done

for arm in "${ARMS[@]}"; do
  mkdir -p "$OUTPUT/$arm"
  tmux new-session -d -s "cfr18_${arm}_train" \
    "cd '$REPO' && CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 '$PYTHON' -u scripts/run_cfr_plus_18_root_schedules.py --output-root '$OUTPUT' --arm '$arm' --hours 19 --extend-from-hours 9 --threads 8 >> '$OUTPUT/$arm/train.log' 2>&1"
done

echo "Continued seven arms to 19 total measured hours: $OUTPUT"
echo "This adds about 10 measured hours to each existing 9-hour checkpoint."
echo "The existing cfr18_root_eval queue is left in place to process new snapshots."
echo "Pause all resumed arms after they reach a durable save: touch '$OUTPUT/PAUSE'"
