#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/root/liars_poker}
PYTHON="$REPO/.venv/bin/python"
OUTPUT="$REPO/artifacts/cfr_plus_18_regret_noise/main_20261001"
mkdir -p "$OUTPUT"
test -x "$PYTHON"
test -f "$REPO/scripts/run_cfr_plus_18_neural_o4_cpu.py"

if [[ "${BENCHMARK:-0}" == 1 ]]; then
  OUTPUT="$REPO/artifacts/cfr_plus_18_regret_noise/standalone_benchmark_20261001"
  mkdir -p "$OUTPUT"
  for arm in c0 c_batch c_anneal c_low; do
    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
      "$PYTHON" -u "$REPO/scripts/run_cfr_plus_18_neural_o4_cpu.py" train \
      --output-root "$OUTPUT" --arm "$arm" --minutes 0.5 \
      --snapshot-minutes 60 --threads 2 | tee "$OUTPUT/${arm}.log"
  done
  echo "standalone timings: $OUTPUT/*/training.jsonl"
  exit 0
fi

# Set SMOKE=1 for a short, separate end-to-end rehearsal.
if [[ "${SMOKE:-0}" == 1 ]]; then
  TAG=cfr18_smoke
  OUTPUT="$REPO/artifacts/cfr_plus_18_regret_noise/smoke_20261001"
  MINUTES=0.2
  SNAPSHOT=0.1
  FIT_STEPS=100
else
  TAG=cfr18_regret
  MINUTES=480
  SNAPSHOT=30
  FIT_STEPS=5000
fi
mkdir -p "$OUTPUT"

for arm in c0 c_batch c_anneal c_low; do
  mkdir -p "$OUTPUT/$arm"
  if ! tmux has-session -t "${TAG}_${arm}_train" 2>/dev/null; then
    tmux new-session -d -s "${TAG}_${arm}_train" \
      "cd '$REPO' && OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 '$PYTHON' -u scripts/run_cfr_plus_18_neural_o4_cpu.py train --output-root '$OUTPUT' --arm '$arm' --minutes '$MINUTES' --snapshot-minutes '$SNAPSHOT' --threads 2 >> '$OUTPUT/$arm/train.log' 2>&1"
  fi
done

if ! tmux has-session -t "${TAG}_fit" 2>/dev/null; then
  tmux new-session -d -s "${TAG}_fit" \
    "cd '$REPO' && OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 '$PYTHON' -u scripts/run_cfr_plus_18_neural_o4_cpu.py fit --output-root '$OUTPUT' --fit-steps '$FIT_STEPS' --fit-batch-size 16384 --threads 2 >> '$OUTPUT/fit.log' 2>&1"
fi
if ! tmux has-session -t "${TAG}_eval" 2>/dev/null; then
  tmux new-session -d -s "${TAG}_eval" \
    "cd '$REPO' && CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 '$PYTHON' -u scripts/run_cfr_plus_18_neural_o4_cpu.py eval --output-root '$OUTPUT' >> '$OUTPUT/eval.log' 2>&1"
fi
echo "results: $OUTPUT"
echo "pause: touch '$OUTPUT/PAUSE'"
