#!/usr/bin/env bash
set -euo pipefail

REPO=/root/liars_poker
PYTHON="$REPO/.venv/bin/python"
OUTPUT="$REPO/artifacts/cfr_plus_18_neural_o4_cpu/main_20261001"
mkdir -p "$OUTPUT"
test -x "$PYTHON"
test -f "$REPO/scripts/run_cfr_plus_18_neural_o4_cpu.py"
test -f "$REPO/scripts/evaluate_cfr_plus_18_batched_bridge_snapshot.py"

for arm in neural_o4_k1024 neural_o4_k4096; do
  mkdir -p "$OUTPUT/$arm"
  if ! tmux has-session -t "cfr18_${arm}_fit" 2>/dev/null; then
    tmux new-session -d -s "cfr18_${arm}_fit" \
      "cd '$REPO' && CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 '$PYTHON' -u scripts/run_cfr_plus_18_neural_o4_cpu.py fit --output-root '$OUTPUT' --arm '$arm' --threads 8 >> '$OUTPUT/$arm/fit.log' 2>&1"
  fi
  if ! tmux has-session -t "cfr18_${arm}_train" 2>/dev/null; then
    tmux new-session -d -s "cfr18_${arm}_train" \
      "cd '$REPO' && CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 '$PYTHON' -u scripts/run_cfr_plus_18_neural_o4_cpu.py train --output-root '$OUTPUT' --arm '$arm' --minutes 600 --snapshot-minutes 15 --threads 8 >> '$OUTPUT/$arm/train.log' 2>&1"
  fi
  echo "started or kept $arm train + fit"
done

if ! tmux has-session -t cfr18_neural_o4_eval 2>/dev/null; then
  tmux new-session -d -s cfr18_neural_o4_eval \
    "cd '$REPO' && CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 '$PYTHON' -u scripts/run_cfr_plus_18_neural_o4_cpu.py eval --output-root '$OUTPUT' >> '$OUTPUT/eval.log' 2>&1"
fi
echo "results: $OUTPUT"
