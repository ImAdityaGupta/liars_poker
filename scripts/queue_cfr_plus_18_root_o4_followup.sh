#!/usr/bin/env bash
set -euo pipefail

repo=/root/liars_poker_20261001
python="$repo/.venv/bin/python"
artifacts=/root/liars_poker/artifacts
output="$artifacts/cfr_plus_18_root_o4_followup/main_20261003"
pvisit="$artifacts/cfr_plus_18_regret_table_distillation/main_20261002"
mkdir -p "$output"

echo "[queue] waiting for both P-visit results and GPU session exit" | tee -a "$output/queue.log"
while tmux has-session -t cfr18_pvisit_gpu 2>/dev/null; do
  sleep 30
done
for source in 0030m 1080m; do
  if [[ ! -s "$pvisit/$source/P-visit/result.json" ]]; then
    echo "[queue] P-visit $source did not finish; follow-up not started" | tee -a "$output/queue.log" >&2
    exit 1
  fi
done

free_kib=$(df -Pk "$output" | awk 'NR==2 {print $4}')
shm_kib=$(df -Pk /dev/shm | awk 'NR==2 {print $4}')
if (( free_kib < 11 * 1024 * 1024 || shm_kib < 8 * 1024 * 1024 )); then
  echo "[queue] insufficient free disk or shared memory; follow-up not started" | tee -a "$output/queue.log" >&2
  exit 1
fi
test -x "$python"
test -f "$repo/scripts/run_cfr_plus_18_root_o4_followup.py"
echo "[queue] starting 7 CPU trainers, 1 GPU fitter, 1 CPU evaluator $(date -u +%FT%TZ)" | tee -a "$output/queue.log"

if ! tmux has-session -t cfr18_root_o4_fit 2>/dev/null; then
  tmux new-session -d -s cfr18_root_o4_fit \
    "cd '$repo' && OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 '$python' -u scripts/run_cfr_plus_18_root_o4_followup.py fit --output-root '$output' --threads 2 >> '$output/fit.log' 2>&1"
fi
if ! tmux has-session -t cfr18_root_o4_eval 2>/dev/null; then
  tmux new-session -d -s cfr18_root_o4_eval \
    "cd '$repo' && CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 '$python' -u scripts/run_cfr_plus_18_root_o4_followup.py eval --output-root '$output' --threads 1 >> '$output/eval.log' 2>&1"
fi

for arm in k1024 k4096 k16384 k32768 ramp ramp8m ramp_exact; do
  mkdir -p "$output/$arm"
  session="cfr18_root_o4_${arm}"
  if ! tmux has-session -t "$session" 2>/dev/null; then
    tmux new-session -d -s "$session" \
      "cd '$repo' && CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 '$python' -u scripts/run_cfr_plus_18_root_o4_followup.py train --output-root '$output' --arm '$arm' --hours 10 --threads 8 >> '$output/$arm/train.log' 2>&1"
  fi
done
echo "[queue] launched; output=$output" | tee -a "$output/queue.log"
