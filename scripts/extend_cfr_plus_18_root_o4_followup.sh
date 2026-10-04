#!/usr/bin/env bash
# Queue a checkpoint handoff for each existing Part B O4 arm.
set -euo pipefail

repo=/root/liars_poker_20261001
python="$repo/.venv/bin/python"
output=/root/liars_poker/artifacts/cfr_plus_18_root_o4_followup/main_20261003
arms=(k1024 k4096 k16384 k32768 ramp ramp8m ramp_exact)

for arm in "${arms[@]}"; do
  test -s "$output/$arm/manifest.json"
  test -s "$output/$arm/latest_checkpoint.pt"
done

echo "[extension] queued seven arms for 600 -> 1200 measured minutes $(date -u +%FT%TZ)" | tee -a "$output/extension.log"
declare -A handed_off=()
remaining=${#arms[@]}
while ((remaining > 0)); do
  for arm in "${arms[@]}"; do
    [[ ${handed_off[$arm]+present} ]] && continue
    session="cfr18_root_o4_${arm}"
    tmux has-session -t "$session" 2>/dev/null && continue
    if "$python" - "$output/$arm/summary.json" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
if not p.exists():
    raise SystemExit(1)
s = json.loads(p.read_text())
raise SystemExit(0 if s.get("status") == "complete" and s.get("measured_training_min", 0) >= 600 else 1)
PY
    then
      if "$python" - "$output/$arm/summary.json" <<'PY'
import json, sys
from pathlib import Path
s = json.loads(Path(sys.argv[1]).read_text())
raise SystemExit(0 if s.get("measured_training_min", 0) >= 1200 else 1)
PY
      then
        echo "[extension] $arm already reached 1200m" | tee -a "$output/extension.log"
      else
        echo "[extension] resuming $arm at $(date -u +%FT%TZ)" | tee -a "$output/extension.log"
        tmux new-session -d -s "$session" \
          "cd '$repo' && CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 '$python' -u scripts/run_cfr_plus_18_root_o4_followup.py train --output-root '$output' --arm '$arm' --hours 20 --extend-from-hours 10 --threads 8 >> '$output/$arm/train.log' 2>&1"
      fi
      handed_off[$arm]=1
      remaining=$((remaining - 1))
    else
      echo "[extension] $arm session ended without a completed 600m summary" | tee -a "$output/extension.log" >&2
      exit 1
    fi
  done
  ((remaining == 0)) || sleep 10
done
echo "[extension] all seven handoffs launched $(date -u +%FT%TZ)" | tee -a "$output/extension.log"
