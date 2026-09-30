#!/usr/bin/env bash
# Start/resume one from-scratch CPU clip-on-read control on the 8765 dashboard.
set -euo pipefail
cd "$(dirname "$0")/.."
RUN_ROOT=artifacts/cfr_plus_18_cumulative_regret/main_20260929
ARM_ROOT="$RUN_ROOT/clip_on_read4096"
SESSION=cfr18_clip_on_read_cpu
TARGET_HOURS=${TARGET_HOURS:-5.5}

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "$SESSION already running"
  exit 0
fi
mkdir -p "$ARM_ROOT"
".venv/bin/python" - "$RUN_ROOT" <<'PY'
import json
import os
from pathlib import Path
import sys

root = Path(sys.argv[1])
path = root / "extra_arms.json"
arm = {"name": "clip_on_read4096", "label": "Clip on read, plain MSE, 4,096 roots",
       "traversals": 4096, "mode": "clip_on_read", "seed": 17,
       "reach_mode": "none", "color": "#D7191C", "target_minutes": 330}
arms = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
if not any(existing["name"] == arm["name"] for existing in arms):
    arms.append(arm)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(arms, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)
PY

if [ -f "$ARM_ROOT/clip_on_read__seed_17/latest_checkpoint.pt" ]; then
  RESUME=--resume
else
  RESUME=
fi
tmux new-session -d -s "$SESSION" \
  "cd /root/liars_poker && CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 .venv/bin/python -u scripts/run_cfr_plus_18_target_order_cpu_overnight.py --output-root '$ARM_ROOT' --hours-per-arm '$TARGET_HOURS' --traversals 4096 --seeds 17 --modes clip_on_read --reach-mode none --regret-accumulation-mode cumulative --regret-buffer-capacity 4000000 --torch-threads 8 --snapshot-minutes 15 --checkpoint-minutes 15 $RESUME >> '$ARM_ROOT/train.log' 2>&1"
echo "Started $SESSION; log: $ARM_ROOT/train.log"
