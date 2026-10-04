#!/usr/bin/env bash
set -euo pipefail

REPO=/root/liars_poker
RUN_ROOT=/root/liars_poker/artifacts/cfr_plus_18_cumulative_regret/main_20260929
PYTHON="$REPO/.venv/bin/python"
THREADS=${THREADS:-8}
HOURS=10

cd "$REPO"
test -x "$PYTHON"
test -f "$RUN_ROOT/parallel_manifest.json"

"$PYTHON" - "$RUN_ROOT/extra_arms.json" <<'PY'
import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
arms = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
additions = [
    {
        "name": "hybrid_plain_mse4096",
        "label": "Hybrid: aggregate signed targets + clip on read · plain MSE",
        "traversals": 4096,
        "mode": "aggregate_then_clip_on_read",
        "seed": 17,
        "reach_mode": "none",
        "color": "#7B2CBF",
        "target_minutes": 600,
    },
    {
        "name": "aggregate_plain_mse4096",
        "label": "Aggregate then clip · plain MSE",
        "traversals": 4096,
        "mode": "aggregate_then_clip",
        "seed": 17,
        "reach_mode": "none",
        "color": "#EC4FA6",
        "target_minutes": 600,
    },
]
existing = {arm["name"] for arm in arms}
for arm in additions:
    if arm["name"] not in existing:
        arms.append(arm)
tmp = path.with_suffix(path.suffix + ".tmp")
tmp.write_text(json.dumps(arms, indent=2) + "\n", encoding="utf-8")
os.replace(tmp, path)
PY

start_arm() {
  local name="$1" mode="$2"
  local output="$RUN_ROOT/$name"
  local session="cfr18_${name}"
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "$session already exists; leaving it untouched"
    return
  fi
  if [ -e "$output/${mode}__seed_17" ]; then
    echo "Refusing to overwrite existing arm data: $output/${mode}__seed_17" >&2
    exit 1
  fi
  mkdir -p "$output"
  tmux new-session -d -s "$session" \
    "cd '$REPO' && CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS='$THREADS' MKL_NUM_THREADS='$THREADS' OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 '$PYTHON' -u scripts/run_cfr_plus_18_target_order_cpu_overnight.py --output-root '$output' --hours-per-arm '$HOURS' --traversals 4096 --seeds 17 --modes '$mode' --reach-mode none --regret-accumulation-mode cumulative --regret-positive-weight 0 --regret-buffer-capacity 4000000 --torch-threads '$THREADS' --snapshot-minutes 15 --checkpoint-minutes 15 >> '$output/train.log' 2>&1"
  echo "Started $session ($mode), $HOURS training hours; log: $output/train.log"
}

start_arm hybrid_plain_mse4096 aggregate_then_clip_on_read
start_arm aggregate_plain_mse4096 aggregate_then_clip

# The dashboard reads the additional-arm registry at startup. Restart only
# its monitor process; all training and evaluation records remain on disk.
if tmux has-session -t cfr18_dashboard 2>/dev/null; then
  tmux kill-session -t cfr18_dashboard
fi
tmux new-session -d -s cfr18_dashboard \
  "cd /root/liars_poker && PORT=8765 bash scripts/launch_cfr_plus_18_cumulative_dashboard.sh"
echo "Dashboard restarted with both new arms: http://127.0.0.1:8765"
