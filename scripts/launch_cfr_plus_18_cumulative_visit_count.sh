#!/usr/bin/env bash
# Run cumulative conditional neural CFR+ with N times the mean sampled increment.
set -euo pipefail
cd "$(dirname "$0")/.."

RUN_ROOT=${RUN_ROOT:-artifacts/cfr_plus_18_cumulative_regret/main_20260929}
TARGET_HOURS=${TARGET_HOURS:-10}
SESSION=cfr18_n4096
ARM_ROOT="$RUN_ROOT/n4096"

test -x .venv/bin/python
test -f "$RUN_ROOT/parallel_manifest.json"
command -v tmux >/dev/null
if tmux has-session -t "=$SESSION" 2>/dev/null; then
  echo "$SESSION is already running" >&2
  exit 1
fi
free_kib=$(df -Pk . | awk 'NR==2 {print $4}')
if (( free_kib < 8 * 1024 * 1024 )); then
  echo "Need at least 8 GiB free disk" >&2
  exit 1
fi

mkdir -p "$ARM_ROOT"
resume_flag=""
if [[ -f "$ARM_ROOT/aggregate_then_clip__seed_17/latest_checkpoint.pt" ]]; then
  resume_flag=" --resume"
elif [[ -f "$ARM_ROOT/manifest.json" ]]; then
  echo "Existing arm manifest has no checkpoint; inspect it before restarting" >&2
  exit 1
fi

# The dashboard reads its arm list at startup. Preserve the original arms and
# their recorded source hashes; this arm carries its own source provenance.
.venv/bin/python - "$RUN_ROOT" "$TARGET_HOURS" <<'PY'
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import sys

root = Path(sys.argv[1])
target_minutes = float(sys.argv[2]) * 60.0
if target_minutes <= 0:
    raise ValueError("TARGET_HOURS must be positive")
path = root / "parallel_manifest.json"
manifest = json.loads(path.read_text(encoding="utf-8"))
arms = manifest["arms"]
existing = next((arm for arm in arms if arm["name"] == "n4096"), None)
if existing is None:
    sources = ("liars_poker/algo/deep_cfr_plus.py",
               "liars_poker/algo/neural_cfr_plus_gpu.py",
               "scripts/run_cfr_plus_18_target_order_cpu_overnight.py")
    arms.append({
        "name": "n4096",
        "label": "Cumulative + N · 4,096 roots",
        "traversals": 4096,
        "reach_mode": "visit_count",
        "regret_buffer_capacity": 4_000_000,
        "color": "#3157A4",
        "mode": "aggregate_then_clip",
        "seed": 17,
        "target_minutes": target_minutes,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": {name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                          for name in sources},
    })
else:
    expected = (existing["traversals"], existing["reach_mode"],
                existing["mode"], existing["seed"])
    if expected != (4096, "visit_count", "aggregate_then_clip", 17):
        raise ValueError(f"Dashboard arm n4096 has unexpected settings: {expected}")
    existing["target_minutes"] = max(float(existing["target_minutes"]), target_minutes)
tmp = path.with_name(path.name + ".tmp")
tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
os.replace(tmp, path)
PY

env_prefix="CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1"
tmux new-session -d -s "$SESSION" "cd /root/liars_poker && $env_prefix .venv/bin/python -u scripts/run_cfr_plus_18_target_order_cpu_overnight.py --output-root $ARM_ROOT --hours-per-arm $TARGET_HOURS --traversals 4096 --seeds 17 --modes aggregate_then_clip --reach-mode visit_count --regret-accumulation-mode cumulative --regret-buffer-capacity 4000000 --torch-threads 8 --snapshot-minutes 15 --checkpoint-minutes 15$resume_flag >> $ARM_ROOT/train.log 2>&1"

if tmux has-session -t =cfr18_dashboard 2>/dev/null; then
  tmux kill-session -t =cfr18_dashboard
fi
tmux new-session -d -s cfr18_dashboard "cd /root/liars_poker && bash scripts/launch_cfr_plus_18_cumulative_dashboard.sh"
echo "started $SESSION; dashboard port 8765; target $TARGET_HOURS training hours"
