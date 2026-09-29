#!/usr/bin/env bash
# Wait for the existing 120-minute tmux session, then extend selected arms.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOTS=${ROOTS:-"128 256 512"}
echo "Waiting for bridge_low_roots to finish; then extending K=$ROOTS to 180 measured minutes"
while tmux list-sessions -F '#S' 2>/dev/null | grep -Fxq bridge_low_roots; do
  sleep 20
done
export RESUME=1 MINUTES=180 ROOTS
bash scripts/launch_cfr_plus_18_tabular_bridge_low_roots.sh
