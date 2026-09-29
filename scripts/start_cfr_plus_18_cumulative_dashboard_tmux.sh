#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if tmux has-session -t cfr18_dashboard 2>/dev/null; then
  echo "cfr18_dashboard already running"
  exit 0
fi
tmux new-session -d -s cfr18_dashboard \
  'cd /root/liars_poker && bash scripts/launch_cfr_plus_18_cumulative_dashboard.sh'
echo 'cfr18_dashboard started on port 8765'
