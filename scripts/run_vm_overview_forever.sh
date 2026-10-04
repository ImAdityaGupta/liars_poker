#!/usr/bin/env bash
set -u
REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
cd "$REPO"
ARTIFACT_ARGS=(--artifacts "$REPO/artifacts")
if [ -d /root/liars_poker_20261001/artifacts ]; then
  ARTIFACT_ARGS+=(/root/liars_poker_20261001/artifacts)
fi
while true; do
  "$REPO/.venv/bin/python" -u scripts/monitor_vast_vm.py \
    "${ARTIFACT_ARGS[@]}" \
    --state-dir "$REPO/artifacts/vm_overview" \
    --port 8765
  status=$?
  echo "VM overview monitor exited ($status); restarting in 5s" >&2
  sleep 5
done
