#!/usr/bin/env bash
set -euo pipefail
REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
cd "$REPO"

# Stop only recognized dashboard listeners on old experiment ports.
# Port 8769 is intentionally preserved. Training/evaluation jobs are untouched.
for port in 8765 8766 8767 8768; do
  pids=$(ss -ltnp "sport = :$port" 2>/dev/null | sed -nE 's/.*pid=([0-9]+).*/\1/p' | sort -u)
  for pid in $pids; do
    [ -r "/proc/$pid/cmdline" ] || continue
    command=$(tr '\0' ' ' < "/proc/$pid/cmdline")
    case "$command" in
      *monitor_cfr_plus_18*|*monitor_cfr_plus*)
        echo "Stopping dashboard listener on $port (pid $pid)"
        kill -TERM "$pid"
        ;;
      *)
        echo "Port $port is held by a non-experiment-monitor process (pid $pid); leaving it alone." >&2
        exit 1
        ;;
    esac
  done
done

if tmux has-session -t vm_overview 2>/dev/null; then
  tmux kill-session -t vm_overview
fi
tmux new-session -d -s vm_overview "REPO='$REPO' bash '$REPO/scripts/run_vm_overview_forever.sh'"
echo "VM overview monitor started in tmux session vm_overview on port 8765."
echo "The experiment-specific 8769 monitor was left running."
