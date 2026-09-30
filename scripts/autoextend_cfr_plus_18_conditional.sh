#!/usr/bin/env bash
# Continue the existing CPU S24 baseline after its current 15.5-hour target.
set -euo pipefail
cd "$(dirname "$0")/.."
RUN=artifacts/cfr_plus_18_cumulative_regret/main_20260929/conditional4096
while ps -p 40918 -o args= 2>/dev/null | grep -Fq -- 'run_cfr_plus_18_target_order_cpu_overnight.py --output-root artifacts/cfr_plus_18_cumulative_regret/main_20260929/conditional4096'; do
  sleep 30
done
export TARGET_HOURS=20
exec /bin/bash scripts/resume_cfr_plus_18_cumulative_conditional.sh >> "$RUN/autoextend.log" 2>&1
