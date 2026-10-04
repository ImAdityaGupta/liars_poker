#!/usr/bin/env bash
set -euo pipefail
cd /root/liars_poker_20261001
OUT=/root/liars_poker/artifacts/cfr_plus_18_approx_br_calibration/main_20261003
exec .venv/bin/python -u scripts/monitor_cfr_plus_18_approx_br_calibration.py \
  --output-root "$OUT" --port 8772
