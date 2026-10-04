#!/usr/bin/env python3
"""Consume milestone regret audits without blocking GPU training."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("c0", "c_batch", "c_anneal", "c_low")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    root = args.output_root.resolve()
    evaluator = ROOT / "scripts/audit_cfr_plus_18_late_update.py"
    while not (root / "STOP_AUDITOR").exists():
        found = False
        for arm in ARMS:
            for input_path in sorted((root / arm / "audits").glob("*/input.pt")):
                found = True
                directory = input_path.parent
                try:
                    for pid in (0, 1):
                        result_dir = directory / f"player{pid + 1}"
                        if (result_dir / "summary.json").exists():
                            continue
                        env = os.environ.copy()
                        env.update({"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2",
                                    "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "1"})
                        subprocess.run([sys.executable, str(evaluator),
                                        "--checkpoint", str(input_path),
                                        "--output-dir", str(result_dir),
                                        "--player", str(pid), "--roots", "4096",
                                        "--threads", "2", "--held-out"],
                                       cwd=ROOT, env=env, check=True)
                    (directory / "DONE.json").write_text(json.dumps(
                        {"arm": arm, "milestone": directory.name,
                         "source_iteration": json.loads(
                             (directory / "player1" / "summary.json").read_text()
                         )["source_iteration"]}, indent=2), encoding="utf-8")
                    input_path.unlink()
                    print(f"[audit done] {arm} {directory.name}", flush=True)
                except Exception as exc:
                    print(f"[audit retry] {arm} {directory.name}: {exc}", flush=True)
                    if args.once:
                        raise
                    time.sleep(60)
        if args.once:
            return
        if not found:
            time.sleep(20)


if __name__ == "__main__":
    main()
