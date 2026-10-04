#!/usr/bin/env python3
"""Evaluate committed discount-arm policy snapshots in separate CPU processes."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_cfr_plus_18_tabular_discount import ARMS, append_jsonl


def completed(run_dir: Path) -> set[str]:
    path = run_dir / "evaluations.jsonl"
    if not path.exists():
        return set()
    names = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            names.add(json.loads(line)["snapshot"])
        except (json.JSONDecodeError, KeyError):
            continue
    return names


def evaluate(run_dir: Path, ready: dict, evaluator: Path) -> dict:
    env = os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1"})
    result = subprocess.run(
        [sys.executable, str(evaluator), ready["policy_dir"]],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=600, check=True,
    )
    score = json.loads(result.stdout.splitlines()[-1])
    return {
        "utc": datetime.now(timezone.utc).isoformat(),
        "arm": run_dir.name, "snapshot": ready["snapshot"],
        "iteration": ready["iteration"],
        "measured_training_min": ready["measured_training_min"],
        "cumulative_roots_per_player": ready.get("cumulative_roots_per_player"),
        "policy_dir": ready["policy_dir"],
        "p_first": score["p_first"], "p_second": score["p_second"],
        "exploitability": score["exploitability"],
        "evaluation_s": score["evaluation_s"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--evaluator", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--arms", nargs="+", default=list(ARMS))
    parser.add_argument("--discard-policy-after-eval", action="store_true",
                        help="Remove large dense policy directory after its row is durable")
    args = parser.parse_args()
    if args.workers <= 0 or not args.evaluator.exists():
        parser.error("Workers must be positive and evaluator must exist")
    root = args.output_root.resolve()
    active = {}
    retry_after = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        while not (root / "STOP_EVALUATOR").exists():
            for key, future in list(active.items()):
                if not future.done():
                    continue
                run_dir, label = key
                del active[key]
                try:
                    row = future.result()
                except Exception as exc:
                    retry_after[key] = time.monotonic() + 60
                    print(f"[eval failed] {run_dir.name} {label}: {exc}", flush=True)
                    continue
                append_jsonl(run_dir / "evaluations.jsonl", row)
                if args.discard_policy_after_eval:
                    policy_dir = Path(row["policy_dir"]).resolve()
                    expected_parent = (run_dir / "policy_snapshots" / label).resolve()
                    if policy_dir.parent != expected_parent or policy_dir.name != "average_policy":
                        raise ValueError(f"Refusing to remove unexpected policy path: {policy_dir}")
                    shutil.rmtree(policy_dir)
                print(f"[eval] {run_dir.name} {label} iter={row['iteration']} "+
                      f"exploitability={row['exploitability']:.6f} "+
                      f"seconds={row['evaluation_s']:.1f}", flush=True)

            for arm in args.arms:
                run_dir = root / arm
                done = completed(run_dir)
                for ready_path in sorted((run_dir / "policy_snapshots").glob("*/READY.json")):
                    ready = json.loads(ready_path.read_text(encoding="utf-8"))
                    key = (run_dir, ready["snapshot"])
                    if (ready["snapshot"] in done or key in active
                            or time.monotonic() < retry_after.get(key, 0)):
                        continue
                    active[key] = pool.submit(evaluate, run_dir, ready, args.evaluator)
            time.sleep(10)


if __name__ == "__main__":
    main()
