#!/usr/bin/env python3
"""Evaluate one saved exact-average tabular-discount snapshot."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import fcntl
except ImportError:  # pragma: no cover - VM evaluator runs on Linux
    fcntl = None

from liars_poker.algo.br_exact_dense_to_dense import best_response_dense
from liars_poker.policies.tabular_dense import DenseTabularPolicy
from liars_poker.serialization import load_policy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--policy-dir", type=Path, required=True)
    parser.add_argument("--snapshot", required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    eval_path = run_dir / "evaluations.jsonl"
    lock_path = run_dir / ".evaluation.lock"
    lock_path.touch(exist_ok=True)
    with lock_path.open("r+") as lock:
        if fcntl is not None:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        existing = []
        if eval_path.exists():
            for line in eval_path.read_text(encoding="utf-8").splitlines():
                try:
                    existing.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        if any(row.get("snapshot") == args.snapshot for row in existing):
            return

        policy, spec = load_policy(str(args.policy_dir))
        if not isinstance(policy, DenseTabularPolicy):
            raise TypeError(f"Expected DenseTabularPolicy; got {type(policy).__name__}")
        _, meta = best_response_dense(spec, policy, store_state_values=False)
        p_first, p_second = meta["computer"].exploitability()
        ready_path = args.policy_dir.parent / "READY.json"
        ready = json.loads(ready_path.read_text(encoding="utf-8"))
        row = {
            **ready,
            "utc": datetime.now(timezone.utc).isoformat(),
            "arm": run_dir.name,
            "snapshot": args.snapshot,
            "p_first": float(p_first),
            "p_second": float(p_second),
            "exploitability": float(p_first + p_second - 1),
            "policy_kind": "exact_average",
        }
        with eval_path.open("a", encoding="utf-8") as out:
            out.write(json.dumps(row) + "\n")
            out.flush()
        print(f"[eval] {run_dir.name} {args.snapshot} "
              f"exploitability={row['exploitability']:.7f}", flush=True)


if __name__ == "__main__":
    main()
