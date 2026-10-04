#!/usr/bin/env python3
"""Profile the precise 30-claim depth-limited scorer on one policy: time, network calls, cache use."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import types  # noqa: E402

sys.modules.setdefault("fcntl", types.ModuleType("fcntl"))  # the queue's file lock is Unix-only; unused here
import torch  # noqa: E402

from evaluate_cfr_plus_30_precise import evaluate_seat  # noqa: E402
from liars_poker.algo.br_limited_dense import LimitedBestResponse  # noqa: E402
from liars_poker.serialization import load_policy  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--sweeps", type=int, default=1)
    parser.add_argument("--solver", default="original", choices=("original", "batched"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(1)
    policy, _ = load_policy(args.policy)
    if args.device != "cpu":
        policy.model_p1.to(args.device)
        policy.model_p2.to(args.device)
        policy.device = torch.device(args.device)
    if args.solver == "batched":
        from liars_poker.algo.br_limited_batched import BatchedLimitedBestResponse
        solver = BatchedLimitedBestResponse(policy, depth=args.depth, epsilon=1e-4)
    else:
        solver = LimitedBestResponse(policy, depth=args.depth, epsilon=1e-4)
    out = {}
    for seat in (0, 1):
        start = time.perf_counter()
        values = evaluate_seat(solver, seat, 0, args.sweeps)
        out[seat] = (values.mean(), time.perf_counter() - start)
    calls = solver.network_queries
    print(f"solver={args.solver} depth={args.depth} sweeps={args.sweeps} games/seat={solver.n * args.sweeps}")
    print(f"  seat0 p={out[0][0]:.6f} t={out[0][1]:.1f}s   seat1 p={out[1][0]:.6f} t={out[1][1]:.1f}s")
    print(f"  network calls={calls} rows={solver.network_rows} rows/call={solver.network_rows / max(calls, 1):.1f} "
          f"network_s={solver.network_query_s:.1f} ({100 * solver.network_query_s / (out[0][1] + out[1][1]):.0f}% of total)")
    print(f"  plan_nodes={solver.plan_nodes} search_branches={solver.search_branches} "
          f"skipped={solver.skipped_search_branches}")
    for name in ("_probs", "_plan", "_choice", "_reach"):
        info = getattr(solver, name).cache_info()
        print(f"  cache {name}: hits={info.hits} misses={info.misses} size={info.currsize}/{info.maxsize}")


if __name__ == "__main__":
    main()
