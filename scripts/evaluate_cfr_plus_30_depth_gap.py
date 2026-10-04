#!/usr/bin/env python3
"""Paired depth-gap scoring for the 30-claim precise BR screen.

Re-scores saved policies with a deeper (or batched) depth-limited responder on
the same shards as the precise depth-2 screen. ``evaluate_seat`` draws every
random input from (seat, shard) before play, so a shard scored here pairs game
for game with the depth-2 shard of the same policy in
``precise_evaluations/shards``. Opponent queries are batched and may run on the
GPU (``BatchedLimitedBestResponse``); the search itself is unchanged.

Commands:
  one     score one (policy, depth, shard) and write an .npz
  queue   run many shards with a worker pool, then report
  report  pair finished shards with depth 2 and summarise
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import types

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
sys.modules.setdefault("fcntl", types.ModuleType("fcntl"))  # imported by the precise script; unused here

from evaluate_cfr_plus_30_precise import evaluate_seat  # noqa: E402

RUN = Path("/root/liars_poker/artifacts/cfr_plus_30_claim_first_run/main_20261003")
OUT = Path("/root/liars_poker/artifacts/cfr_plus_30_br_depth_gap/main_20261004")


def policy_path(target: str) -> Path:
    """``<arm>_<snapshot>_<kind>``: kind is o4 (O4 average), online (online average) or current."""
    arm, snap, kind = target.split("_")
    folder = {"o4": "average_policy", "online": "online_policy", "current": "current_policy"}[kind]
    return RUN / arm / "snapshots" / snap / folder


def one(args: argparse.Namespace) -> None:
    import torch
    from liars_poker.algo.br_limited_batched import BatchedLimitedBestResponse
    from liars_poker.serialization import load_policy

    torch.set_num_threads(1)
    dst = OUT / "shards" / f"d{args.depth}" / args.target / f"{args.shard:03d}.npz"
    if dst.exists():
        return
    started = time.perf_counter()
    policy, _ = load_policy(str(policy_path(args.target)))
    if args.device != "cpu":
        policy.model_p1.to(args.device)
        policy.model_p2.to(args.device)
        policy.device = torch.device(args.device)
    solver = BatchedLimitedBestResponse(policy, depth=args.depth, epsilon=1e-4, cache_size=args.cache)
    seat0 = evaluate_seat(solver, 0, args.shard, args.sweeps)
    seat1 = evaluate_seat(solver, 1, args.shard, args.sweeps)
    dst.parent.mkdir(parents=True, exist_ok=True)
    staged = dst.with_name(dst.name + ".tmp")
    with staged.open("wb") as out:
        np.savez_compressed(out, seat0=seat0, seat1=seat1, n_hands=np.int64(solver.n),
                            sweeps=np.int64(args.sweeps), shard=np.int64(args.shard))
    os.replace(staged, dst)
    print(f"{args.target} d={args.depth} shard={args.shard} sweeps={args.sweeps} "
          f"e={seat0.mean() + seat1.mean() - 1:.6f} calls={solver.network_queries} "
          f"rows={solver.network_rows} elapsed={time.perf_counter() - started:.1f}s", flush=True)


def load(depth: int, target: str, shard: int):
    if depth == 2 and not (OUT / "shards" / "d2" / target / f"{shard:03d}.npz").exists():
        path = RUN / "precise_evaluations" / "shards" / target / f"{shard:03d}.npz"
    else:
        path = OUT / "shards" / f"d{depth}" / target / f"{shard:03d}.npz"
    if not path.exists():
        return None
    with np.load(path) as arr:
        return arr["seat0"].astype(np.float64), arr["seat1"].astype(np.float64)


def mean_se(x: np.ndarray) -> tuple[float, float]:
    return float(x.mean()), float(x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 else float("nan")


def report(targets: list[str], depth: int, shards: int) -> list[dict]:
    rows = []
    for target in targets:
        deep, base = [], []
        for shard in range(shards):
            d = load(depth, target, shard)
            b = load(2, target, shard)
            if d is None or b is None:
                continue
            if len(d[0]) != len(b[0]):
                continue  # pairing needs the same sweeps per shard
            deep.append(d)
            base.append(b)
        if not deep:
            continue
        s0d = np.concatenate([x[0] for x in deep]); s1d = np.concatenate([x[1] for x in deep])
        s0b = np.concatenate([x[0] for x in base]); s1b = np.concatenate([x[1] for x in base])
        row = {"target": target, "depth": depth, "paired_shards": len(deep), "sweeps": len(s0d)}
        for name, a, b in (("first_seat", s0d, s0b), ("second_seat", s1d, s1b),
                           ("exploitability", s0d + s1d, s0b + s1b)):
            m_deep, se_deep = mean_se(a)
            m_base, _ = mean_se(b)
            diff, se_diff = mean_se(a - b)
            offset = -1.0 if name == "exploitability" else 0.0
            row[name] = {f"d{depth}": m_deep + offset, "d2": m_base + offset,
                         "d2_se_unpaired": se_deep, "gain": diff, "gain_ci95": [diff - 1.96 * se_diff,
                                                                               diff + 1.96 * se_diff]}
        rows.append(row)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"summary_d{depth}.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    for r in rows:
        e = r["exploitability"]
        print(f"{r['target']:18s} shards={r['paired_shards']:2d} d2={e['d2']:.4f} d{depth}={e[f'd{depth}']:.4f} "
              f"gain={e['gain']:+.4f} [{e['gain_ci95'][0]:+.4f},{e['gain_ci95'][1]:+.4f}]  "
              f"first-seat gain={r['first_seat']['gain']:+.4f} second-seat gain={r['second_seat']['gain']:+.4f}",
              flush=True)
    return rows


def queue(args: argparse.Namespace) -> None:
    jobs = [(t, s) for s in range(args.shards) for t in args.targets]
    active: list[tuple[subprocess.Popen, str, int]] = []
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    logs = OUT / "logs" / f"d{args.depth}"
    logs.mkdir(parents=True, exist_ok=True)
    while jobs or active:
        active = [(p, t, s) for p, t, s in active if p.poll() is None]
        while jobs and len(active) < args.workers:
            target, shard = jobs.pop(0)
            if (OUT / "shards" / f"d{args.depth}" / target / f"{shard:03d}.npz").exists():
                continue
            handle = (logs / f"{target}_{shard:03d}.log").open("a", encoding="utf-8")
            cmd = [sys.executable, "-u", str(Path(__file__).resolve()), "one", "--target", target,
                   "--shard", str(shard), "--depth", str(args.depth), "--sweeps", str(args.sweeps),
                   "--device", args.device, "--cache", str(args.cache)]
            active.append((subprocess.Popen(cmd, cwd=REPO, env=env, stdout=handle,
                                            stderr=subprocess.STDOUT), target, shard))
        report(args.targets, args.depth, args.shards)
        time.sleep(30)
    report(args.targets, args.depth, args.shards)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("one", "queue", "report"))
    parser.add_argument("--target", help="e.g. w2048_1440m_o4")
    parser.add_argument("--targets", nargs="+", default=["w2048_1440m_o4", "w512_1440m_o4"])
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=30)
    parser.add_argument("--sweeps", type=int, default=58)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache", type=int, default=1_000_000, help="opponent histories cached per worker")
    args = parser.parse_args()
    if args.command == "one":
        one(args)
    elif args.command == "queue":
        queue(args)
    else:
        report(args.targets, args.depth, args.shards)


if __name__ == "__main__":
    main()
