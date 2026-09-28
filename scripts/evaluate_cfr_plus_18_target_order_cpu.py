#!/usr/bin/env python3
"""Exact exploitability of saved 18-claim target-order experiment policies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from liars_poker.algo.br_exact_dense_to_dense import best_response_dense
from liars_poker.policies.neural import compile_neural_to_dense
from liars_poker.serialization import load_policy


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def plot(rows: list[dict], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    modes = ("clip_each_record", "aggregate_then_clip")
    colors = {modes[0]: "C0", modes[1]: "C1"}
    for mode in modes:
        arm_rows = [row for row in rows if row["mode"] == mode]
        if not arm_rows:
            continue
        for seed in sorted({row["seed"] for row in arm_rows}):
            sub = sorted((row for row in arm_rows if row["seed"] == seed), key=lambda r: r["snapshot_min"])
            ax.plot([r["snapshot_min"] for r in sub], [r["exploitability"] for r in sub],
                    marker="o", alpha=0.65, color=colors[mode], label=f"{mode}, seed {seed}")
    ax.set(xlabel="CFR+ training minutes", ylabel="Exact exploitability",
           title="18-claim neural CFR+: regret target construction")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    result_path = run_dir / "exact_evaluations.jsonl"
    rows = read_rows(result_path)
    done = {(r["mode"], r["seed"], r["snapshot_min"]) for r in rows}
    tasks = []
    for arm in sorted(run_dir.glob("*__seed_*")):
        if not arm.is_dir():
            continue
        mode, seed_text = arm.name.split("__seed_")
        seed = int(seed_text)
        for snap in sorted((arm / "snapshots").glob("*m"), reverse=True):
            minute = int(snap.name.removesuffix("m"))
            if (mode, seed, minute) not in done:
                tasks.append((mode, seed, minute, snap / "average_policy"))
    # Final snapshots first, so an interrupted evaluation still gives the
    # main endpoint comparison. Thereafter fill in the time curves.
    tasks.sort(key=lambda x: (-x[2], x[0], x[1]))
    for mode, seed, minute, policy_dir in tasks:
        start = time.perf_counter()
        policy, spec = load_policy(str(policy_dir))
        dense = compile_neural_to_dense(policy, batch_size=65_536)
        _, meta = best_response_dense(spec, dense, store_state_values=False)
        p_first, p_second = meta["computer"].exploitability()
        row = {
            "mode": mode,
            "seed": seed,
            "snapshot_min": minute,
            "p_first": float(p_first),
            "p_second": float(p_second),
            "exploitability": float(p_first + p_second - 1.0),
            "evaluation_s": time.perf_counter() - start,
            "policy_dir": str(policy_dir),
        }
        with result_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
            handle.flush()
        rows.append(row)
        plot(rows, run_dir / "exact_exploitability.png")
        print(f"{mode} seed={seed} {minute:02d}m: exact={row['exploitability']:.6f} "
              f"eval={row['evaluation_s']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
