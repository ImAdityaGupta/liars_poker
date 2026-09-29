"""Profile fixed per-iteration costs on a separate tabular bridge checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
from statistics import median
import time

from run_cfr_plus_18_tabular_bridge import TabularBridge


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--arm", default="sample_both")
    parser.add_argument("--roots", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=5)
    args = parser.parse_args()

    bridge = TabularBridge(args.arm, seed=17, roots=args.roots)
    meta = bridge.restore(args.checkpoint)
    print("source iteration:", meta["iteration"], flush=True)
    solver = bridge.solver
    samples: list[dict[str, float]] = []
    current: dict[str, float] = {}

    def wrap(name: str, label: str):
        original = getattr(solver, name)

        def timed(*a, **kw):
            start = time.perf_counter()
            result = original(*a, **kw)
            current[label] = current.get(label, 0.0) + time.perf_counter() - start
            return result

        setattr(solver, name, timed)

    wrap("_update_strategy_for_player", "strategy_rebuild_s")
    wrap("_recompute_likelihoods", "likelihood_recompute_s")
    for index in range(args.iterations):
        current = {}
        start = time.perf_counter()
        detail = bridge.iterate()
        elapsed = time.perf_counter() - start
        current.update(detail)
        current["iteration_s"] = elapsed
        current["unattributed_s"] = elapsed - sum(current[key] for key in (
            "strategy_rebuild_s", "likelihood_recompute_s", "sampling_s", "update_s"
        ))
        samples.append(current)
        print("sample", index + 1, {key: round(value, 4) for key, value in current.items()
                                  if key.endswith("_s")}, flush=True)
    print("median", {key: round(median(row[key] for row in samples), 4)
                     for key in ("iteration_s", "strategy_rebuild_s",
                                 "likelihood_recompute_s", "sampling_s", "update_s",
                                 "unattributed_s")}, flush=True)


if __name__ == "__main__":
    main()
