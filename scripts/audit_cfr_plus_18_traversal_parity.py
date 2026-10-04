#!/usr/bin/env python3
"""Compare recursive bridge and batched CPU traversal on fixed deals and throughput."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from liars_poker.algo.neural_cfr_plus_gpu import GPUDeepCFRPlusTraverser
from scripts.run_cfr_plus_18_tabular_bridge import TabularBridge
from scripts.run_cfr_plus_18_tabular_discount import make_trainer


def main() -> None:
    torch.set_num_threads(4)
    old = TabularBridge("conditional", seed=17, roots=1)
    new = make_trainer("A_cfr_plus_linear")
    traverser = GPUDeepCFRPlusTraverser(new)
    solver = old.solver

    # Use the first legal claim everywhere. Opponent sampling is then
    # deterministic, while both implementations still expand all traverser
    # actions and evaluate both terminal wins and losses.
    solver.S.fill(0)
    for hid, cols in enumerate(solver.legal_cols):
        if cols:
            claims = [col for col in cols if col > 0]
            solver.S[hid, :, claims[0] if claims else 0] = 1.0

    def fixed_strategy(_actor, features, legal_mask):
        claim_legal = legal_mask[:, 1:]
        first_claim = claim_legal.float().argmax(dim=1) + 1
        selected = torch.where(claim_legal.any(dim=1), first_claim,
                               torch.zeros_like(first_claim))
        strategy = torch.nn.functional.one_hot(
            selected, num_classes=new.encoder.action_dim,
        ).float()
        return torch.zeros_like(strategy), strategy

    traverser._regrets_and_strategy = fixed_strategy
    hand_index = {
        tuple(int(x) for x in row.tolist()): i
        for i, row in enumerate(new.hand_counts)
    }
    checked = 0
    max_target_error = 0.0
    max_root_error = 0.0
    for _ in range(2):
        h0, h1 = old._deal()
        hand0 = torch.tensor(solver.hand_rank_counts[h0, 1:], dtype=torch.float32)[None, :]
        hand1 = torch.tensor(solver.hand_rank_counts[h1, 1:], dtype=torch.float32)[None, :]
        traverser._sample_deals = lambda _n, a=hand0, b=hand1: (a, b, a + b)
        for pid in (0, 1):
            counts = np.zeros((solver.H, solver.n_hands), dtype=np.uint32)
            sums = np.zeros_like(solver.S)
            touched = []
            old_value = old._traverse(0, h0, h1, pid, counts, sums, touched)
            new.regret_buffers[pid].clear()
            result = traverser.run_traversals(pid, 1, profile=True)
            new_value = float(result["root_values"][0])
            max_root_error = max(max_root_error, abs(old_value - new_value))

            expected = {}
            for hid, hand in touched:
                counts_tuple = tuple(int(x) for x in solver.hand_rank_counts[hand, 1:])
                key = hid * new.n_rank_hands + hand_index[counts_tuple]
                expected[key] = sums[hid, hand] / counts[hid, hand]
            buffer = new.regret_buffers[pid]
            keys = new._table_indices(buffer.features[:buffer.size]).numpy()
            actual = {}
            for key, value in zip(keys, buffer.targets[:buffer.size].numpy()):
                if int(key) in actual:
                    raise AssertionError("Repeated infoset in a single fixed-deal traversal")
                actual[int(key)] = value
            if expected.keys() != actual.keys():
                raise AssertionError({
                    "missing": len(expected.keys() - actual.keys()),
                    "extra": len(actual.keys() - expected.keys()),
                })
            for key in expected:
                max_target_error = max(
                    max_target_error,
                    float(np.max(np.abs(expected[key] - actual[key]))),
                )
            checked += len(expected)
    if max_root_error > 1e-5 or max_target_error > 1e-5:
        raise AssertionError((max_root_error, max_target_error))
    print(json.dumps({"fixed_deal_records_checked": checked,
                      "max_root_value_error": max_root_error,
                      "max_regret_target_error": max_target_error}), flush=True)

    # Restore the shared initial uniform policy, and time traversal only.
    solver.S[:] = solver.uniform_rows[:, None, :]
    del traverser._regrets_and_strategy
    del traverser._sample_deals
    roots = 1024
    timings = {}
    for pid in (0, 1):
        old.roots = roots
        start = time.perf_counter()
        counts, _, _ = old._sample(pid)
        old_s = time.perf_counter() - start
        new.regret_buffers[pid].clear()
        start = time.perf_counter()
        first = traverser.run_traversals(pid, roots // 2)
        second = traverser.run_traversals(pid, roots // 2)
        new_s = time.perf_counter() - start
        timings[f"player_{pid}"] = {
            "roots": roots, "recursive_s": old_s, "batched_s": new_s,
            "recursive_regret_records": int(counts.sum()),
            "batched_regret_records": int(first["regret_records"] + second["regret_records"]),
        }
    print(json.dumps({"cpu_traversal_timing": timings}), flush=True)


if __name__ == "__main__":
    main()
