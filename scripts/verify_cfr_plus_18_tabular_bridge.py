#!/usr/bin/env python3
"""Small-spec mathematical checks before spending hours on the 18-claim bridge."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from liars_poker.algo.cfr_plus_dense import CFRPlusDense
from liars_poker.core import GameSpec
import run_cfr_plus_18_tabular_bridge as bridge


def main() -> None:
    bridge.SPEC = GameSpec(ranks=2, suits=2, hand_size=1,
                           claim_kinds=("RankHigh", "Pair"), suit_symmetry=True)
    reference = CFRPlusDense(bridge.SPEC)
    exact = bridge.TabularBridge("exact", seed=17, roots=16)
    for _ in range(3):
        reference.iterate()
        exact.iterate()
        for name in ("R0", "R1"):
            expected = getattr(reference, name) * exact.row_scale[None, :, None]
            np.testing.assert_allclose(getattr(exact.solver, name), expected, rtol=2e-12, atol=2e-12)
        for name in ("SS0", "SS1", "S"):
            np.testing.assert_allclose(getattr(exact.solver, name), getattr(reference, name),
                                       rtol=2e-12, atol=2e-12)
    before_current_eval = exact.solver.S.copy()
    exact.policy("current")
    np.testing.assert_array_equal(exact.solver.S, before_current_eval)
    reference.iterate()
    exact.iterate()
    np.testing.assert_allclose(exact.solver.R0,
                               reference.R0 * exact.row_scale[None, :, None], rtol=2e-12, atol=2e-12)
    print("PASS exact bridge equals dense CFR+ up to per-hand regret units", flush=True)

    sampled = bridge.TabularBridge("sample_both", seed=23, roots=12_000)
    s = sampled.solver
    s._update_strategy_for_player(0)
    s._recompute_likelihoods()
    counts, sums, _ = sampled._sample(0)
    raw_by_hid = {}

    def capture(hid, opp_reach, raw):
        raw_by_hid[hid] = raw.copy()
        return np.zeros_like(raw)

    s._update_player(0, weight=0.0, increment_transform=capture)
    root_qhat = counts[0] / sampled.roots
    np.testing.assert_allclose(root_qhat, sampled.hand_prob, atol=.02)
    # This comparison checks both the root chance normalization and the
    # all-roots sampled action-value estimate before clipping.
    target = raw_by_hid[0] * sampled.row_scale[None, :]
    estimate = sums[0, :, s.legal_cols[0]] / sampled.roots
    np.testing.assert_allclose(estimate, target, atol=.035)
    print("PASS sampled root reach/value estimates match exact units", flush=True)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "arm" / "latest_checkpoint.npz"
        path.parent.mkdir()
        exact.checkpoint(path, measured_s=71.0, next_eval_s=100.0, next_checkpoint_s=120.0)
        restored = bridge.TabularBridge("exact", seed=17, roots=16)
        meta = restored.restore(path)
        assert meta["iteration"] == 4 and meta["measured_s"] == 71.0
        for name in ("R0", "R1", "SS0", "SS1"):
            np.testing.assert_array_equal(getattr(restored.solver, name), getattr(exact.solver, name))
        assert exact._deal() == restored._deal()
    print("PASS rolling checkpoint restores tables and RNG", flush=True)

    first_step = {}
    for arm in bridge.ARMS:
        t = bridge.TabularBridge(arm, seed=17, roots=8)
        stats = t.iterate()
        assert t.solver.iteration == 1 and stats["update_s"] >= 0
        assert np.isfinite(t.solver.R0).all() and np.isfinite(t.solver.R1).all()
        first_step[arm] = t.policy("current").S.copy()
    np.testing.assert_allclose(first_step["exact"], first_step["ignore_reach"], atol=1e-7)
    np.testing.assert_allclose(first_step["sample_both"], first_step["conditional"], atol=1e-7)
    print("PASS all eight update paths complete a small-spec iteration", flush=True)

    for gated_arm, reference_arm in (("exact_reach_gated", "exact"),
                                     ("unit_reach_gated", "ignore_reach")):
        gated = bridge.TabularBridge(gated_arm, seed=31, roots=32)
        reference = bridge.TabularBridge(reference_arm, seed=31, roots=32)
        for trainer in (gated, reference):
            trainer.solver.iteration = 1
            trainer.solver._update_strategy_for_player(0)
            trainer.solver._recompute_likelihoods()
        counts, sums, _ = gated._sample(0)
        gated._exact_update(0, counts, sums)
        reference._exact_update(0, None, None)
        np.testing.assert_allclose(
            gated.solver.R0,
            reference.solver.R0 * (counts[:, :, None] > 0),
            rtol=2e-12, atol=2e-12,
        )
    print("PASS both visit-gated controls apply the intended exact increments", flush=True)


if __name__ == "__main__":
    main()
