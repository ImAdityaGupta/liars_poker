#!/usr/bin/env python3
"""Compare the direct table compiler with the established dense compiler."""

from __future__ import annotations

from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from liars_poker.algo.cfr_discount_exact_average import ExactAverageTabularDiscountTrainer
from scripts.run_cfr_plus_18_tabular_discount import make_trainer


def main() -> None:
    torch.set_num_threads(4)
    trainer = make_trainer("A_cfr_plus_linear",
                           trainer_type=ExactAverageTabularDiscountTrainer)
    # Include sparse and populated rows, zero fallbacks and varying supports.
    rng = np.random.default_rng(1749)
    keys = torch.from_numpy(rng.choice(len(trainer.regret_table), 100_000,
                                       replace=False)).long()
    trainer.regret_table[keys] = torch.from_numpy(
        rng.exponential(1.0, (len(keys), trainer.encoder.action_dim)).astype(np.float32)
    )
    trainer.table_initialized[keys] = True
    start = time.perf_counter()
    reference = trainer.current_policy_dense()
    reference_s = time.perf_counter() - start
    start = time.perf_counter()
    candidate = trainer.current_policy_exact_dense()
    candidate_s = time.perf_counter() - start
    start = time.perf_counter()
    trainer.current_policy_exact_dense()
    reuse_s = time.perf_counter() - start
    strategy_error = float(np.max(np.abs(reference.S - candidate.S)))
    reach_error = max(
        float(np.max(np.abs(reference.L_pid0 - candidate.L_pid0))),
        float(np.max(np.abs(reference.L_pid1 - candidate.L_pid1))),
    )
    print({"strategy_max_abs_error": strategy_error,
           "own_reach_max_abs_error": reach_error,
           "reference_s": reference_s, "direct_first_s": candidate_s,
           "direct_reuse_s": reuse_s}, flush=True)
    if strategy_error > 1e-6 or reach_error > 1e-6:
        raise AssertionError("Direct table compiler does not match reference")


if __name__ == "__main__":
    main()
