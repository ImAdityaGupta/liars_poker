#!/usr/bin/env python3
"""Exercise dense snapshot serialization and exact evaluation from a saved control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from liars_poker.algo.br_exact_dense_to_dense import best_response_dense
from liars_poker.algo.cfr_discount_exact_average import ExactAverageTabularDiscountTrainer
from liars_poker.serialization import load_policy, save_policy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    start = time.perf_counter()
    trainer = ExactAverageTabularDiscountTrainer.load_fork_checkpoint(args.checkpoint)
    policy = trainer.exact_average_policy()
    with tempfile.TemporaryDirectory(prefix="bridge_dense_probe_") as temp:
        directory = Path(temp) / "policy"
        save_policy(policy, str(directory))
        size = sum(p.stat().st_size for p in directory.iterdir() if p.is_file())
        restored, spec = load_policy(str(directory))
        _, meta = best_response_dense(spec, restored, store_state_values=False)
        p0, p1 = meta["computer"].exploitability()
    print(json.dumps({
        "iteration": trainer.iteration,
        "policy_mib": size / 2**20,
        "exploitability": float(p0 + p1 - 1),
        "elapsed_s": time.perf_counter() - start,
    }), flush=True)


if __name__ == "__main__":
    main()
