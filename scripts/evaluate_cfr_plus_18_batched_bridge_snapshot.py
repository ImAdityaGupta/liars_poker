#!/usr/bin/env python3
"""Exact exploitability of a saved dense or neural 18-claim average policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from liars_poker.algo.br_exact_dense_to_dense import best_response_dense
from liars_poker.policies.neural import compile_neural_to_dense
from liars_poker.policies.tabular_dense import DenseTabularPolicy
from liars_poker.serialization import load_policy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("policy_dir", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    start = time.perf_counter()
    policy, spec = load_policy(str(args.policy_dir))
    dense = (policy if isinstance(policy, DenseTabularPolicy)
             else compile_neural_to_dense(policy, batch_size=65_536))
    _, meta = best_response_dense(spec, dense, store_state_values=False)
    p_first, p_second = meta["computer"].exploitability()
    print(json.dumps({
        "policy_kind": policy.POLICY_KIND,
        "p_first": float(p_first), "p_second": float(p_second),
        "exploitability": float(p_first + p_second - 1),
        "evaluation_s": time.perf_counter() - start,
    }))


if __name__ == "__main__":
    main()
