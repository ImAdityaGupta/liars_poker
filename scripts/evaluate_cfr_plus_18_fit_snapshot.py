#!/usr/bin/env python3
"""CPU exact evaluation of a saved average or neural regret-matching policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from liars_poker.algo.br_exact_dense_to_dense import best_response_dense
from liars_poker.policies.neural import compile_neural_to_dense
from liars_poker.policies.neural_regret import NeuralRegretMatchingPolicy
from liars_poker.policies.tabular_dense import DenseTabularPolicy
from liars_poker.serialization import load_policy


def compile_current(policy: NeuralRegretMatchingPolicy,
                    batch_size: int = 16_384) -> DenseTabularPolicy:
    """CPU equivalent of DeepCFRPlusTrainer.current_policy_dense()."""
    dense = DenseTabularPolicy(policy.spec)
    hands = dense.hands
    n_hands = len(hands)
    per_batch = max(1, batch_size // n_hands)
    rank_dim = policy.spec.ranks
    input_dim = policy.encoder.input_dim
    action_dim = policy.encoder.action_dim
    bits = np.arange(policy.encoder.k, dtype=np.int64)
    hand_features = policy.encoder.encode_hands(hands, ())
    with torch.inference_mode():
        for pid in (0, 1):
            hids_for_player = np.flatnonzero((dense.popcount & 1) == pid)
            model = policy._model(pid)
            for start in range(0, len(hids_for_player), per_batch):
                hids = hids_for_player[start:start + per_batch]
                history = ((hids[:, None].astype(np.int64) >> bits[None, :]) & 1).astype(np.float32)
                features = np.empty((len(hids), n_hands, input_dim), dtype=np.float32)
                features[:, :, :rank_dim] = hand_features[None, :, :rank_dim]
                features[:, :, rank_dim:] = history[:, None, :]
                x = torch.from_numpy(features.reshape(-1, input_dim))
                values = model(x).float().reshape(len(hids), n_hands, action_dim)
                legal = torch.from_numpy(dense.legal_mask[hids])
                positive = torch.relu(values) * legal[:, None, :]
                total = positive.sum(dim=2, keepdim=True)
                matched = positive / total.clamp_min(1e-8)
                fallback = legal[:, None, :].float()
                fallback = fallback / fallback.sum(dim=2, keepdim=True).clamp_min(1.0)
                dense.S[hids] = torch.where(total > 0, matched, fallback).numpy()
    dense.recompute_likelihoods()
    return dense


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("policy_dir", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    start = time.perf_counter()
    policy, spec = load_policy(str(args.policy_dir))
    dense = (policy if isinstance(policy, DenseTabularPolicy)
             else compile_current(policy) if isinstance(policy, NeuralRegretMatchingPolicy)
             else compile_neural_to_dense(policy, batch_size=65_536))
    _, meta = best_response_dense(spec, dense, store_state_values=False)
    p_first, p_second = meta["computer"].exploitability()
    print(json.dumps({"policy_kind": policy.POLICY_KIND,
                      "p_first": float(p_first), "p_second": float(p_second),
                      "exploitability": float(p_first + p_second - 1.0),
                      "evaluation_s": time.perf_counter() - start}))


if __name__ == "__main__":
    main()
