#!/usr/bin/env python3
"""Small-game CPU screen: tabularized neural CFR+ regression targets.

Both variants use the same external/action sampler and exact tabular policy
storage. Only the order of clipping and averaging sampled targets changes.
This intentionally models infinite-capacity *conditional regression* at each
visited infoset under ordinary squared error. It omits the production
positive-target loss weighting and is not a standard tabular MCCFR+ implementation.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import sys
from typing import DefaultDict

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from liars_poker.algo.br_exact_dense_to_dense import best_response_dense
from liars_poker.algo.cfr_plus_dense import CFRPlusDense
from liars_poker.core import generate_deck
from liars_poker.env import resolve_call_winner, rules_for_spec
from liars_poker.infoset import CALL
from liars_poker.policies.tabular_dense import DenseTabularPolicy
from scripts.diagnose_neural_cfr_plus_cpu import SPEC


def exact_exploitability(policy: DenseTabularPolicy) -> float:
    _, meta = best_response_dense(SPEC, policy, store_state_values=False)
    first, second = meta["computer"].exploitability()
    return float(first + second - 1.0)


class SampledTabularRegression:
    def __init__(self, *, cap: int | None, clip_order: str, seed: int):
        if clip_order not in {"before", "after"}:
            raise ValueError(clip_order)
        self.cap = cap
        self.clip_order = clip_order
        self.rng = random.Random(seed)
        self.rules = rules_for_spec(SPEC)
        self.deck = list(generate_deck(SPEC))
        self.policy = DenseTabularPolicy(SPEC)
        self.H, self.N, self.A = self.policy.S.shape
        self.regret = [np.zeros_like(self.policy.S, dtype=np.float64) for _ in (0, 1)]
        self.average_sums = np.zeros_like(self.policy.S, dtype=np.float64)
        self.iteration = 0

    def _refresh_strategy(self, player: int) -> None:
        for hid in range(self.H):
            if int(self.policy.popcount[hid] & 1) != player:
                continue
            legal = self.policy.legal_mask[hid]
            if not legal.any():
                continue
            for hand_idx in range(self.N):
                positive = np.maximum(self.regret[player][hid, hand_idx], 0.0)
                positive[~legal] = 0.0
                total = positive.sum()
                if total > 0:
                    self.policy.S[hid, hand_idx] = positive / total
                else:
                    self.policy.S[hid, hand_idx] = legal / legal.sum()
        self.policy.recompute_likelihoods()

    def _accumulate_average(self, player: int) -> None:
        own_reach = self.policy.L_pid0 if player == 0 else self.policy.L_pid1
        for hid in range(self.H):
            if int(self.policy.popcount[hid] & 1) == player:
                self.average_sums[hid] += (
                    self.iteration
                    * own_reach[hid, :, None]
                    * self.policy.S[hid]
                )

    def _sample_opponent_action(self, hid: int, hand_idx: int, legal) -> int:
        probs = self.policy.S[hid, hand_idx]
        pick = self.rng.random()
        running = 0.0
        for action in legal:
            running += float(probs[0 if action == CALL else action + 1])
            if pick <= running:
                return action
        return legal[-1]

    def _traverse(
        self,
        history: tuple[int, ...],
        hands: tuple[int, int],
        traverser: int,
        inverse_path_probability: float,
        records: list[tuple[int, int, np.ndarray, float]],
    ) -> float:
        if history and history[-1] == CALL:
            winner = resolve_call_winner(
                SPEC, history, (hands[0],), (hands[1],)
            )
            return 1.0 if winner == ("P1" if traverser == 0 else "P2") else -1.0

        actor = len(history) & 1
        hid = sum(1 << claim for claim in history)
        hand_idx = self.policy.hand_to_idx[(hands[actor],)]
        legal = self.rules.legal_actions_from_last(history[-1] if history else None)
        if actor != traverser:
            action = self._sample_opponent_action(hid, hand_idx, legal)
            return self._traverse(
                history + (action,),
                hands,
                traverser,
                inverse_path_probability,
                records,
            )

        values = np.zeros(self.A, dtype=np.float64)
        if CALL in legal:
            values[0] = self._traverse(
                history + (CALL,),
                hands,
                traverser,
                inverse_path_probability,
                records,
            )
        claims = [action for action in legal if action != CALL]
        if self.cap is None or len(claims) <= self.cap:
            selected = claims
            inclusion = 1.0
        else:
            selected = self.rng.sample(claims, self.cap)
            inclusion = self.cap / len(claims)
        for action in selected:
            child = self._traverse(
                history + (action,),
                hands,
                traverser,
                inverse_path_probability / inclusion,
                records,
            )
            values[action + 1] = child / inclusion

        sigma = self.policy.S[hid, hand_idx]
        node_value = float(np.dot(sigma, values))
        instant = (values - node_value) * self.policy.legal_mask[hid]
        records.append((hid, hand_idx, instant, inverse_path_probability))
        return node_value

    def _collect_and_update(self, player: int, traversals: int) -> None:
        # One aggregate per infoset. The 'before' branch averages individually
        # clipped targets, matching the infinite-capacity supervised target;
        # 'after' averages raw updates first and clips once.
        accum: DefaultDict[tuple[int, int], dict] = defaultdict(
            lambda: {
                "weight": 0.0,
                "weighted_raw": np.zeros(self.A, dtype=np.float64),
                "weighted_clipped": np.zeros(self.A, dtype=np.float64),
            }
        )
        t = self.iteration
        old_factor = (t - 1) / t
        for _ in range(traversals):
            deck = self.deck.copy()
            self.rng.shuffle(deck)
            hands = (deck[0], deck[1])
            records: list[tuple[int, int, np.ndarray, float]] = []
            self._traverse((), hands, player, 1.0, records)
            for hid, hand_idx, instant, weight in records:
                row = accum[(hid, hand_idx)]
                old = self.regret[player][hid, hand_idx]
                row["weight"] += weight
                row["weighted_raw"] += weight * instant
                row["weighted_clipped"] += weight * np.maximum(
                    old_factor * old + instant / t, 0.0
                )

        self.regret[player] *= old_factor
        for (hid, hand_idx), row in accum.items():
            if self.clip_order == "before":
                self.regret[player][hid, hand_idx] = (
                    row["weighted_clipped"] / row["weight"]
                )
            else:
                self.regret[player][hid, hand_idx] = np.maximum(
                    self.regret[player][hid, hand_idx]
                    + row["weighted_raw"] / row["weight"] / t,
                    0.0,
                )

    def iterate(self, traversals: int) -> None:
        self.iteration += 1
        for player in (0, 1):
            self._refresh_strategy(player)
            self._accumulate_average(player)
            self._collect_and_update(player, traversals)

    def average_policy(self) -> DenseTabularPolicy:
        average = DenseTabularPolicy(SPEC)
        totals = self.average_sums.sum(axis=2)
        have_data = totals > 0
        average.S[have_data] = (
            self.average_sums[have_data] / totals[have_data][:, None]
        )
        average.recompute_likelihoods()
        return average


def run_exact_reference(iterations: int, eval_every: int) -> list[dict]:
    solver = CFRPlusDense(SPEC)
    rows = []
    for iteration in range(1, iterations + 1):
        solver.iterate()
        if iteration == 1 or iteration % eval_every == 0:
            rows.append(
                {
                    "method": "exact dense CFR+",
                    "iteration": iteration,
                    "exploitability": exact_exploitability(solver.average_policy()),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--traversals", type=int, default=32)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--seeds", default="17,23")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.iterations, args.traversals, args.eval_every) <= 0:
        parser.error("iterations, traversals, and eval-every must be positive")

    rows = run_exact_reference(args.iterations, args.eval_every)
    for cap in (None, 2):
        for clip_order in ("after", "before"):
            for seed in (int(x) for x in args.seeds.split(",")):
                trainer = SampledTabularRegression(
                    cap=cap, clip_order=clip_order, seed=seed
                )
                label = f"cap={cap or 'full'} clip={clip_order} seed={seed}"
                for iteration in range(1, args.iterations + 1):
                    trainer.iterate(args.traversals)
                    if iteration == 1 or iteration % args.eval_every == 0:
                        score = exact_exploitability(trainer.average_policy())
                        rows.append(
                            {
                                "method": label,
                                "iteration": iteration,
                                "exploitability": score,
                            }
                        )
                        print(label, iteration, f"{score:.6f}", flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print("Wrote", args.output)


if __name__ == "__main__":
    main()
