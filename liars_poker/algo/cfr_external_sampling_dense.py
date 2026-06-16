from __future__ import annotations

import random
import time
from typing import Dict, Tuple

import numpy as np

from liars_poker.core import GameSpec, generate_deck, possible_starting_hands
from liars_poker.env import resolve_call_winner, rules_for_spec
from liars_poker.infoset import CALL
from liars_poker.policies.tabular_dense import DenseTabularPolicy


class CFRExternalSamplingDense:
    """Tabular external-sampling CFR using the same sampled traversal as Deep CFR."""

    def __init__(
        self,
        spec: GameSpec,
        *,
        seed: int = 0,
        strategy_weighting: str = "linear",
        dtype: np.dtype = np.float64,
    ) -> None:
        if strategy_weighting not in {"linear", "uniform"}:
            raise ValueError("strategy_weighting must be 'linear' or 'uniform'.")

        self.spec = spec
        self.rules = rules_for_spec(spec)
        self.rng = random.Random(seed)
        self.strategy_weighting = strategy_weighting
        self.dtype = dtype
        self.iteration = 0

        template = DenseTabularPolicy(spec)
        self.k = template.k
        self.H, self.n_hands, self.A = template.S.shape
        self.hands = tuple(possible_starting_hands(spec))
        self.hand_to_idx = {hand: idx for idx, hand in enumerate(self.hands)}
        self.legal_actions = template.legal_actions
        self.legal_mask = template.legal_mask

        shape = (self.H, self.n_hands, self.A)
        self.R0 = np.zeros(shape, dtype=dtype)
        self.R1 = np.zeros(shape, dtype=dtype)
        self.SS0 = np.zeros(shape, dtype=dtype)
        self.SS1 = np.zeros(shape, dtype=dtype)

    @staticmethod
    def _action_col(action: int) -> int:
        return 0 if action == CALL else action + 1

    def _history_to_hid(self, history: Tuple[int, ...]) -> int:
        hid = 0
        for action in history:
            hid |= 1 << action
        return hid

    def _strategy(self, pid: int, hid: int, hand_idx: int) -> np.ndarray:
        regrets = self.R0[hid, hand_idx] if pid == 0 else self.R1[hid, hand_idx]
        mask = self.legal_mask[hid]
        positive = np.maximum(regrets, 0.0) * mask
        total = float(positive.sum())
        if total > 0.0:
            return positive / total

        strategy = np.zeros(self.A, dtype=self.dtype)
        strategy[mask] = 1.0 / int(mask.sum())
        return strategy

    def _sample_action(self, legal: Tuple[int, ...], strategy: np.ndarray) -> int:
        pick = self.rng.random()
        cumulative = 0.0
        for action in legal:
            cumulative += float(strategy[self._action_col(action)])
            if pick <= cumulative:
                return action
        return legal[-1]

    def _deal(self) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        deck = list(generate_deck(self.spec))
        self.rng.shuffle(deck)
        n = self.spec.hand_size
        return tuple(sorted(deck[:n])), tuple(sorted(deck[n : 2 * n]))

    def _traverse(
        self,
        history: Tuple[int, ...],
        p1_hand: Tuple[int, ...],
        p2_hand: Tuple[int, ...],
        traverser: int,
    ) -> float:
        if history and history[-1] == CALL:
            winner = resolve_call_winner(self.spec, history, p1_hand, p2_hand)
            return 1.0 if winner == ("P1" if traverser == 0 else "P2") else -1.0

        to_play = len(history) & 1
        hand = p1_hand if to_play == 0 else p2_hand
        hand_idx = self.hand_to_idx[hand]
        hid = self._history_to_hid(history)
        legal = self.legal_actions[hid]
        strategy = self._strategy(to_play, hid, hand_idx)

        if to_play == traverser:
            action_values = np.zeros(self.A, dtype=self.dtype)
            node_value = 0.0
            for action in legal:
                col = self._action_col(action)
                value = self._traverse(history + (action,), p1_hand, p2_hand, traverser)
                action_values[col] = value
                node_value += float(strategy[col]) * value

            regrets = self.R0 if traverser == 0 else self.R1
            cols = np.flatnonzero(self.legal_mask[hid])
            regrets[hid, hand_idx, cols] += action_values[cols] - node_value
            return node_value

        strategy_sum = self.SS0 if to_play == 0 else self.SS1
        weight = 1.0 if self.strategy_weighting == "uniform" else float(self.iteration)
        strategy_sum[hid, hand_idx] += weight * strategy
        action = self._sample_action(legal, strategy)
        return self._traverse(history + (action,), p1_hand, p2_hand, traverser)

    def run_iteration(self, *, traversals_per_player: int = 100) -> Dict[str, float | int]:
        self.iteration += 1
        start = time.perf_counter()
        for traverser in (0, 1):
            for _ in range(traversals_per_player):
                p1_hand, p2_hand = self._deal()
                self._traverse((), p1_hand, p2_hand, traverser)
        return {
            "iteration": self.iteration,
            "traversals_per_player": traversals_per_player,
            "traversal_s": time.perf_counter() - start,
        }

    def current_policy(self) -> DenseTabularPolicy:
        policy = DenseTabularPolicy(self.spec)
        for hid in range(self.H):
            pid = policy.pid_to_act(hid)
            for hand_idx in range(self.n_hands):
                policy.S[hid, hand_idx] = self._strategy(pid, hid, hand_idx)
        policy.recompute_likelihoods()
        return policy

    def average_policy(self) -> DenseTabularPolicy:
        policy = DenseTabularPolicy(self.spec)
        for hid in range(self.H):
            source = self.SS0 if policy.pid_to_act(hid) == 0 else self.SS1
            rows = source[hid] * self.legal_mask[hid]
            sums = rows.sum(axis=1)
            use_rows = sums > 0.0
            if np.any(use_rows):
                policy.S[hid, use_rows] = rows[use_rows] / sums[use_rows][:, None]
        policy.recompute_likelihoods()
        return policy
