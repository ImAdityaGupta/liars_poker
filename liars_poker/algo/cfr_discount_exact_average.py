"""Exact own-reach-weighted average observer for a batched tabular CFR+ trainer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from liars_poker.algo.cfr_discount_tabular import TabularDiscountTrainer
from liars_poker.core import card_rank
from liars_poker.policies.tabular_dense import DenseTabularPolicy


class ExactAverageTabularDiscountTrainer(TabularDiscountTrainer):
    """Keep the current regret update and accumulate an exact average."""

    def __init__(self, *args, average_weight_power: int = 1, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if average_weight_power not in (0, 1, 2):
            raise ValueError("average_weight_power must be 0, 1, or 2")
        self.average_weight_power = average_weight_power
        self.exact_average_sum: np.ndarray | None = None
        self._current_dense: DenseTabularPolicy | None = None
        self._dense_hand_order: np.ndarray | None = None
        self._dense_uniform: np.ndarray | None = None

    def current_policy_exact_dense(self) -> DenseTabularPolicy:
        """Read the table directly into a reusable dense policy for this spec."""
        if self.regret_table is None:
            raise ValueError("Direct dense compilation requires a regret table")
        if self._current_dense is None:
            dense = DenseTabularPolicy(self.spec)
            table_hands = {
                tuple(int(x) for x in row.tolist()): index
                for index, row in enumerate(self.hand_counts)
            }
            order = []
            for hand in dense.hands:
                counts = [0] * self.spec.ranks
                for card in hand:
                    counts[card_rank(card, self.spec) - 1] += 1
                order.append(table_hands[tuple(counts)])
            self._current_dense = dense
            self._dense_hand_order = np.asarray(order, dtype=np.int64)
            self._dense_uniform = (
                dense.legal_mask.astype(np.float32)
                / np.maximum(dense.legal_counts[:, None], 1)
            )
        dense = self._current_dense
        rows = self.regret_table.numpy().reshape(
            1 << self.encoder.k, self.n_rank_hands, self.encoder.action_dim,
        )
        values = np.maximum(rows[:, self._dense_hand_order, :], 0.0)
        values *= dense.legal_mask[:, None, :]
        totals = values.sum(axis=2, keepdims=True)
        dense.S[:] = self._dense_uniform[:, None, :]
        np.divide(values, totals, out=dense.S, where=totals > 0)
        dense.recompute_likelihoods()
        return dense

    def accumulate_exact_average(self) -> None:
        """Add the policy played at the start of the next outer iteration."""
        # The two players' own policies have not changed yet. In alternating
        # CFR+, updating player 0 before player 1 does not change player 1's
        # own reach or its strategy, so one compilation serves both players.
        current = self.current_policy_exact_dense()
        if self.exact_average_sum is None:
            self.exact_average_sum = np.zeros(current.S.shape, dtype=np.float64)
        own_reach = np.where(
            (current.popcount & 1)[:, None] == 0,
            current.L_pid0,
            current.L_pid1,
        )
        weight = float(self.iteration + 1) ** self.average_weight_power
        self.exact_average_sum += weight * own_reach[:, :, None] * current.S

    def exact_average_policy(self) -> DenseTabularPolicy:
        policy = DenseTabularPolicy(self.spec)
        if self.exact_average_sum is not None:
            totals = self.exact_average_sum.sum(axis=2, keepdims=True)
            np.divide(self.exact_average_sum, totals, out=policy.S,
                      where=totals > 0)
            policy.recompute_likelihoods()
        return policy

    def checkpoint_dict(self) -> dict:
        state = super().checkpoint_dict()
        state["exact_average_observer"] = {
            "format": 2,
            "average_weight_power": self.average_weight_power,
            "sum": (None if self.exact_average_sum is None
                    else torch.from_numpy(self.exact_average_sum)),
        }
        return state

    @classmethod
    def load_fork_checkpoint(cls, path: str | Path) -> "ExactAverageTabularDiscountTrainer":
        trainer = super().load_fork_checkpoint(path)
        state = torch.load(path, map_location="cpu", weights_only=False)
        observer = state.get("exact_average_observer")
        if observer is None or observer.get("format") not in (1, 2):
            raise ValueError("Missing exact average observer")
        trainer.average_weight_power = int(observer.get("average_weight_power", 1))
        if trainer.average_weight_power not in (0, 1, 2):
            raise ValueError("Invalid exact average weight power")
        total = observer["sum"]
        if total is not None:
            expected = ((1 << trainer.encoder.k), trainer.n_rank_hands,
                        trainer.encoder.action_dim)
            if total.dtype != torch.float64 or tuple(total.shape) != expected:
                raise ValueError("Invalid exact average array")
            trainer.exact_average_sum = total.numpy().copy()
        return trainer
