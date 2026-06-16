from __future__ import annotations

import numpy as np

from liars_poker.core import GameSpec
from liars_poker.policies.tabular_dense import DenseTabularPolicy


class ExactDenseStrategyAverager:
    """Incremental reach-weighted average of dense behavioral strategies."""

    def __init__(self, spec: GameSpec, *, dtype: np.dtype = np.float64) -> None:
        self.spec = spec
        template = DenseTabularPolicy(spec)
        self.strategy_sum = np.zeros(template.S.shape, dtype=dtype)
        self.reach_sum = np.zeros(template.S.shape[:2], dtype=dtype)
        self.observations = 0

    def observe(self, policy: DenseTabularPolicy, *, weight: float = 1.0) -> None:
        if policy.spec != self.spec:
            raise ValueError("Dense policy spec does not match averager spec.")
        actor_is_p0 = (policy.popcount & 1)[:, None] == 0
        actor_reach = np.where(actor_is_p0, policy.L_pid0, policy.L_pid1)
        weighted_reach = float(weight) * actor_reach
        self.strategy_sum += weighted_reach[:, :, None] * policy.S
        self.reach_sum += weighted_reach
        self.observations += 1

    def average_policy(self) -> DenseTabularPolicy:
        policy = DenseTabularPolicy(self.spec)
        use_rows = self.reach_sum > 0.0
        policy.S[use_rows] = (
            self.strategy_sum[use_rows] / self.reach_sum[use_rows][:, None]
        ).astype(np.float32)
        policy.recompute_likelihoods()
        return policy
