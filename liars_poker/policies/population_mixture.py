from __future__ import annotations

import random
from typing import Dict, List, Sequence, Tuple

import numpy as np

from liars_poker.infoset import InfoSet
from .base import Policy


class PopulationMixturePolicy(Policy):
    """Commit-once mixture over a population of policies."""

    POLICY_KIND = "PopulationMixturePolicy"

    def __init__(self, policies: Sequence[Policy], weights: Sequence[float]):
        super().__init__()
        if len(policies) != len(weights):
            raise ValueError("Policies and weights must have the same length.")
        if not policies:
            raise ValueError("PopulationMixturePolicy requires at least one policy.")

        w = np.asarray(weights, dtype=float)
        if np.any(w < 0):
            raise ValueError("PopulationMixturePolicy weights must be non-negative.")
        total = float(w.sum())
        if total <= 0.0:
            raise ValueError("PopulationMixturePolicy requires positive total weight.")
        self.weights = w / total
        self.policies = list(policies)

        self._rng = random.Random()
        self._choice: int | None = None

    def bind_rules(self, rules) -> None:  # type: ignore[override]
        super().bind_rules(rules)
        for policy in self.policies:
            policy.bind_rules(rules)

    def begin_episode(self, rng: random.Random | None = None) -> None:
        if rng is not None:
            self._rng = rng
        pick = self._rng.random()
        cumulative = 0.0
        idx = len(self.weights) - 1
        for i, w in enumerate(self.weights):
            cumulative += float(w)
            if pick <= cumulative:
                idx = i
                break
        self._choice = idx
        self.policies[idx].begin_episode(self._rng)

    def action_probs(self, infoset: InfoSet) -> Dict[int, float]:
        if self._choice is None:
            raise RuntimeError("PopulationMixturePolicy: begin_episode() not yet called.")
        return self.policies[self._choice].action_probs(infoset)

    def prob_dist_at_infoset(self, infoset: InfoSet) -> Dict[int, float]:
        if self._choice is not None:
            return self.policies[self._choice].prob_dist_at_infoset(infoset)

        mixed: Dict[int, float] = {}
        for weight, policy in zip(self.weights, self.policies):
            dist = policy.prob_dist_at_infoset(infoset)
            for action, prob in dist.items():
                mixed[action] = mixed.get(action, 0.0) + float(weight) * prob

        total = sum(mixed.values())
        if total <= 0.0:
            return {}
        return {action: value / total for action, value in mixed.items()}

    def iter_children(self):
        return ()
