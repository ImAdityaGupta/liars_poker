from __future__ import annotations

import random
from typing import Dict, List, Sequence, Tuple

from liars_poker.infoset import CALL, NO_CLAIM, InfoSet

from .base import Policy


class CommitOnceMixture(Policy):
    """Commit-once mixture that can expose posterior-aware distributions."""

    def __init__(
        self,
        policies: Sequence[Policy],
        weights: Sequence[float],
        rng: random.Random | None = None,
    ) -> None:
        super().__init__()
        if len(policies) != len(weights):
            raise ValueError("Policies and weights must have the same length.")

        filtered: List[Tuple[Policy, float]] = []
        total = 0.0
        for policy, weight in zip(policies, weights):
            if weight < 0:
                raise ValueError("CommitOnceMixture weights must be non-negative.")
            if weight > 0:
                filtered.append((policy, weight))
                total += weight

        if not filtered or total <= 0:
            raise ValueError("At least one positive weight is required.")

        self.policies: List[Policy] = [p for p, _ in filtered]
        self.weights: List[float] = [w / total for _, w in filtered]
        self._rng = rng or random.Random()
        self._choice: int | None = None

    def bind_rules(self, rules) -> None:  # type: ignore[override]
        super().bind_rules(rules)
        for policy in self.policies:
            policy.bind_rules(rules)

    def begin_episode(self, rng: random.Random | None = None) -> None:
        if rng is not None:
            self._rng = rng
        self._choice = self._weighted_choice()
        for policy in self.policies:
            policy.begin_episode(self._rng)

    def action_probs(self, infoset: InfoSet) -> Dict[int, float]:
        if self._choice is None:
            # If begin_episode not yet called, fall back to posterior-aware mixture.
            return self.prob_dist_at_infoset(infoset)
        return self.policies[self._choice].action_probs(infoset)

    def sample(self, infoset: InfoSet, rng: random.Random) -> int:
        if self._choice is None:
            raise RuntimeError("CommitOnceMixture: call begin_episode() after env.reset().")
        return self.policies[self._choice].sample(infoset, rng)

    def prob_dist_at_infoset(self, infoset: InfoSet) -> Dict[int, float]:
        posteriors = self._posterior_weights(infoset)
        mixed: Dict[int, float] = {}
        for weight, policy in zip(posteriors, self.policies):
            dist = policy.prob_dist_at_infoset(infoset)
            for action, prob in dist.items():
                mixed[action] = mixed.get(action, 0.0) + weight * prob

        total = sum(mixed.values())
        if total <= 0:
            legal = self._legal_actions(infoset)
            if not legal:
                return {}
            prob = 1.0 / len(legal)
            return {action: prob for action in legal}
        return {action: value / total for action, value in mixed.items()}

    def _weighted_choice(self) -> int:
        pick = self._rng.random()
        cumulative = 0.0
        for idx, weight in enumerate(self.weights):
            cumulative += weight
            if pick < cumulative:
                return idx
        return len(self.weights) - 1

    def _posterior_weights(self, infoset: InfoSet) -> List[float]:
        likelihoods: List[float] = []
        for policy, prior in zip(self.policies, self.weights):
            likelihood = self._sequence_likelihood(policy, infoset)
            likelihoods.append(prior * likelihood)

        total = sum(likelihoods)
        if total <= 0:
            return list(self.weights)
        return [value / total for value in likelihoods]

    def _sequence_likelihood(self, policy: Policy, infoset: InfoSet) -> float:
        history = infoset.history
        pid = infoset.pid
        likelihood = 1.0

        for turn in range(len(history)):
            if turn % 2 != pid:
                continue
            action = history[turn]
            prefix = history[:turn]
            partial_iset = InfoSet(
                pid=pid,
                last_idx=self._last_claim_idx_before(prefix),
                hand=infoset.hand,
                history=prefix,
            )
            dist = policy.prob_dist_at_infoset(partial_iset)
            likelihood *= dist.get(action, 0.0)
            if likelihood == 0.0:
                break
        return likelihood

    @staticmethod
    def _last_claim_idx_before(history: Tuple[int, ...]) -> int:
        for action in reversed(history):
            if action == CALL:
                continue
            return action
        return NO_CLAIM
