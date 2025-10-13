from __future__ import annotations

import random
from typing import Dict, Tuple

from liars_poker.infoset import InfoSet

if False:  # pragma: nocover - import guard for type checkers without runtime cost
    from liars_poker.env import Rules


class Policy:
    """Policy abstraction queried only on its own turn."""

    def __init__(self) -> None:
        self._rules: "Rules | None" = None

    def bind_rules(self, rules: "Rules") -> None:
        """Associate the policy with static game rules."""

        self._rules = rules

    def begin_episode(self, rng: random.Random | None = None) -> None:
        """Hook invoked exactly once after each env.reset."""

        _ = rng

    def action_probs(self, infoset: InfoSet) -> Dict[int, float]:
        raise NotImplementedError

    def prob_dist_at_infoset(self, infoset: InfoSet) -> Dict[int, float]:
        """Return the conditional distribution at this infoset.

        Default implementation delegates to action_probs; subclasses may override
        (e.g., mixtures that incorporate posterior reasoning).
        """

        return self.action_probs(infoset)

    def sample(self, infoset: InfoSet, rng: random.Random) -> int:
        """Sample an action according to action_probs at the infoset."""

        dist = self.action_probs(infoset)
        if not dist:
            raise ValueError("Cannot sample from empty policy distribution.")
        actions = sorted(dist.keys())
        probs = [dist[a] for a in actions]
        total = sum(probs)
        if total <= 0:
            raise ValueError("Policy distribution must have positive mass.")
        # Normalize for safety
        cumulative = 0.0
        pick = rng.random()
        for action, prob in zip(actions, (p / total for p in probs)):
            cumulative += prob
            if pick <= cumulative:
                return action
        return actions[-1]

    def _require_rules(self) -> "Rules":
        rules = self._rules
        if rules is None:
            raise RuntimeError("Policy must be bound to Rules before use.")
        return rules

    def _legal_actions(self, infoset: InfoSet) -> Tuple[int, ...]:
        return self._require_rules().legal_actions_for(infoset)

