from __future__ import annotations

import os
import pickle
import random
from typing import ClassVar, Dict, Optional, Tuple, Type

from liars_poker.core import GameSpec
from liars_poker.env import rules_for_spec
from liars_poker.infoset import InfoSet

if False:  # pragma: nocover - import guard for type checkers without runtime cost
    from liars_poker.env import Rules


class Policy:
    POLICY_BINARY_FILENAME: ClassVar[str] = "policy.bin"
    POLICY_KIND: ClassVar[Optional[str]] = None
    _registry: ClassVar[Dict[str, Type["Policy"]]] = {}

    """Policy abstraction queried only on its own turn."""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        kind = getattr(cls, "POLICY_KIND", None)
        if kind:
            Policy._registry[kind] = cls

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

    def store_efficiently(self, directory: str) -> None:
        """Persist the policy and associated spec to `directory` using a binary format."""

        raise NotImplementedError

    @classmethod
    def load_efficiently(cls, directory: str) -> Tuple["Policy", GameSpec]:
        """Load a policy and spec from `directory`."""

        raise NotImplementedError

    @classmethod
    def _from_serialized(cls, payload: Dict, directory: str) -> Tuple["Policy", GameSpec]:  # pragma: nocover - abstract helper
        raise NotImplementedError

    @staticmethod
    def load_policy(directory: str) -> Tuple["Policy", GameSpec]:
        path = os.path.join(directory, Policy.POLICY_BINARY_FILENAME)
        with open(path, "rb") as handle:
            payload = pickle.load(handle)
        kind = payload.get("kind")
        if kind is None:
            raise ValueError("Serialized policy missing 'kind'")
        policy_cls = Policy._registry.get(kind)
        if policy_cls is None:
            raise ValueError(f"Unknown policy kind: {kind}")
        policy, spec = policy_cls._from_serialized(payload, directory)
        policy.bind_rules(rules_for_spec(spec))
        return policy, spec
