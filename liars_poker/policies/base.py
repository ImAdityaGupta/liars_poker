from __future__ import annotations

import random
from typing import ClassVar, Dict, Iterable, Optional, Tuple

from liars_poker.infoset import InfoSet
from liars_poker.env import Rules


class Policy:
    """Policy abstraction queried only on its own turn."""

    POLICY_KIND: ClassVar[str]  # subclasses must set
    POLICY_VERSION: ClassVar[int] = 1

    def __init__(self) -> None:
        self._rules: "Rules | None" = None

    # --- Core gameplay API ---

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

    # --- Serialization hooks (to be consumed by liars_poker.serialization) ---

    def to_payload(self) -> Tuple[Dict, Dict[str, "object"]]:
        """Return (data_payload, blobs) for this policy.

        - data_payload: JSON-serializable structure describing the policy.
        - blobs: mapping of local blob names -> numpy arrays / binary-friendly buffers.
        """

        raise NotImplementedError

    @classmethod
    def from_payload(
        cls,
        payload: Dict,
        *,
        blob_prefix: str,
        blobs: Dict[str, "object"],
        children: Iterable["Policy"],
    ) -> "Policy":
        """Rehydrate a policy from its payload, scoped blobs, and already-instantiated children."""

        raise NotImplementedError

    def iter_children(self) -> Iterable["Policy | Tuple[str, \"Policy\"]"]:
        """Return any child policies (for composites like mixtures).

        Each entry may be either a Policy or a (label, Policy) tuple to influence blob-prefix naming.
        Default is no children.
        """

        return ()

    # --- Convenience persistence wrappers ---

    def save(self, directory: str) -> None:
        from liars_poker.serialization import save_policy

        save_policy(self, directory)

    @staticmethod
    def load(directory: str) -> Tuple["Policy", "object"]:
        from liars_poker.serialization import load_policy

        return load_policy(directory)
