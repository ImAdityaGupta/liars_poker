from __future__ import annotations

import random
from typing import Dict, Tuple

from liars_poker.infoset import InfoSet

from .base import Policy


class RandomPolicy(Policy):
    POLICY_KIND = "RandomPolicy"

    """Uniform random over current legal actions."""

    def action_probs(self, infoset: InfoSet) -> Dict[int, float]:
        legal = self._legal_actions(infoset)
        n = len(legal)
        if n == 0:
            return {}
        prob = 1.0 / n
        return {action: prob for action in legal}

    def sample_action_fast(
        self,
        *,
        pid: int,
        hand: Tuple[int, ...],
        history: Tuple[int, ...],
        legal: Tuple[int, ...],
        rng: random.Random,
    ) -> int:
        _ = (pid, hand, history)
        if not legal:
            raise ValueError("Cannot sample from empty policy distribution.")
        return legal[rng.randrange(len(legal))]

    # --- Serialization ---

    def to_payload(self) -> Tuple[Dict, Dict[str, object]]:
        # Stateless; only identifier matters.
        return {"kind": self.POLICY_KIND, "version": self.POLICY_VERSION}, {}

    @classmethod
    def from_payload(cls, payload, *, blob_prefix, blobs, children) -> "RandomPolicy":
        _ = (payload, blob_prefix, blobs, children)
        return cls()

    def iter_children(self):
        return ()
