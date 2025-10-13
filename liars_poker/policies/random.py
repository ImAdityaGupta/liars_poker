from __future__ import annotations

from typing import Dict

from liars_poker.infoset import InfoSet

from .base import Policy


class RandomPolicy(Policy):
    """Uniform random over current legal actions."""

    def action_probs(self, infoset: InfoSet) -> Dict[int, float]:
        legal = self._legal_actions(infoset)
        n = len(legal)
        if n == 0:
            return {}
        prob = 1.0 / n
        return {action: prob for action in legal}

