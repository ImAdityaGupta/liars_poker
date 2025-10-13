from __future__ import annotations

from typing import Dict, Optional

from liars_poker.infoset import InfoSet

from .base import Policy


class TabularPolicy(Policy):
    """Dictionary backed policy with optional annotations."""

    def __init__(self) -> None:
        super().__init__()
        self.probs: Dict[InfoSet, Dict[int, float]] = {}
        self._state_value: Dict[InfoSet, float] = {}
        self._state_visits: Dict[InfoSet, int] = {}

    def set(self, infoset: InfoSet, dist: Dict[int, float]) -> None:
        self.probs[infoset] = dict(dist)

    def action_probs(self, infoset: InfoSet) -> Dict[int, float]:
        legal = self._legal_actions(infoset)
        if not legal:
            return {}
        if infoset not in self.probs:
            prob = 1.0 / len(legal)
            return {action: prob for action in legal}

        dist = {action: self.probs[infoset].get(action, 0.0) for action in legal}
        total = sum(dist.values())
        if total <= 0.0:
            prob = 1.0 / len(legal)
            return {action: prob for action in legal}
        return {action: value / total for action, value in dist.items()}

    def get_value(self, infoset: InfoSet) -> Optional[float]:
        return self._state_value.get(infoset)

    def get_visits(self, infoset: InfoSet) -> Optional[int]:
        return self._state_visits.get(infoset)

    def values(self) -> Dict[InfoSet, float]:
        return dict(self._state_value)

    def visits(self) -> Dict[InfoSet, int]:
        return dict(self._state_visits)

    def set_annotations(
        self,
        values: Dict[InfoSet, float] | None = None,
        visits: Dict[InfoSet, int] | None = None,
    ) -> None:
        if values is not None:
            self._state_value = dict(values)
        if visits is not None:
            self._state_visits = dict(visits)
