from __future__ import annotations

import os
import pickle
from typing import Dict, Optional, Tuple

from liars_poker.core import GameSpec
from liars_poker.env import rules_for_spec
from liars_poker.infoset import InfoSet

from .base import Policy


class TabularPolicy(Policy):
    POLICY_KIND = "TabularPolicy"

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

    def store_efficiently(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        spec = self._require_rules().spec
        payload = {
            "kind": self.POLICY_KIND,
            "spec": spec,
            "probs": self.probs,
            "values": self._state_value,
            "visits": self._state_visits,
        }
        path = os.path.join(directory, self.POLICY_BINARY_FILENAME)
        with open(path, "wb") as handle:
            pickle.dump(payload, handle)

    @classmethod
    def load_efficiently(cls, directory: str) -> Tuple["TabularPolicy", GameSpec]:
        path = os.path.join(directory, cls.POLICY_BINARY_FILENAME)
        with open(path, "rb") as handle:
            payload = pickle.load(handle)
        policy, spec = cls._from_serialized(payload, directory)
        policy.bind_rules(rules_for_spec(spec))
        return policy, spec

    @classmethod
    def _from_serialized(cls, payload, directory: str) -> Tuple["TabularPolicy", GameSpec]:
        spec: GameSpec = payload["spec"]
        policy = cls()
        probs = payload.get("probs", {})
        policy.probs = {iset: dict(dist) for iset, dist in probs.items()}
        policy._state_value = dict(payload.get("values", {}))
        policy._state_visits = dict(payload.get("visits", {}))
        return policy, spec
