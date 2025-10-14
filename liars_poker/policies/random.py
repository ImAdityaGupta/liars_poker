from __future__ import annotations

import os
import pickle
from typing import Dict, Tuple

from liars_poker.core import GameSpec
from liars_poker.env import rules_for_spec
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

    def store_efficiently(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        spec = self._require_rules().spec
        payload = {"kind": self.POLICY_KIND, "spec": spec}
        path = os.path.join(directory, self.POLICY_BINARY_FILENAME)
        with open(path, "wb") as handle:
            pickle.dump(payload, handle)

    @classmethod
    def load_efficiently(cls, directory: str) -> Tuple["RandomPolicy", GameSpec]:
        path = os.path.join(directory, cls.POLICY_BINARY_FILENAME)
        with open(path, "rb") as handle:
            payload = pickle.load(handle)
        policy, spec = cls._from_serialized(payload, directory)
        policy.bind_rules(rules_for_spec(spec))
        return policy, spec

    @classmethod
    def _from_serialized(cls, payload, directory: str) -> Tuple["RandomPolicy", GameSpec]:
        spec: GameSpec = payload["spec"]
        policy = cls()
        return policy, spec
