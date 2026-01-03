from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import numpy as np

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

    def sample_action_fast(
        self,
        *,
        pid: int,
        hand: Tuple[int, ...],
        history: Tuple[int, ...],
        legal: Tuple[int, ...],
        rng: random.Random,
    ) -> int:
        if not legal:
            raise ValueError("Cannot sample from empty policy distribution.")

        infoset = InfoSet(pid=pid, hand=hand, history=history)
        row = self.probs.get(infoset)
        if not row:
            return legal[rng.randrange(len(legal))]

        total = 0.0
        for action in legal:
            total += row.get(action, 0.0)
        if total <= 0.0:
            return legal[rng.randrange(len(legal))]

        pick = rng.random() * total
        cumulative = 0.0
        last_action = legal[-1]
        for action in legal:
            cumulative += row.get(action, 0.0)
            if pick <= cumulative:
                return action
        return last_action

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

    # --- Serialization ---

    def _encode_infoset(self, iset: InfoSet) -> Tuple[int, Tuple[int, ...], Tuple[int, ...]]:
        return (iset.pid, tuple(iset.hand), tuple(iset.history))

    def _decode_infoset(self, tup: Tuple[int, Tuple[int, ...], Tuple[int, ...]]) -> InfoSet:
        pid, hand, history = tup
        return InfoSet(pid=pid, hand=tuple(hand), history=tuple(history))

    def to_payload(self) -> Tuple[Dict, Dict[str, object]]:
        # Sort infosets deterministically by string representation of encoded key
        encoded_keys = [(self._encode_infoset(k), k) for k in self.probs.keys()]
        encoded_keys = sorted(encoded_keys, key=lambda kv: str(kv[0]))
        keys_only = [ek for ek, _ in encoded_keys]

        # Flatten actions/probs
        flat_actions: List[int] = []
        flat_probs: List[float] = []
        offsets = [0]
        for enc, orig in encoded_keys:
            dist = self.probs.get(orig, {})
            for action, prob in sorted(dist.items(), key=lambda kv: kv[0]):
                flat_actions.append(int(action))
                flat_probs.append(float(prob))
            offsets.append(len(flat_actions))

        # Values/visits aligned with keys (sentinel for missing)
        values_arr = np.full(len(keys_only), np.nan, dtype=float)
        visits_arr = np.full(len(keys_only), -1, dtype=int)
        for idx, (_, orig) in enumerate(encoded_keys):
            if orig in self._state_value:
                values_arr[idx] = float(self._state_value[orig])
            if orig in self._state_visits:
                visits_arr[idx] = int(self._state_visits[orig])

        blobs: Dict[str, object] = {
            "keys": np.array(keys_only, dtype=object),
            "actions": np.asarray(flat_actions, dtype=int),
            "probs": np.asarray(flat_probs, dtype=float),
            "offsets": np.asarray(offsets, dtype=int),
            "values": values_arr,
            "visits": visits_arr,
        }

        payload = {
            "kind": self.POLICY_KIND,
            "version": self.POLICY_VERSION,
            "counts": {"n_infosets": len(keys_only)},
        }
        return payload, blobs

    @classmethod
    def from_payload(
        cls,
        payload: Dict,
        *,
        blob_prefix: str,
        blobs: Dict[str, object],
        children,
    ) -> "TabularPolicy":
        _ = (blob_prefix, children)
        policy = cls()

        keys_arr = blobs.get("keys")
        actions = np.asarray(blobs.get("actions", []), dtype=int)
        probs = np.asarray(blobs.get("probs", []), dtype=float)
        offsets = np.asarray(blobs.get("offsets", []), dtype=int)
        values_arr = np.asarray(blobs.get("values", []), dtype=float)
        visits_arr = np.asarray(blobs.get("visits", []), dtype=int)

        if keys_arr is None or offsets is None:
            raise ValueError("Missing keys/offsets blobs for TabularPolicy.")

        n = len(keys_arr)
        for idx in range(n):
            enc = tuple(keys_arr[idx].tolist() if hasattr(keys_arr[idx], "tolist") else keys_arr[idx])  # type: ignore[assignment]
            iset = policy._decode_infoset(enc)  # type: ignore[arg-type]
            start = offsets[idx]
            end = offsets[idx + 1]
            slice_actions = actions[start:end] if end > start else []
            slice_probs = probs[start:end] if end > start else []
            dist = {int(a): float(p) for a, p in zip(slice_actions, slice_probs)}
            if dist:
                policy.probs[iset] = dist

            if idx < len(values_arr) and not np.isnan(values_arr[idx]):
                policy._state_value[iset] = float(values_arr[idx])
            if idx < len(visits_arr) and visits_arr[idx] >= 0:
                policy._state_visits[iset] = int(visits_arr[idx])

        return policy

    def iter_children(self):
        return ()
