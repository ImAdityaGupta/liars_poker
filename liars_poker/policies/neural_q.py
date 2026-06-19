from __future__ import annotations

import random
from typing import Dict, Iterable, Sequence, Tuple

import numpy as np
import torch

from liars_poker.core import GameSpec
from liars_poker.infoset import CALL, InfoSet

from .base import Policy
from .neural import InfosetEncoder, NeuralMLP, _decode_spec, _encode_spec


class NeuralQPolicy(Policy):
    """Greedy playable policy backed by role-specific Q-networks."""

    POLICY_KIND = "NeuralQPolicy"
    POLICY_VERSION = 1

    def __init__(
        self,
        spec: GameSpec,
        *,
        hidden_sizes: Sequence[int] = (512, 512),
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        from liars_poker.env import rules_for_spec

        self.spec = spec
        self.rules = rules_for_spec(spec)
        self.bind_rules(self.rules)
        self.encoder = InfosetEncoder(spec)
        self.hidden_sizes = tuple(int(size) for size in hidden_sizes)
        self.device = torch.device(device)
        self.model_p1 = NeuralMLP(
            self.encoder.input_dim,
            self.encoder.action_dim,
            self.hidden_sizes,
        ).to(self.device)
        self.model_p2 = NeuralMLP(
            self.encoder.input_dim,
            self.encoder.action_dim,
            self.hidden_sizes,
        ).to(self.device)
        self.eval()

    def eval(self) -> "NeuralQPolicy":
        self.model_p1.eval()
        self.model_p2.eval()
        return self

    def _model(self, pid: int) -> NeuralMLP:
        return self.model_p1 if pid == 0 else self.model_p2

    @staticmethod
    def _action_col(action: int) -> int:
        return 0 if action == CALL else action + 1

    def q_values(
        self,
        *,
        pid: int,
        hand: Tuple[int, ...],
        history: Tuple[int, ...],
    ) -> np.ndarray:
        features = torch.from_numpy(self.encoder.encode(hand, history)).to(self.device)
        with torch.inference_mode():
            values = self._model(pid)(features)
        return values.float().cpu().numpy()

    def action_probs(self, infoset: InfoSet) -> Dict[int, float]:
        legal = self._legal_actions(infoset)
        if not legal:
            return {}
        values = self.q_values(
            pid=infoset.pid,
            hand=infoset.hand,
            history=infoset.history,
        )
        best = max(legal, key=lambda action: values[self._action_col(action)])
        return {action: float(action == best) for action in legal}

    def sample_action_fast(
        self,
        *,
        pid: int,
        hand: Tuple[int, ...],
        history: Tuple[int, ...],
        legal: Tuple[int, ...],
        rng: random.Random,
    ) -> int:
        _ = rng
        if not legal:
            raise ValueError("Cannot sample from empty policy distribution.")
        values = self.q_values(pid=pid, hand=hand, history=history)
        return max(legal, key=lambda action: values[self._action_col(action)])

    def __repr__(self) -> str:
        return (
            f"NeuralQPolicy(spec={self.spec}, hidden_sizes={self.hidden_sizes}, "
            f"device='{self.device.type}')"
        )

    def to_payload(self) -> Tuple[Dict, Dict[str, object]]:
        payload = {
            "kind": self.POLICY_KIND,
            "version": self.POLICY_VERSION,
            "spec": _encode_spec(self.spec),
            "hidden_sizes": list(self.hidden_sizes),
        }
        blobs: Dict[str, object] = {}
        for prefix, model in (("p1", self.model_p1), ("p2", self.model_p2)):
            for name, tensor in model.state_dict().items():
                blobs[f"{prefix}::{name}"] = tensor.detach().cpu().numpy()
        return payload, blobs

    @classmethod
    def from_payload(
        cls,
        payload: Dict,
        *,
        blob_prefix: str,
        blobs: Dict[str, object],
        children: Iterable[Policy],
    ) -> "NeuralQPolicy":
        _ = (blob_prefix, children)
        policy = cls(
            _decode_spec(payload["spec"]),
            hidden_sizes=tuple(payload.get("hidden_sizes", (512, 512))),
            device="cpu",
        )
        for prefix, model in (("p1", policy.model_p1), ("p2", policy.model_p2)):
            state = {
                name: torch.from_numpy(np.asarray(blobs[f"{prefix}::{name}"]))
                for name in model.state_dict()
            }
            model.load_state_dict(state)
        return policy.eval()

    def iter_children(self):
        return ()
