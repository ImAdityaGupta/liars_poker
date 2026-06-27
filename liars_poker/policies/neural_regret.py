from __future__ import annotations

import random
from typing import Dict, Iterable, Sequence, Tuple

import numpy as np
import torch

from liars_poker.core import GameSpec
from liars_poker.infoset import CALL, InfoSet

from .base import Policy
from .neural import InfosetEncoder, NeuralMLP, _decode_spec, _encode_spec


class NeuralRegretMatchingPolicy(Policy):
    """Playable current-strategy policy backed by role-specific regret networks.

    The action distribution is CFR+ clipped-regret matching:
    positive predicted regrets are normalized over legal actions; if all legal
    predictions are non-positive, the policy falls back to uniform legal play.
    """

    POLICY_KIND = "NeuralRegretMatchingPolicy"
    POLICY_VERSION = 1

    def __init__(
        self,
        spec: GameSpec,
        *,
        hidden_sizes: Sequence[int] = (2048, 2048),
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

    @classmethod
    def from_models(
        cls,
        spec: GameSpec,
        models: Sequence[NeuralMLP],
        *,
        hidden_sizes: Sequence[int],
        device: str | torch.device = "cpu",
    ) -> "NeuralRegretMatchingPolicy":
        if len(models) != 2:
            raise ValueError("Expected exactly two regret models.")
        policy = cls(spec, hidden_sizes=hidden_sizes, device=device)
        policy.model_p1.load_state_dict(
            {
                key: value.detach().to(policy.device).clone()
                for key, value in models[0].state_dict().items()
            }
        )
        policy.model_p2.load_state_dict(
            {
                key: value.detach().to(policy.device).clone()
                for key, value in models[1].state_dict().items()
            }
        )
        return policy.eval()

    def eval(self) -> "NeuralRegretMatchingPolicy":
        self.model_p1.eval()
        self.model_p2.eval()
        return self

    def _model(self, pid: int) -> NeuralMLP:
        return self.model_p1 if pid == 0 else self.model_p2

    @staticmethod
    def _action_col(action: int) -> int:
        return 0 if action == CALL else action + 1

    def regret_values(
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

    def _legal_probs(
        self,
        *,
        pid: int,
        hand: Tuple[int, ...],
        history: Tuple[int, ...],
        legal: Sequence[int],
    ) -> np.ndarray:
        values = self.regret_values(pid=pid, hand=hand, history=history)
        cols = [self._action_col(action) for action in legal]
        positive = np.maximum(values[cols], 0.0)
        total = float(positive.sum())
        if total > 0.0:
            return positive / total
        return np.full(len(cols), 1.0 / len(cols), dtype=np.float32)

    def action_probs(self, infoset: InfoSet) -> Dict[int, float]:
        legal = self._legal_actions(infoset)
        if not legal:
            return {}
        probs = self._legal_probs(
            pid=infoset.pid,
            hand=infoset.hand,
            history=infoset.history,
            legal=legal,
        )
        return {action: float(prob) for action, prob in zip(legal, probs)}

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
        probs = self._legal_probs(pid=pid, hand=hand, history=history, legal=legal)
        pick = rng.random()
        cumulative = 0.0
        for action, prob in zip(legal, probs):
            cumulative += float(prob)
            if pick <= cumulative:
                return action
        return legal[-1]

    def __repr__(self) -> str:
        return (
            "NeuralRegretMatchingPolicy("
            f"spec={self.spec}, hidden_sizes={self.hidden_sizes}, "
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
    ) -> "NeuralRegretMatchingPolicy":
        _ = (blob_prefix, children)
        policy = cls(
            _decode_spec(payload["spec"]),
            hidden_sizes=tuple(payload.get("hidden_sizes", (2048, 2048))),
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
