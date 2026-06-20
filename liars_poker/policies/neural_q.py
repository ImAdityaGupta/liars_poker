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


def compile_neural_q_to_dense(
    policy: NeuralQPolicy,
    *,
    batch_size: int = 65_536,
):
    """Compile a greedy role-specific Q policy into a dense deterministic policy."""

    from .tabular_dense import DenseTabularPolicy

    dense = DenseTabularPolicy(policy.spec)
    dense.S.fill(0.0)
    n_hands = len(dense.hands)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    histories_per_batch = max(1, int(batch_size) // n_hands)
    rank_dim = policy.spec.ranks
    input_dim = policy.encoder.input_dim
    action_dim = policy.encoder.action_dim
    claim_bits = np.arange(policy.encoder.k, dtype=np.int64)
    hand_features = policy.encoder.encode_hands(dense.hands, ())[:, :rank_dim]

    with torch.inference_mode():
        for pid in (0, 1):
            actor_hids = np.flatnonzero((dense.popcount & 1) == pid)
            model = policy._model(pid)
            for start in range(0, len(actor_hids), histories_per_batch):
                hids = actor_hids[start:start + histories_per_batch]
                history_bits = (
                    (hids[:, None].astype(np.int64) >> claim_bits[None, :]) & 1
                ).astype(np.float32)

                features = np.empty(
                    (len(hids), n_hands, input_dim),
                    dtype=np.float32,
                )
                features[:, :, :rank_dim] = hand_features[None, :, :rank_dim]
                features[:, :, rank_dim:] = history_bits[:, None, :]

                x = torch.from_numpy(features.reshape(-1, input_dim)).to(policy.device)
                q_values = model(x).reshape(len(hids), n_hands, action_dim)
                legal_mask = torch.from_numpy(dense.legal_mask[hids]).to(policy.device)
                best_cols = q_values.masked_fill(
                    ~legal_mask[:, None, :],
                    -torch.inf,
                ).argmax(dim=2)
                best_cols = best_cols.cpu().numpy()

                block = np.zeros(
                    (len(hids), n_hands, action_dim),
                    dtype=np.float32,
                )
                block[
                    np.arange(len(hids))[:, None],
                    np.arange(n_hands)[None, :],
                    best_cols,
                ] = 1.0
                dense.S[hids] = block

    dense.recompute_likelihoods()
    return dense
