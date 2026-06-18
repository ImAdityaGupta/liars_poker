from __future__ import annotations

import random
from typing import Dict, Iterable, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from liars_poker.core import GameSpec, card_rank
from liars_poker.env import rules_for_spec
from liars_poker.infoset import CALL, InfoSet

from .base import Policy


def _encode_spec(spec: GameSpec) -> Dict[str, object]:
    return {
        "ranks": spec.ranks,
        "suits": spec.suits,
        "hand_size": spec.hand_size,
        "claim_kinds": list(spec.claim_kinds),
        "suit_symmetry": spec.suit_symmetry,
    }


def _decode_spec(data: Dict[str, object]) -> GameSpec:
    return GameSpec(
        ranks=int(data["ranks"]),
        suits=int(data["suits"]),
        hand_size=int(data["hand_size"]),
        claim_kinds=tuple(data["claim_kinds"]),
        suit_symmetry=bool(data["suit_symmetry"]),
    )


class InfosetEncoder:
    """Fixed-spec neural encoding: raw hand rank counts plus claim-history bits."""

    def __init__(self, spec: GameSpec) -> None:
        self.spec = spec
        self.rules = rules_for_spec(spec)
        self.k = len(self.rules.claims)
        self.input_dim = spec.ranks + self.k
        self.action_dim = self.k + 1

    def encode(self, hand: Tuple[int, ...], history: Tuple[int, ...]) -> np.ndarray:
        out = np.zeros(self.input_dim, dtype=np.float32)
        for card in hand:
            out[card_rank(card, self.spec) - 1] += 1.0
        offset = self.spec.ranks
        for action in history:
            if action != CALL:
                out[offset + action] = 1.0
        return out

    def encode_hands(self, hands: Sequence[Tuple[int, ...]], history: Tuple[int, ...]) -> np.ndarray:
        out = np.zeros((len(hands), self.input_dim), dtype=np.float32)
        for i, hand in enumerate(hands):
            for card in hand:
                out[i, card_rank(card, self.spec) - 1] += 1.0
        offset = self.spec.ranks
        for action in history:
            if action != CALL:
                out[:, offset + action] = 1.0
        return out


class NeuralMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_sizes: Sequence[int] = (128, 128)) -> None:
        super().__init__()
        sizes = (input_dim, *hidden_sizes, output_dim)
        layers: list[nn.Module] = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

        final = self.net[-1]
        if isinstance(final, nn.Linear):
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NeuralPolicy(Policy):
    """Playable fixed-spec policy backed by separate P1 and P2 strategy networks."""

    POLICY_KIND = "NeuralPolicy"
    POLICY_VERSION = 1

    def __init__(
        self,
        spec: GameSpec,
        *,
        hidden_sizes: Sequence[int] = (128, 128),
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
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

    def eval(self) -> "NeuralPolicy":
        self.model_p1.eval()
        self.model_p2.eval()
        return self

    def _model(self, pid: int) -> NeuralMLP:
        return self.model_p1 if pid == 0 else self.model_p2

    @staticmethod
    def _cols(legal: Sequence[int]) -> list[int]:
        return [0 if action == CALL else action + 1 for action in legal]

    def _legal_probs(
        self,
        *,
        pid: int,
        hand: Tuple[int, ...],
        history: Tuple[int, ...],
        legal: Sequence[int],
    ) -> np.ndarray:
        features = torch.from_numpy(self.encoder.encode(hand, history)).to(self.device)
        cols = self._cols(legal)
        with torch.no_grad():
            logits = self._model(pid)(features)
            probs = torch.softmax(logits[cols], dim=0)
        return probs.cpu().numpy()

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
            f"NeuralPolicy(spec={self.spec}, hidden_sizes={self.hidden_sizes}, "
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
    ) -> "NeuralPolicy":
        _ = (blob_prefix, children)
        policy = cls(
            _decode_spec(payload["spec"]),
            hidden_sizes=tuple(payload.get("hidden_sizes", (128, 128))),
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


def compile_neural_to_dense(
    policy: NeuralPolicy,
    *,
    batch_size: int = 16_384,
):
    """Compile a neural strategy to a dense policy in batched infoset blocks.

    ``batch_size`` is the approximate maximum number of (history, hand)
    infosets passed through a network at once.
    """

    from .tabular_dense import DenseTabularPolicy

    dense = DenseTabularPolicy(policy.spec)
    hands = dense.hands
    n_hands = len(hands)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    histories_per_batch = max(1, int(batch_size) // n_hands)
    rank_dim = policy.spec.ranks
    input_dim = policy.encoder.input_dim
    action_dim = policy.encoder.action_dim
    claim_bits = np.arange(policy.encoder.k, dtype=np.int64)
    hand_features = policy.encoder.encode_hands(hands, ())

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
                logits = model(x).reshape(len(hids), n_hands, action_dim)
                legal_mask = torch.from_numpy(dense.legal_mask[hids]).to(policy.device)
                logits = logits.masked_fill(~legal_mask[:, None, :], -torch.inf)
                dense.S[hids] = torch.softmax(logits, dim=2).cpu().numpy()

    dense.recompute_likelihoods()
    return dense
