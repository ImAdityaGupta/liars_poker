from __future__ import annotations

import math
import random
from typing import Dict, Iterable, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from liars_poker.core import GameSpec
from liars_poker.env import rules_for_spec
from liars_poker.infoset import CALL, InfoSet

from .base import Policy
from .neural import InfosetEncoder, _decode_spec, _encode_spec


class ActionFeatureEncoder:
    """Semantic fixed-spec features for CALL and every claim action."""

    CLAIM_KINDS = (
        "RankHigh",
        "Pair",
        "TwoPair",
        "Trips",
        "FullHouse",
        "Quads",
    )

    def __init__(self, spec: GameSpec) -> None:
        self.spec = spec
        self.rules = rules_for_spec(spec)
        self.action_dim = len(self.rules.claims) + 1
        self.kind_offset = 1
        self.requirement_offset = self.kind_offset + len(self.CLAIM_KINDS)
        self.primary_rank_offset = self.requirement_offset + spec.ranks
        self.secondary_rank_offset = self.primary_rank_offset + spec.ranks
        self.order_offset = self.secondary_rank_offset + spec.ranks
        self.two_rank_offset = self.order_offset + 1
        self.feature_dim = self.two_rank_offset + 1
        self.feature_names = (
            ["is_call"]
            + [f"kind_{kind}" for kind in self.CLAIM_KINDS]
            + [f"requires_rank_{rank}" for rank in range(1, spec.ranks + 1)]
            + [f"primary_rank_{rank}" for rank in range(1, spec.ranks + 1)]
            + [f"secondary_rank_{rank}" for rank in range(1, spec.ranks + 1)]
            + ["claim_order", "uses_two_ranks"]
        )
        self.features = self._build()

    def _claim_ranks(self, kind: str, value: int) -> tuple[int, int | None, int, int]:
        if kind == "RankHigh":
            return value, None, 1, 0
        if kind == "Pair":
            return value, None, 2, 0
        if kind == "TwoPair":
            low, high = self.rules.two_pair_ranks[value]
            return high, low, 2, 2
        if kind == "Trips":
            return value, None, 3, 0
        if kind == "FullHouse":
            trip, pair = self.rules.full_house_ranks[value]
            return trip, pair, 3, 2
        if kind == "Quads":
            return value, None, 4, 0
        raise ValueError(f"Unsupported claim kind: {kind}")

    def _build(self) -> np.ndarray:
        out = np.zeros((self.action_dim, self.feature_dim), dtype=np.float32)
        out[0, 0] = 1.0
        kind_to_idx = {kind: idx for idx, kind in enumerate(self.CLAIM_KINDS)}
        denominator = max(1, len(self.rules.claims) - 1)

        for claim_id, (kind, value) in enumerate(self.rules.claims):
            col = claim_id + 1
            primary, secondary, primary_count, secondary_count = self._claim_ranks(
                kind,
                value,
            )
            out[col, self.kind_offset + kind_to_idx[kind]] = 1.0
            out[col, self.requirement_offset + primary - 1] = float(primary_count)
            out[col, self.primary_rank_offset + primary - 1] = 1.0
            if secondary is not None:
                out[col, self.requirement_offset + secondary - 1] = float(
                    secondary_count
                )
                out[col, self.secondary_rank_offset + secondary - 1] = 1.0
                out[col, self.two_rank_offset] = 1.0
            out[col, self.order_offset] = claim_id / denominator
        return out

    def tensor(self, device: str | torch.device) -> torch.Tensor:
        return torch.as_tensor(self.features, device=device)


class _Tower(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_sizes: Sequence[int],
        output_dim: int,
    ) -> None:
        super().__init__()
        sizes = (input_dim, *hidden_sizes, output_dim)
        layers: list[nn.Module] = []
        for idx in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[idx], sizes[idx + 1]))
            if idx < len(sizes) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ActionConditionedScorer(nn.Module):
    """Two-tower scalar scorer supporting pairwise and all-action evaluation."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        *,
        state_hidden_sizes: Sequence[int] = (512, 512),
        action_hidden_sizes: Sequence[int] = (128, 128),
        embedding_dim: int = 256,
    ) -> None:
        super().__init__()
        self.state_tower = _Tower(state_dim, state_hidden_sizes, embedding_dim)
        self.action_tower = _Tower(action_dim, action_hidden_sizes, embedding_dim)
        self.state_bias = nn.Linear(state_dim, 1)
        self.action_bias = nn.Linear(action_dim, 1)
        self.scale = math.sqrt(embedding_dim)
        self._cached_action_embedding: torch.Tensor | None = None
        self._cached_action_bias: torch.Tensor | None = None

    def train(self, mode: bool = True) -> "ActionConditionedScorer":
        super().train(mode)
        if mode:
            self.clear_action_cache()
        return self

    def clear_action_cache(self) -> None:
        self._cached_action_embedding = None
        self._cached_action_bias = None

    def initialize_action_neutral(self) -> None:
        final = self.action_tower.net[-1]
        if isinstance(final, nn.Linear):
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)
        nn.init.zeros_(self.action_bias.weight)
        nn.init.zeros_(self.action_bias.bias)
        self.clear_action_cache()

    def cache_actions(self, action_features: torch.Tensor) -> None:
        with torch.no_grad():
            self._cached_action_embedding = self.action_tower(
                action_features
            ).detach()
            self._cached_action_bias = self.action_bias(
                action_features
            ).squeeze(1).detach()

    def _score_encoded_actions(
        self,
        state_features: torch.Tensor,
        action_embedding: torch.Tensor,
        action_bias: torch.Tensor,
    ) -> torch.Tensor:
        state_embedding = self.state_tower(state_features)
        interaction = state_embedding @ action_embedding.T / self.scale
        return (
            interaction
            + self.state_bias(state_features)
            + action_bias[None, :]
        )

    def score_pairs(
        self,
        state_features: torch.Tensor,
        action_features: torch.Tensor,
    ) -> torch.Tensor:
        state_embedding = self.state_tower(state_features)
        action_embedding = self.action_tower(action_features)
        interaction = (state_embedding * action_embedding).sum(dim=1) / self.scale
        return (
            interaction
            + self.state_bias(state_features).squeeze(1)
            + self.action_bias(action_features).squeeze(1)
        )

    def score_selected(
        self,
        state_features: torch.Tensor,
        action_features: torch.Tensor,
        action_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Score one selected action per state after one action-table encoding."""

        action_embedding = self.action_tower(action_features)
        action_bias = self.action_bias(action_features).squeeze(1)
        state_embedding = self.state_tower(state_features)
        selected_embedding = action_embedding.index_select(0, action_ids)
        return (
            (state_embedding * selected_embedding).sum(dim=1) / self.scale
            + self.state_bias(state_features).squeeze(1)
            + action_bias.index_select(0, action_ids)
        )

    def score_all(
        self,
        state_features: torch.Tensor,
        action_features: torch.Tensor,
    ) -> torch.Tensor:
        action_embedding = self.action_tower(action_features)
        action_bias = self.action_bias(action_features).squeeze(1)
        return self._score_encoded_actions(
            state_features,
            action_embedding,
            action_bias,
        )

    def score_all_cached(self, state_features: torch.Tensor) -> torch.Tensor:
        if (
            self._cached_action_embedding is None
            or self._cached_action_bias is None
        ):
            raise RuntimeError("Action embeddings have not been cached.")
        return self._score_encoded_actions(
            state_features,
            self._cached_action_embedding,
            self._cached_action_bias,
        )

    def forward(
        self,
        state_features: torch.Tensor,
        action_features: torch.Tensor,
    ) -> torch.Tensor:
        return self.score_all(state_features, action_features)


class ActionConditionedQPolicy(Policy):
    """Greedy policy backed by role-specific action-conditioned Q scorers."""

    POLICY_KIND = "ActionConditionedQPolicy"
    POLICY_VERSION = 1

    def __init__(
        self,
        spec: GameSpec,
        *,
        state_hidden_sizes: Sequence[int] = (512, 512),
        action_hidden_sizes: Sequence[int] = (128, 128),
        embedding_dim: int = 256,
        device: str | torch.device = "cpu",
    ) -> None:
        Policy.__init__(self)
        self.spec = spec
        self.rules = rules_for_spec(spec)
        self.bind_rules(self.rules)
        self.encoder = InfosetEncoder(spec)
        self.action_encoder = ActionFeatureEncoder(spec)
        self.state_hidden_sizes = tuple(int(size) for size in state_hidden_sizes)
        self.action_hidden_sizes = tuple(int(size) for size in action_hidden_sizes)
        self.embedding_dim = int(embedding_dim)
        self.device = torch.device(device)
        self.action_features = self.action_encoder.tensor(self.device)
        self.model_p1 = self._new_model().to(self.device)
        self.model_p2 = self._new_model().to(self.device)
        self._initialize_action_neutral()
        self.eval()

    def _new_model(self) -> ActionConditionedScorer:
        return ActionConditionedScorer(
            self.encoder.input_dim,
            self.action_encoder.feature_dim,
            state_hidden_sizes=self.state_hidden_sizes,
            action_hidden_sizes=self.action_hidden_sizes,
            embedding_dim=self.embedding_dim,
        )

    def _initialize_action_neutral(self) -> None:
        for model in (self.model_p1, self.model_p2):
            model.initialize_action_neutral()

    def eval(self) -> "ActionConditionedQPolicy":
        self.model_p1.eval()
        self.model_p2.eval()
        self.model_p1.cache_actions(self.action_features)
        self.model_p2.cache_actions(self.action_features)
        return self

    def _model(self, pid: int) -> ActionConditionedScorer:
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
            values = self._model(pid).score_all_cached(features[None, :])[0]
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
        values = self.q_values(pid=pid, hand=hand, history=history)
        return max(legal, key=lambda action: values[self._action_col(action)])

    def to_payload(self) -> Tuple[Dict, Dict[str, object]]:
        payload = {
            "kind": self.POLICY_KIND,
            "version": self.POLICY_VERSION,
            "spec": _encode_spec(self.spec),
            "state_hidden_sizes": list(self.state_hidden_sizes),
            "action_hidden_sizes": list(self.action_hidden_sizes),
            "embedding_dim": self.embedding_dim,
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
    ) -> "ActionConditionedQPolicy":
        _ = (blob_prefix, children)
        policy = cls(
            _decode_spec(payload["spec"]),
            state_hidden_sizes=tuple(payload["state_hidden_sizes"]),
            action_hidden_sizes=tuple(payload["action_hidden_sizes"]),
            embedding_dim=int(payload["embedding_dim"]),
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


class ActionConditionedPolicy(ActionConditionedQPolicy):
    """Stochastic policy backed by role-specific action-conditioned logits."""

    POLICY_KIND = "ActionConditionedPolicy"
    POLICY_VERSION = 1

    def __init__(
        self,
        spec: GameSpec,
        *,
        state_hidden_sizes: Sequence[int] = (512, 512),
        action_hidden_sizes: Sequence[int] = (128, 128),
        embedding_dim: int = 256,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__(
            spec,
            state_hidden_sizes=state_hidden_sizes,
            action_hidden_sizes=action_hidden_sizes,
            embedding_dim=embedding_dim,
            device=device,
        )
        self._initialize_uniform()

    def _initialize_uniform(self) -> None:
        self._initialize_action_neutral()
        self.eval()

    def legal_logits(
        self,
        *,
        pid: int,
        hand: Tuple[int, ...],
        history: Tuple[int, ...],
        legal: Sequence[int],
    ) -> tuple[list[int], torch.Tensor]:
        cols = [self._action_col(action) for action in legal]
        features = torch.from_numpy(self.encoder.encode(hand, history)).to(self.device)
        logits = self._model(pid).score_all_cached(features[None, :])[0, cols]
        return cols, logits

    def action_probs(self, infoset: InfoSet) -> Dict[int, float]:
        legal = self._legal_actions(infoset)
        if not legal:
            return {}
        with torch.inference_mode():
            _, logits = self.legal_logits(
                pid=infoset.pid,
                hand=infoset.hand,
                history=infoset.history,
                legal=legal,
            )
            probs = torch.softmax(logits, dim=0).float().cpu().numpy()
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
        with torch.inference_mode():
            _, logits = self.legal_logits(
                pid=pid,
                hand=hand,
                history=history,
                legal=legal,
            )
            probs = torch.softmax(logits, dim=0).float().cpu().numpy()
        pick = rng.random()
        cumulative = 0.0
        for action, prob in zip(legal, probs):
            cumulative += float(prob)
            if pick <= cumulative:
                return action
        return legal[-1]


def compile_action_conditioned_to_dense(
    policy: ActionConditionedPolicy,
    *,
    batch_size: int = 65_536,
):
    """Compile an action-conditioned stochastic policy into a dense policy."""

    from .tabular_dense import DenseTabularPolicy

    dense = DenseTabularPolicy(policy.spec)
    n_hands = len(dense.hands)
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
                features[:, :, :rank_dim] = hand_features[None, :, :]
                features[:, :, rank_dim:] = history_bits[:, None, :]

                x = torch.from_numpy(features.reshape(-1, input_dim)).to(policy.device)
                logits = model.score_all_cached(x).reshape(
                    len(hids),
                    n_hands,
                    action_dim,
                )
                legal_mask = torch.from_numpy(dense.legal_mask[hids]).to(policy.device)
                logits = logits.masked_fill(~legal_mask[:, None, :], -torch.inf)
                dense.S[hids] = torch.softmax(logits, dim=2).float().cpu().numpy()

    dense.recompute_likelihoods()
    return dense


def compile_action_conditioned_q_to_dense(
    policy: ActionConditionedQPolicy,
    *,
    batch_size: int = 65_536,
):
    """Compile an action-conditioned greedy Q policy into a dense policy."""

    from .tabular_dense import DenseTabularPolicy

    dense = DenseTabularPolicy(policy.spec)
    dense.S.fill(0.0)
    n_hands = len(dense.hands)
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
                features[:, :, :rank_dim] = hand_features[None, :, :]
                features[:, :, rank_dim:] = history_bits[:, None, :]

                x = torch.from_numpy(features.reshape(-1, input_dim)).to(policy.device)
                q_values = model.score_all_cached(x).reshape(
                    len(hids),
                    n_hands,
                    action_dim,
                )
                legal_mask = torch.from_numpy(dense.legal_mask[hids]).to(policy.device)
                best_cols = q_values.masked_fill(
                    ~legal_mask[:, None, :],
                    -torch.inf,
                ).argmax(dim=2).cpu().numpy()

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
