from __future__ import annotations

from contextlib import nullcontext
import random
import time
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import torch

from liars_poker.algo.deep_cfr import (
    DeviceReservoirBuffer,
    ReservoirBuffer,
    _spec_from_dict,
    _spec_to_dict,
)
from liars_poker.core import GameSpec, generate_deck
from liars_poker.env import resolve_call_winner, rules_for_spec
from liars_poker.infoset import CALL, InfoSet
from liars_poker.policies.neural import InfosetEncoder, NeuralMLP, NeuralPolicy
from liars_poker.policies.tabular_dense import DenseTabularPolicy


class RecentBuffer:
    """Fixed-capacity FIFO-ish replay buffer for nonstationary CFR+ regret targets."""

    def __init__(self, capacity: int, input_dim: int, action_dim: int) -> None:
        self.capacity = int(capacity)
        self.input_dim = int(input_dim)
        self.action_dim = int(action_dim)
        self.features = np.empty((capacity, input_dim), dtype=np.float32)
        self.targets = np.empty((capacity, action_dim), dtype=np.float32)
        self.legal_masks = np.empty((capacity, action_dim), dtype=bool)
        self.weights = np.empty(capacity, dtype=np.float32)
        self.size = 0
        self.seen = 0
        self.cursor = 0

    def add(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        legal_mask: np.ndarray,
        weight: float,
        rng: random.Random | None = None,
    ) -> None:
        _ = rng
        idx = self.cursor
        self.features[idx] = features
        self.targets[idx] = targets
        self.legal_masks[idx] = legal_mask
        self.weights[idx] = weight

        self.seen += 1
        self.size = min(self.size + 1, self.capacity)
        self.cursor = (self.cursor + 1) % self.capacity

    def sample(self, batch_size: int, rng: random.Random) -> Tuple[np.ndarray, ...]:
        n = min(batch_size, self.size)
        indices = np.fromiter((rng.randrange(self.size) for _ in range(n)), dtype=np.int64)
        return (
            self.features[indices],
            self.targets[indices],
            self.legal_masks[indices],
            self.weights[indices],
        )

    def clear(self) -> None:
        self.size = 0
        self.seen = 0
        self.cursor = 0

    def state_dict(self) -> Dict[str, object]:
        return {
            "kind": "cpu_recent",
            "capacity": self.capacity,
            "input_dim": self.input_dim,
            "action_dim": self.action_dim,
            "features": self.features[: self.size].copy(),
            "targets": self.targets[: self.size].copy(),
            "legal_masks": self.legal_masks[: self.size].copy(),
            "weights": self.weights[: self.size].copy(),
            "size": self.size,
            "seen": self.seen,
            "cursor": self.cursor,
        }

    @classmethod
    def from_state_dict(cls, state: Dict[str, object]) -> "RecentBuffer":
        buffer = cls(
            int(state["capacity"]),
            int(state["input_dim"]),
            int(state["action_dim"]),
        )
        buffer.size = int(state["size"])
        buffer.seen = int(state["seen"])
        buffer.cursor = int(state["cursor"])
        buffer.features[: buffer.size] = np.asarray(state["features"])[: buffer.size]
        buffer.targets[: buffer.size] = np.asarray(state["targets"])[: buffer.size]
        buffer.legal_masks[: buffer.size] = np.asarray(state["legal_masks"])[: buffer.size]
        buffer.weights[: buffer.size] = np.asarray(state["weights"])[: buffer.size]
        return buffer


class DeviceRecentBuffer:
    """Fixed-capacity recent-record ring stored on one Torch device."""

    def __init__(
        self,
        capacity: int,
        input_dim: int,
        action_dim: int,
        device: str | torch.device,
    ) -> None:
        self.capacity = int(capacity)
        self.input_dim = int(input_dim)
        self.action_dim = int(action_dim)
        self.device = torch.device(device)
        self.features = torch.empty(
            (capacity, input_dim),
            dtype=torch.float32,
            device=self.device,
        )
        self.targets = torch.empty(
            (capacity, action_dim),
            dtype=torch.float32,
            device=self.device,
        )
        self.legal_masks = torch.empty(
            (capacity, action_dim),
            dtype=torch.bool,
            device=self.device,
        )
        self.weights = torch.empty(capacity, dtype=torch.float32, device=self.device)
        self.size = 0
        self.seen = 0
        self.cursor = 0

    def add(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
        legal_mask: torch.Tensor,
        weight: float,
        rng: random.Random | None = None,
    ) -> None:
        _ = rng
        self.add_many(
            features.unsqueeze(0),
            targets.unsqueeze(0),
            legal_mask.unsqueeze(0),
            weight,
        )

    def add_many(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
        legal_masks: torch.Tensor,
        weights: torch.Tensor | float,
        rng: random.Random | None = None,
    ) -> None:
        _ = rng
        features = features.to(self.device, dtype=torch.float32)
        targets = targets.to(self.device, dtype=torch.float32)
        legal_masks = legal_masks.to(self.device, dtype=torch.bool)
        n = int(features.shape[0])
        if n == 0:
            return

        if torch.is_tensor(weights):
            weights_t = weights.to(self.device, dtype=torch.float32)
            if weights_t.ndim == 0:
                weights_t = weights_t.expand(n)
        else:
            weights_t = torch.full(
                (n,),
                float(weights),
                dtype=torch.float32,
                device=self.device,
            )

        if n >= self.capacity:
            self.features.copy_(features[-self.capacity :])
            self.targets.copy_(targets[-self.capacity :])
            self.legal_masks.copy_(legal_masks[-self.capacity :])
            self.weights.copy_(weights_t[-self.capacity :])
            self.size = self.capacity
            self.seen += n
            self.cursor = 0
            return

        indices = (
            torch.arange(n, device=self.device, dtype=torch.long) + self.cursor
        ) % self.capacity
        self.features[indices] = features
        self.targets[indices] = targets
        self.legal_masks[indices] = legal_masks
        self.weights[indices] = weights_t
        self.cursor = (self.cursor + n) % self.capacity
        self.size = min(self.capacity, self.size + n)
        self.seen += n

    def sample(
        self,
        batch_size: int,
        rng: random.Random | None = None,
    ) -> Tuple[torch.Tensor, ...]:
        _ = rng
        n = min(int(batch_size), self.size)
        indices = torch.randint(self.size, (n,), device=self.device)
        return (
            self.features.index_select(0, indices),
            self.targets.index_select(0, indices),
            self.legal_masks.index_select(0, indices),
            self.weights.index_select(0, indices),
        )

    def clear(self) -> None:
        self.size = 0
        self.seen = 0
        self.cursor = 0

    def state_dict(self) -> Dict[str, object]:
        return {
            "kind": "device_recent",
            "capacity": self.capacity,
            "input_dim": self.input_dim,
            "action_dim": self.action_dim,
            "features": self.features[: self.size].detach().cpu(),
            "targets": self.targets[: self.size].detach().cpu(),
            "legal_masks": self.legal_masks[: self.size].detach().cpu(),
            "weights": self.weights[: self.size].detach().cpu(),
            "size": self.size,
            "seen": self.seen,
            "cursor": self.cursor,
        }

    @classmethod
    def from_state_dict(
        cls,
        state: Dict[str, object],
        *,
        device: str | torch.device,
    ) -> "DeviceRecentBuffer":
        buffer = cls(
            int(state["capacity"]),
            int(state["input_dim"]),
            int(state["action_dim"]),
            device,
        )
        buffer.size = int(state["size"])
        buffer.seen = int(state["seen"])
        buffer.cursor = int(state["cursor"])
        buffer.features[: buffer.size].copy_(
            torch.as_tensor(state["features"], device=buffer.device)
        )
        buffer.targets[: buffer.size].copy_(
            torch.as_tensor(state["targets"], device=buffer.device)
        )
        buffer.legal_masks[: buffer.size].copy_(
            torch.as_tensor(state["legal_masks"], device=buffer.device)
        )
        buffer.weights[: buffer.size].copy_(
            torch.as_tensor(state["weights"], device=buffer.device)
        )
        return buffer


class DeepCFRPlusTrainer:
    """External-sampling neural CFR+ with clipped cumulative-regret targets."""

    CHECKPOINT_VERSION = 2

    def __init__(
        self,
        spec: GameSpec,
        *,
        hidden_sizes: Sequence[int] | None = None,
        regret_hidden_sizes: Sequence[int] | None = None,
        strategy_hidden_sizes: Sequence[int] | None = None,
        device: str | torch.device = "cpu",
        seed: int = 0,
        regret_buffer_capacity: int = 100_000,
        strategy_buffer_capacity: int = 100_000,
        learning_rate: float = 1e-3,
        batch_size: int = 256,
        regret_train_steps: int = 100,
        strategy_train_steps: int = 50,
        strategy_weighting: str = "linear",
        regret_positive_weight: float = 0.5,
        validation_fraction: float = 0.0,
        validation_buffer_capacity: int = 10_000,
        traversal_backend: str = "recursive",
        traversal_batch_size: int = 256,
        traverser_action_sample_count: int | None = None,
        traverser_action_baseline: str = "none",
        device_replay: bool = False,
        fused_optimizer: bool | None = None,
        amp_dtype: str | None = None,
        compile_models: bool = False,
    ) -> None:
        self.spec = spec
        self.rules = rules_for_spec(spec)
        self.encoder = InfosetEncoder(spec)
        shared_hidden_sizes = (
            (256, 256)
            if hidden_sizes is None
            else tuple(int(size) for size in hidden_sizes)
        )
        self.regret_hidden_sizes = tuple(
            int(size)
            for size in (
                shared_hidden_sizes
                if regret_hidden_sizes is None
                else regret_hidden_sizes
            )
        )
        self.strategy_hidden_sizes = tuple(
            int(size)
            for size in (
                shared_hidden_sizes
                if strategy_hidden_sizes is None
                else strategy_hidden_sizes
            )
        )
        self.hidden_sizes = self.strategy_hidden_sizes
        self.device = torch.device(device)
        self.seed = int(seed)
        self.rng = random.Random(seed)
        self.validation_rng = random.Random(seed + 1_000_003)
        torch.manual_seed(seed)

        self.learning_rate = float(learning_rate)
        self.batch_size = int(batch_size)
        self.regret_train_steps = int(regret_train_steps)
        self.strategy_train_steps = int(strategy_train_steps)
        if strategy_weighting not in {"linear", "uniform"}:
            raise ValueError("strategy_weighting must be 'linear' or 'uniform'.")
        self.strategy_weighting = strategy_weighting
        self.regret_positive_weight = float(regret_positive_weight)
        self.validation_fraction = float(validation_fraction)
        self.validation_buffer_capacity = int(validation_buffer_capacity)
        if traversal_backend not in {"recursive", "gpu_native"}:
            raise ValueError("traversal_backend must be 'recursive' or 'gpu_native'.")
        self.traversal_backend = traversal_backend
        self.traversal_batch_size = int(traversal_batch_size)
        self.traverser_action_sample_count = (
            None
            if traverser_action_sample_count is None
            else int(traverser_action_sample_count)
        )
        if (
            self.traverser_action_sample_count is not None
            and self.traverser_action_sample_count <= 0
        ):
            raise ValueError("traverser_action_sample_count must be positive.")
        if traverser_action_baseline not in {"none", "call"}:
            raise ValueError(
                "traverser_action_baseline must be 'none' or 'call'."
            )
        self.traverser_action_baseline = traverser_action_baseline
        if (
            self.traversal_backend != "gpu_native"
            and (
                self.traverser_action_sample_count is not None
                or self.traverser_action_baseline != "none"
            )
        ):
            raise ValueError(
                "Traverser action sampling is only available with "
                "traversal_backend='gpu_native'."
            )
        self.device_replay = bool(device_replay)
        if self.traversal_backend == "gpu_native":
            self.device_replay = True
        self.fused_optimizer = (
            self.device.type == "cuda"
            if fused_optimizer is None
            else bool(fused_optimizer)
        )
        if amp_dtype not in {None, "float16", "bfloat16"}:
            raise ValueError("amp_dtype must be None, 'float16', or 'bfloat16'.")
        if amp_dtype is not None and self.device.type != "cuda":
            amp_dtype = None
        self.amp_dtype = amp_dtype
        self.compile_models = bool(compile_models)
        self.iteration = 0
        self._compiled_forwards: Dict[int, object] = {}
        self._gpu_traverser = None
        scaler_enabled = self.amp_dtype == "float16"
        try:
            self._grad_scaler = torch.amp.GradScaler(
                "cuda",
                enabled=scaler_enabled,
            )
        except TypeError:
            self._grad_scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)

        self.regret_nets = [
            NeuralMLP(
                self.encoder.input_dim,
                self.encoder.action_dim,
                self.regret_hidden_sizes,
            ).to(self.device)
            for _ in range(2)
        ]
        self.strategy_nets = [
            NeuralMLP(
                self.encoder.input_dim,
                self.encoder.action_dim,
                self.strategy_hidden_sizes,
            ).to(self.device)
            for _ in range(2)
        ]
        self.regret_optimizers = [
            self._make_optimizer(model)
            for model in self.regret_nets
        ]
        self.strategy_optimizers = [
            self._make_optimizer(model)
            for model in self.strategy_nets
        ]

        recent_cls = DeviceRecentBuffer if self.device_replay else RecentBuffer
        recent_args = (self.device,) if self.device_replay else ()
        reservoir_cls = DeviceReservoirBuffer if self.device_replay else ReservoirBuffer
        reservoir_args = (self.device,) if self.device_replay else ()
        self.regret_buffers = [
            recent_cls(
                regret_buffer_capacity,
                self.encoder.input_dim,
                self.encoder.action_dim,
                *recent_args,
            )
            for _ in range(2)
        ]
        self.strategy_buffers = [
            reservoir_cls(
                strategy_buffer_capacity,
                self.encoder.input_dim,
                self.encoder.action_dim,
                *reservoir_args,
            )
            for _ in range(2)
        ]
        self.regret_validation_buffers = [
            recent_cls(
                validation_buffer_capacity,
                self.encoder.input_dim,
                self.encoder.action_dim,
                *recent_args,
            )
            for _ in range(2)
        ]
        self.strategy_validation_buffers = [
            reservoir_cls(
                validation_buffer_capacity,
                self.encoder.input_dim,
                self.encoder.action_dim,
                *reservoir_args,
            )
            for _ in range(2)
        ]

    def _make_optimizer(self, model: NeuralMLP) -> torch.optim.Optimizer:
        kwargs = {"lr": self.learning_rate}
        if self.fused_optimizer and self.device.type == "cuda":
            kwargs["fused"] = True
        try:
            return torch.optim.Adam(model.parameters(), **kwargs)
        except TypeError:
            kwargs.pop("fused", None)
            return torch.optim.Adam(model.parameters(), **kwargs)

    def _autocast(self):
        if self.amp_dtype is None:
            return nullcontext()
        dtype = torch.float16 if self.amp_dtype == "float16" else torch.bfloat16
        return torch.autocast(device_type=self.device.type, dtype=dtype)

    def _forward(self, model: NeuralMLP, x: torch.Tensor) -> torch.Tensor:
        if not self.compile_models:
            return model(x)
        key = id(model)
        compiled = self._compiled_forwards.get(key)
        if compiled is None:
            compiled = torch.compile(model, dynamic=True)
            self._compiled_forwards[key] = compiled
        return compiled(x)

    def _synchronize(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @property
    def regret_net_p1(self) -> NeuralMLP:
        return self.regret_nets[0]

    @property
    def regret_net_p2(self) -> NeuralMLP:
        return self.regret_nets[1]

    @property
    def strategy_net_p1(self) -> NeuralMLP:
        return self.strategy_nets[0]

    @property
    def strategy_net_p2(self) -> NeuralMLP:
        return self.strategy_nets[1]

    @staticmethod
    def _action_col(action: int) -> int:
        return 0 if action == CALL else action + 1

    def _legal_mask(self, legal: Tuple[int, ...]) -> np.ndarray:
        mask = np.zeros(self.encoder.action_dim, dtype=bool)
        for action in legal:
            mask[self._action_col(action)] = True
        return mask

    def _regret_values_from_features(self, pid: int, features: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(features).to(self.device)
        with torch.inference_mode():
            with self._autocast():
                values = self._forward(self.regret_nets[pid], x)
            values = values.float().cpu().numpy()
        return values.astype(np.float32, copy=False)

    def _snapshot_regret_values_from_features(self, pid: int, features: np.ndarray) -> np.ndarray:
        # Collection and fitting never overlap. The live network is therefore
        # already frozen for the complete traversal phase.
        return self._regret_values_from_features(pid, features)

    def _strategy_from_features(
        self,
        pid: int,
        features: np.ndarray,
        legal: Tuple[int, ...],
        *,
        use_snapshot: bool = False,
    ) -> np.ndarray:
        _ = use_snapshot
        values = (
            self._snapshot_regret_values_from_features(pid, features)
            if use_snapshot
            else self._regret_values_from_features(pid, features)
        )
        strategy = np.zeros(self.encoder.action_dim, dtype=np.float32)
        cols = [self._action_col(action) for action in legal]
        positive = np.maximum(values[cols], 0.0)
        total = float(positive.sum())
        if total > 0.0:
            strategy[cols] = positive / total
        else:
            strategy[cols] = 1.0 / len(cols)
        return strategy

    def current_strategy(self, infoset: InfoSet) -> Dict[int, float]:
        legal = self.rules.legal_actions_for(infoset)
        features = self.encoder.encode(infoset.hand, infoset.history)
        strategy = self._strategy_from_features(infoset.pid, features, legal)
        return {action: float(strategy[self._action_col(action)]) for action in legal}

    def current_policy_dense(self, *, batch_size: int = 16_384) -> DenseTabularPolicy:
        """Compile the current clipped-regret strategy in batched infoset blocks."""

        dense = DenseTabularPolicy(self.spec)
        hands = dense.hands
        n_hands = len(hands)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        histories_per_batch = max(1, int(batch_size) // n_hands)
        rank_dim = self.spec.ranks
        input_dim = self.encoder.input_dim
        action_dim = self.encoder.action_dim
        claim_bits = np.arange(self.encoder.k, dtype=np.int64)
        hand_features = self.encoder.encode_hands(hands, ())

        with torch.inference_mode():
            for pid in (0, 1):
                actor_hids = np.flatnonzero((dense.popcount & 1) == pid)
                model = self.regret_nets[pid]
                for start in range(0, len(actor_hids), histories_per_batch):
                    hids = actor_hids[start : start + histories_per_batch]
                    history_bits = (
                        (
                            hids[:, None].astype(np.int64)
                            >> claim_bits[None, :]
                        )
                        & 1
                    ).astype(np.float32)

                    features = np.empty(
                        (len(hids), n_hands, input_dim),
                        dtype=np.float32,
                    )
                    features[:, :, :rank_dim] = hand_features[
                        None,
                        :,
                        :rank_dim,
                    ]
                    features[:, :, rank_dim:] = history_bits[:, None, :]

                    x = torch.from_numpy(
                        features.reshape(-1, input_dim)
                    ).to(self.device)
                    with self._autocast():
                        values = self._forward(model, x)
                    values = values.float().reshape(
                        len(hids),
                        n_hands,
                        action_dim,
                    )
                    legal_mask = torch.from_numpy(
                        dense.legal_mask[hids]
                    ).to(self.device)
                    positive = torch.relu(values) * legal_mask[:, None, :]
                    totals = positive.sum(dim=2, keepdim=True)
                    matched = positive / totals.clamp_min(1e-8)
                    fallback = legal_mask[:, None, :].float()
                    fallback = fallback / fallback.sum(
                        dim=2,
                        keepdim=True,
                    ).clamp_min(1.0)
                    dense.S[hids] = torch.where(
                        totals > 0.0,
                        matched,
                        fallback,
                    ).cpu().numpy()

        dense.recompute_likelihoods()
        return dense

    def regret_values(self, infoset: InfoSet) -> Dict[int, float]:
        legal = self.rules.legal_actions_for(infoset)
        x = torch.from_numpy(self.encoder.encode(infoset.hand, infoset.history)).to(self.device)
        with torch.inference_mode():
            with self._autocast():
                values = self._forward(self.regret_nets[infoset.pid], x)
            values = values.float().cpu().numpy()
        return {action: float(values[self._action_col(action)]) for action in legal}

    def _sample_action(self, legal: Tuple[int, ...], strategy: np.ndarray) -> int:
        pick = self.rng.random()
        cumulative = 0.0
        for action in legal:
            cumulative += float(strategy[self._action_col(action)])
            if pick <= cumulative:
                return action
        return legal[-1]

    def _add_regret_record(
        self,
        pid: int,
        features: np.ndarray,
        targets: np.ndarray,
        legal_mask: np.ndarray,
    ) -> None:
        if isinstance(self.regret_buffers[pid], DeviceRecentBuffer):
            features = torch.as_tensor(features, device=self.device)
            targets = torch.as_tensor(targets, device=self.device)
            legal_mask = torch.as_tensor(legal_mask, device=self.device)
        if self.validation_fraction > 0.0 and self.validation_rng.random() < self.validation_fraction:
            self.regret_validation_buffers[pid].add(features, targets, legal_mask, 1.0)
        else:
            self.regret_buffers[pid].add(features, targets, legal_mask, 1.0)

    def _add_strategy_record(
        self,
        pid: int,
        features: np.ndarray,
        strategy: np.ndarray,
        legal_mask: np.ndarray,
    ) -> None:
        weight = 1.0 if self.strategy_weighting == "uniform" else float(self.iteration)
        if isinstance(self.strategy_buffers[pid], DeviceReservoirBuffer):
            features = torch.as_tensor(features, device=self.device)
            strategy = torch.as_tensor(strategy, device=self.device)
            legal_mask = torch.as_tensor(legal_mask, device=self.device)
        if self.validation_fraction > 0.0 and self.validation_rng.random() < self.validation_fraction:
            self.strategy_validation_buffers[pid].add(
                features,
                strategy,
                legal_mask,
                weight,
                self.validation_rng,
            )
        else:
            self.strategy_buffers[pid].add(
                features,
                strategy,
                legal_mask,
                weight,
                self.rng,
            )

    def _add_device_records(
        self,
        training_buffer,
        validation_buffer,
        features: torch.Tensor,
        targets: torch.Tensor,
        legal_masks: torch.Tensor,
        weights: torch.Tensor | float,
    ) -> None:
        n = int(features.shape[0])
        if n == 0:
            return

        if torch.is_tensor(weights):
            weights_t = weights.to(self.device, dtype=torch.float32)
            if weights_t.ndim == 0:
                weights_t = weights_t.expand(n)
        else:
            weights_t = torch.full(
                (n,),
                float(weights),
                dtype=torch.float32,
                device=self.device,
            )

        if self.validation_fraction <= 0.0:
            training_buffer.add_many(features, targets, legal_masks, weights_t)
            return

        use_validation = torch.rand(n, device=self.device) < self.validation_fraction
        validation_buffer.add_many(
            features[use_validation],
            targets[use_validation],
            legal_masks[use_validation],
            weights_t[use_validation],
        )
        use_training = ~use_validation
        training_buffer.add_many(
            features[use_training],
            targets[use_training],
            legal_masks[use_training],
            weights_t[use_training],
        )

    def _deal(self) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        deck = list(generate_deck(self.spec))
        self.rng.shuffle(deck)
        n = self.spec.hand_size
        return tuple(sorted(deck[:n])), tuple(sorted(deck[n : 2 * n]))

    def _traverse(
        self,
        history: Tuple[int, ...],
        p1_hand: Tuple[int, ...],
        p2_hand: Tuple[int, ...],
        traverser: int,
    ) -> float:
        if history and history[-1] == CALL:
            winner = resolve_call_winner(self.spec, history, p1_hand, p2_hand)
            winner_pid = 0 if winner == "P1" else 1
            return 1.0 if winner_pid == traverser else -1.0

        to_play = len(history) & 1
        hand = p1_hand if to_play == 0 else p2_hand
        last_claim = history[-1] if history else None
        legal = self.rules.legal_actions_from_last(last_claim)
        features = self.encoder.encode(hand, history)
        legal_mask = self._legal_mask(legal)
        strategy = self._strategy_from_features(to_play, features, legal, use_snapshot=True)

        if to_play == traverser:
            action_values = np.zeros(self.encoder.action_dim, dtype=np.float32)
            node_value = 0.0
            for action in legal:
                col = self._action_col(action)
                value = self._traverse(history + (action,), p1_hand, p2_hand, traverser)
                action_values[col] = value
                node_value += float(strategy[col]) * value

            instant_regret = np.zeros(self.encoder.action_dim, dtype=np.float32)
            instant_regret[legal_mask] = action_values[legal_mask] - node_value

            old_scaled = np.maximum(self._snapshot_regret_values_from_features(traverser, features), 0.0)
            old_scaled[~legal_mask] = 0.0
            target = ((self.iteration - 1.0) / self.iteration) * old_scaled
            target += instant_regret / self.iteration
            target = np.maximum(target, 0.0).astype(np.float32)
            target[~legal_mask] = 0.0
            self._add_regret_record(traverser, features, target, legal_mask)
            return node_value

        self._add_strategy_record(to_play, features, strategy, legal_mask)
        action = self._sample_action(legal, strategy)
        return self._traverse(history + (action,), p1_hand, p2_hand, traverser)

    def _train_regret(self, pid: int) -> float:
        return self._train_model(
            self.regret_nets[pid],
            self.regret_optimizers[pid],
            self.regret_buffers[pid],
            self.regret_train_steps,
            strategy_loss=False,
        )

    def _train_strategy(self, pid: int) -> float:
        return self._train_model(
            self.strategy_nets[pid],
            self.strategy_optimizers[pid],
            self.strategy_buffers[pid],
            self.strategy_train_steps,
            strategy_loss=True,
        )

    def _train_model(
        self,
        model: NeuralMLP,
        optimizer: torch.optim.Optimizer,
        buffer,
        steps: int,
        *,
        strategy_loss: bool,
    ) -> float:
        if buffer.size == 0 or steps <= 0:
            return 0.0

        model.train()
        total_loss = torch.zeros((), dtype=torch.float32, device=self.device)
        for _ in range(steps):
            features, targets, masks, weights = buffer.sample(self.batch_size, self.rng)
            if torch.is_tensor(features):
                x = features
                y = targets
                mask = masks
                weight = weights
            else:
                x = torch.from_numpy(features).to(self.device, non_blocking=True)
                y = torch.from_numpy(targets).to(self.device, non_blocking=True)
                mask = torch.from_numpy(masks).to(self.device, non_blocking=True)
                weight = torch.from_numpy(weights).to(self.device, non_blocking=True)
            weight = weight / weight.mean().clamp_min(1e-8)

            with self._autocast():
                pred = self._forward(model, x)
            pred = pred.float()
            y = y.float()
            if strategy_loss:
                masked_logits = pred.masked_fill(~mask, -1e9)
                per_sample = -(y * torch.log_softmax(masked_logits, dim=1)).sum(dim=1)
            else:
                mask_float = mask.float()
                entry_weight = 1.0 + self.regret_positive_weight * (y > 1e-6).float()
                squared = (pred - y).square() * mask_float * entry_weight
                denom = (mask_float * entry_weight).sum(dim=1).clamp_min(1.0)
                per_sample = squared.sum(dim=1) / denom
            loss = (per_sample * weight).mean()

            optimizer.zero_grad(set_to_none=True)
            if self._grad_scaler.is_enabled():
                self._grad_scaler.scale(loss).backward()
                self._grad_scaler.step(optimizer)
                self._grad_scaler.update()
            else:
                loss.backward()
                optimizer.step()
            total_loss.add_(loss.detach())

        model.eval()
        return float((total_loss / steps).item())

    def _regret_matching_tensor(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        positive = torch.relu(values) * mask
        totals = positive.sum(dim=1, keepdim=True)
        matched = positive / totals.clamp_min(1e-8)
        fallback = mask / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        return torch.where(totals > 0.0, matched, fallback)

    def _validation_metrics_for(
        self,
        model: NeuralMLP,
        buffer,
        *,
        strategy_targets: bool,
        max_records: int,
    ) -> Dict[str, float]:
        size = min(buffer.size, max_records)
        if size == 0:
            return {"records": 0}

        if isinstance(buffer, (DeviceRecentBuffer, DeviceReservoirBuffer)):
            x = buffer.features[:size]
            targets = buffer.targets[:size]
            mask = buffer.legal_masks[:size]
            weights = buffer.weights[:size]
        else:
            x = torch.from_numpy(buffer.features[:size]).to(self.device)
            targets = torch.from_numpy(buffer.targets[:size]).to(self.device)
            mask = torch.from_numpy(buffer.legal_masks[:size]).to(self.device)
            weights = torch.from_numpy(buffer.weights[:size]).to(self.device)
        mask_float = mask.float()
        weights = weights / weights.mean().clamp_min(1e-8)

        with torch.inference_mode():
            with self._autocast():
                pred = self._forward(model, x)
            pred = pred.float()
            if strategy_targets:
                logits = pred.masked_fill(~mask, -1e9)
                probs = torch.softmax(logits, dim=1)
                cross_entropy = -(targets * torch.log_softmax(logits, dim=1)).sum(dim=1)
                tv = 0.5 * torch.abs(probs - targets).sum(dim=1)
                return {
                    "records": size,
                    "cross_entropy": float((cross_entropy * weights).mean().cpu()),
                    "strategy_tv": float((tv * weights).mean().cpu()),
                }

            entry_weight = 1.0 + self.regret_positive_weight * (targets > 1e-6).float()
            squared = (pred - targets).square() * mask_float * entry_weight
            denom = (mask_float * entry_weight).sum(dim=1).clamp_min(1.0)
            mse = squared.sum(dim=1) / denom
            pred_strategy = self._regret_matching_tensor(pred, mask_float)
            target_strategy = self._regret_matching_tensor(targets, mask_float)
            tv = 0.5 * torch.abs(pred_strategy - target_strategy).sum(dim=1)
            support_correct = ((pred > 0.0) == (targets > 0.0)) & mask
            return {
                "records": size,
                "mse": float((mse * weights).mean().cpu()),
                "support_accuracy": float(support_correct.sum().item() / mask.sum().item()),
                "strategy_tv": float((tv * weights).mean().cpu()),
            }

    def validation_metrics(self, *, max_records: int = 2048) -> Dict[str, object]:
        return {
            "regret": [
                self._validation_metrics_for(
                    self.regret_nets[pid],
                    self.regret_validation_buffers[pid],
                    strategy_targets=False,
                    max_records=max_records,
                )
                for pid in (0, 1)
            ],
            "strategy": [
                self._validation_metrics_for(
                    self.strategy_nets[pid],
                    self.strategy_validation_buffers[pid],
                    strategy_targets=True,
                    max_records=max_records,
                )
                for pid in (0, 1)
            ],
        }

    def run_iteration(self, *, traversals_per_player: int = 100) -> Dict[str, object]:
        self.iteration += 1
        strategy_seen_before = [buffer.seen for buffer in self.strategy_buffers]

        traversal_s = 0.0
        regret_training_s = 0.0
        regret_losses = [0.0, 0.0]
        action_sampling_totals = {
            "full_claim_edges": 0,
            "sampled_claim_edges": 0,
            "regret_weight_sum": 0.0,
            "regret_weight_square_sum": 0.0,
            "regret_weight_count": 0,
            "max_regret_weight": 0.0,
        }
        for traverser in (0, 1):
            self.regret_buffers[traverser].clear()
            self.regret_validation_buffers[traverser].clear()
            self._synchronize()
            start = time.perf_counter()
            if self.traversal_backend == "gpu_native":
                from liars_poker.algo.neural_cfr_plus_gpu import (
                    GPUDeepCFRPlusTraverser,
                )

                if self._gpu_traverser is None:
                    self._gpu_traverser = GPUDeepCFRPlusTraverser(self)
                remaining = int(traversals_per_player)
                while remaining > 0:
                    batch = min(self.traversal_batch_size, remaining)
                    traversal_stats = self._gpu_traverser.run_traversals(
                        traverser,
                        batch,
                    )
                    for key in (
                        "full_claim_edges",
                        "sampled_claim_edges",
                        "regret_weight_sum",
                        "regret_weight_square_sum",
                        "regret_weight_count",
                    ):
                        action_sampling_totals[key] += traversal_stats.get(key, 0)
                    action_sampling_totals["max_regret_weight"] = max(
                        action_sampling_totals["max_regret_weight"],
                        traversal_stats.get("max_regret_weight", 0.0),
                    )
                    remaining -= batch
            else:
                for _ in range(traversals_per_player):
                    p1_hand, p2_hand = self._deal()
                    self._traverse((), p1_hand, p2_hand, traverser)
            self._synchronize()
            traversal_s += time.perf_counter() - start

            start = time.perf_counter()
            regret_losses[traverser] = self._train_regret(traverser)
            self._synchronize()
            regret_training_s += time.perf_counter() - start

        start = time.perf_counter()
        strategy_losses = [self._train_strategy(pid) for pid in (0, 1)]
        self._synchronize()
        strategy_training_s = time.perf_counter() - start

        regret_seen = [buffer.seen for buffer in self.regret_buffers]
        strategy_seen = [buffer.seen for buffer in self.strategy_buffers]
        full_edges = action_sampling_totals["full_claim_edges"]
        sampled_edges = action_sampling_totals["sampled_claim_edges"]
        weight_sum = action_sampling_totals["regret_weight_sum"]
        weight_square_sum = action_sampling_totals["regret_weight_square_sum"]
        weight_count = action_sampling_totals["regret_weight_count"]
        sampling_diagnostics = {
            **action_sampling_totals,
            "claim_edge_fraction": (
                sampled_edges / full_edges if full_edges else 1.0
            ),
            "mean_regret_weight": (
                weight_sum / weight_count if weight_count else 1.0
            ),
            "regret_weight_ess_fraction": (
                (weight_sum * weight_sum)
                / (weight_count * weight_square_sum)
                if weight_count and weight_square_sum
                else 1.0
            ),
        }
        return {
            "iteration": self.iteration,
            "regret_loss": regret_losses,
            "strategy_loss": strategy_losses,
            "regret_buffer_sizes": [buffer.size for buffer in self.regret_buffers],
            "strategy_buffer_sizes": [buffer.size for buffer in self.strategy_buffers],
            "regret_records_seen": regret_seen,
            "strategy_records_seen": strategy_seen,
            "new_regret_records": list(regret_seen),
            "new_strategy_records": [
                after - before for before, after in zip(strategy_seen_before, strategy_seen)
            ],
            "action_sampling": sampling_diagnostics,
            "timing": {
                "traversal_s": traversal_s,
                "regret_training_s": regret_training_s,
                "strategy_training_s": strategy_training_s,
            },
        }

    def average_policy(self) -> NeuralPolicy:
        policy = NeuralPolicy(
            self.spec,
            hidden_sizes=self.strategy_hidden_sizes,
            device=self.device,
        )
        policy.model_p1.load_state_dict(self.strategy_nets[0].state_dict())
        policy.model_p2.load_state_dict(self.strategy_nets[1].state_dict())
        return policy.eval()

    def checkpoint_dict(self) -> Dict[str, object]:
        return {
            "version": self.CHECKPOINT_VERSION,
            "spec": _spec_to_dict(self.spec),
            "config": {
                "regret_hidden_sizes": self.regret_hidden_sizes,
                "strategy_hidden_sizes": self.strategy_hidden_sizes,
                "seed": self.seed,
                "regret_buffer_capacity": self.regret_buffers[0].capacity,
                "strategy_buffer_capacity": self.strategy_buffers[0].capacity,
                "learning_rate": self.learning_rate,
                "batch_size": self.batch_size,
                "regret_train_steps": self.regret_train_steps,
                "strategy_train_steps": self.strategy_train_steps,
                "strategy_weighting": self.strategy_weighting,
                "regret_positive_weight": self.regret_positive_weight,
                "validation_fraction": self.validation_fraction,
                "validation_buffer_capacity": self.validation_buffer_capacity,
                "traversal_backend": self.traversal_backend,
                "traversal_batch_size": self.traversal_batch_size,
                "traverser_action_sample_count": self.traverser_action_sample_count,
                "traverser_action_baseline": self.traverser_action_baseline,
                "device_replay": self.device_replay,
                "fused_optimizer": self.fused_optimizer,
                "amp_dtype": self.amp_dtype,
                "compile_models": self.compile_models,
            },
            "iteration": self.iteration,
            "regret_nets": [model.state_dict() for model in self.regret_nets],
            "strategy_nets": [model.state_dict() for model in self.strategy_nets],
            "regret_optimizers": [opt.state_dict() for opt in self.regret_optimizers],
            "strategy_optimizers": [opt.state_dict() for opt in self.strategy_optimizers],
            "grad_scaler": self._grad_scaler.state_dict(),
            "strategy_buffers": [buffer.state_dict() for buffer in self.strategy_buffers],
            "strategy_validation_buffers": [
                buffer.state_dict() for buffer in self.strategy_validation_buffers
            ],
            "random_state": self.rng.getstate(),
            "validation_random_state": self.validation_rng.getstate(),
            "torch_random_state": torch.get_rng_state(),
            "torch_cuda_random_state": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
        }

    def save_checkpoint(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.checkpoint_dict(), path)

    @classmethod
    def load_checkpoint(
        cls,
        path: str | Path,
        *,
        device: str | torch.device = "cpu",
    ) -> "DeepCFRPlusTrainer":
        state = torch.load(path, map_location=device, weights_only=False)
        config = dict(state["config"])
        if (
            "regret_hidden_sizes" not in config
            or "strategy_hidden_sizes" not in config
        ):
            legacy_hidden_sizes = config.pop("hidden_sizes", (256, 256))
            config.setdefault("regret_hidden_sizes", legacy_hidden_sizes)
            config.setdefault("strategy_hidden_sizes", legacy_hidden_sizes)
        else:
            config.pop("hidden_sizes", None)
        config.setdefault("traversal_backend", "recursive")
        config.setdefault("traversal_batch_size", 256)
        config.setdefault("traverser_action_sample_count", None)
        config.setdefault("traverser_action_baseline", "none")
        config.setdefault("device_replay", False)
        config.setdefault("fused_optimizer", None)
        config.setdefault("amp_dtype", None)
        config.setdefault("compile_models", False)
        trainer = cls(_spec_from_dict(state["spec"]), device=device, **config)
        trainer.iteration = int(state["iteration"])

        for model, model_state in zip(trainer.regret_nets, state["regret_nets"]):
            model.load_state_dict(model_state)
            model.eval()
        for model, model_state in zip(trainer.strategy_nets, state["strategy_nets"]):
            model.load_state_dict(model_state)
            model.eval()
        for optimizer, optimizer_state in zip(trainer.regret_optimizers, state["regret_optimizers"]):
            optimizer.load_state_dict(optimizer_state)
        for optimizer, optimizer_state in zip(
            trainer.strategy_optimizers,
            state["strategy_optimizers"],
        ):
            optimizer.load_state_dict(optimizer_state)
        if "grad_scaler" in state:
            trainer._grad_scaler.load_state_dict(state["grad_scaler"])

        def restore_reservoir(buffer_state):
            if trainer.device_replay:
                return DeviceReservoirBuffer.from_state_dict(
                    buffer_state,
                    device=trainer.device,
                )
            return ReservoirBuffer.from_state_dict(buffer_state)

        trainer.strategy_buffers = [
            restore_reservoir(buffer_state)
            for buffer_state in state["strategy_buffers"]
        ]
        if "strategy_validation_buffers" in state:
            trainer.strategy_validation_buffers = [
                restore_reservoir(buffer_state)
                for buffer_state in state["strategy_validation_buffers"]
            ]
        trainer.rng.setstate(state["random_state"])
        if "validation_random_state" in state:
            trainer.validation_rng.setstate(state["validation_random_state"])
        torch.set_rng_state(state["torch_random_state"].cpu())
        if (
            state.get("torch_cuda_random_state") is not None
            and torch.cuda.is_available()
        ):
            torch.cuda.set_rng_state_all(
                [rng_state.cpu() for rng_state in state["torch_cuda_random_state"]]
            )
        return trainer
