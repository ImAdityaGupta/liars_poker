from __future__ import annotations

from contextlib import nullcontext
import random
import time
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import torch

from liars_poker.core import GameSpec, generate_deck, possible_starting_hands
from liars_poker.env import resolve_call_winner, rules_for_spec
from liars_poker.infoset import CALL, InfoSet
from liars_poker.policies.neural import InfosetEncoder, NeuralMLP, NeuralPolicy
from liars_poker.policies.tabular_dense import DenseTabularPolicy


def _spec_to_dict(spec: GameSpec) -> Dict[str, object]:
    return {
        "ranks": spec.ranks,
        "suits": spec.suits,
        "hand_size": spec.hand_size,
        "claim_kinds": list(spec.claim_kinds),
        "suit_symmetry": spec.suit_symmetry,
    }


def _spec_from_dict(data: Dict[str, object]) -> GameSpec:
    return GameSpec(
        ranks=int(data["ranks"]),
        suits=int(data["suits"]),
        hand_size=int(data["hand_size"]),
        claim_kinds=tuple(data["claim_kinds"]),
        suit_symmetry=bool(data["suit_symmetry"]),
    )


class ReservoirBuffer:
    """Fixed-capacity reservoir for neural training records."""

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

    def add(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        legal_mask: np.ndarray,
        weight: float,
        rng: random.Random,
    ) -> None:
        self.seen += 1
        if self.size < self.capacity:
            idx = self.size
            self.size += 1
        else:
            idx = rng.randrange(self.seen)
            if idx >= self.capacity:
                return

        self.features[idx] = features
        self.targets[idx] = targets
        self.legal_masks[idx] = legal_mask
        self.weights[idx] = weight

    def add_many(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        legal_masks: np.ndarray,
        weights: np.ndarray | float,
        rng: random.Random,
    ) -> None:
        n = int(len(features))
        if n == 0:
            return

        weights_arr = (
            np.full(n, float(weights), dtype=np.float32)
            if np.isscalar(weights)
            else np.asarray(weights, dtype=np.float32)
        )

        room = min(n, self.capacity - self.size)
        if room > 0:
            start = self.size
            stop = start + room
            self.features[start:stop] = features[:room]
            self.targets[start:stop] = targets[:room]
            self.legal_masks[start:stop] = legal_masks[:room]
            self.weights[start:stop] = weights_arr[:room]
            self.size = stop
            self.seen += room

        for offset in range(room, n):
            self.seen += 1
            idx = rng.randrange(self.seen)
            if idx >= self.capacity:
                continue
            self.features[idx] = features[offset]
            self.targets[idx] = targets[offset]
            self.legal_masks[idx] = legal_masks[offset]
            self.weights[idx] = weights_arr[offset]

    def sample(self, batch_size: int, rng: random.Random) -> Tuple[np.ndarray, ...]:
        indices = np.fromiter(
            (rng.randrange(self.size) for _ in range(min(batch_size, self.size))),
            dtype=np.int64,
        )
        return (
            self.features[indices],
            self.targets[indices],
            self.legal_masks[indices],
            self.weights[indices],
        )

    def state_dict(self) -> Dict[str, object]:
        return {
            "kind": "cpu",
            "capacity": self.capacity,
            "input_dim": self.input_dim,
            "action_dim": self.action_dim,
            "features": self.features[: self.size].copy(),
            "targets": self.targets[: self.size].copy(),
            "legal_masks": self.legal_masks[: self.size].copy(),
            "weights": self.weights[: self.size].copy(),
            "size": self.size,
            "seen": self.seen,
        }

    @classmethod
    def from_state_dict(cls, state: Dict[str, object]) -> "ReservoirBuffer":
        buffer = cls(
            int(state["capacity"]),
            int(state["input_dim"]),
            int(state["action_dim"]),
        )
        buffer.size = int(state["size"])
        buffer.seen = int(state["seen"])
        buffer.features[: buffer.size] = state["features"]
        buffer.targets[: buffer.size] = state["targets"]
        buffer.legal_masks[: buffer.size] = state["legal_masks"]
        buffer.weights[: buffer.size] = state["weights"]
        return buffer


class DeviceReservoirBuffer:
    """Exact Algorithm-R reservoir stored entirely on one Torch device."""

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

        room = min(n, self.capacity - self.size)
        if room:
            start = self.size
            stop = start + room
            self.features[start:stop].copy_(features[:room])
            self.targets[start:stop].copy_(targets[:room])
            self.legal_masks[start:stop].copy_(legal_masks[:room])
            self.weights[start:stop].copy_(weights_t[:room])
            self.size = stop
            self.seen += room

        remaining = n - room
        if remaining <= 0:
            return

        rem_features = features[room:]
        rem_targets = targets[room:]
        rem_masks = legal_masks[room:]
        rem_weights = weights_t[room:]

        # Algorithm R: record t draws a replacement slot uniformly from [0, t].
        # Multiple new records may select the same slot; the last such record wins,
        # exactly as it would in the sequential algorithm.
        bounds = torch.arange(
            self.seen + 1,
            self.seen + remaining + 1,
            dtype=torch.float64,
            device=self.device,
        )
        slots = torch.floor(torch.rand(remaining, device=self.device) * bounds).long()
        accepted = slots < self.capacity
        self.seen += remaining

        accepted_slots = slots[accepted]
        accepted_sources = torch.arange(
            remaining,
            dtype=torch.long,
            device=self.device,
        )[accepted]
        unique_slots, inverse = torch.unique(
            accepted_slots,
            sorted=False,
            return_inverse=True,
        )
        last_source = torch.full(
            (len(unique_slots),),
            -1,
            dtype=torch.long,
            device=self.device,
        )
        last_source.scatter_reduce_(
            0,
            inverse,
            accepted_sources,
            reduce="amax",
            include_self=True,
        )
        self.features[unique_slots] = rem_features[last_source]
        self.targets[unique_slots] = rem_targets[last_source]
        self.legal_masks[unique_slots] = rem_masks[last_source]
        self.weights[unique_slots] = rem_weights[last_source]

    def sample(self, batch_size: int, rng: random.Random | None = None) -> Tuple[torch.Tensor, ...]:
        _ = rng
        n = min(int(batch_size), self.size)
        indices = torch.randint(self.size, (n,), device=self.device)
        return (
            self.features.index_select(0, indices),
            self.targets.index_select(0, indices),
            self.legal_masks.index_select(0, indices),
            self.weights.index_select(0, indices),
        )

    def state_dict(self) -> Dict[str, object]:
        return {
            "kind": "device",
            "capacity": self.capacity,
            "input_dim": self.input_dim,
            "action_dim": self.action_dim,
            "features": self.features[: self.size].detach().cpu(),
            "targets": self.targets[: self.size].detach().cpu(),
            "legal_masks": self.legal_masks[: self.size].detach().cpu(),
            "weights": self.weights[: self.size].detach().cpu(),
            "size": self.size,
            "seen": self.seen,
        }

    @classmethod
    def from_state_dict(
        cls,
        state: Dict[str, object],
        *,
        device: str | torch.device,
    ) -> "DeviceReservoirBuffer":
        buffer = cls(
            int(state["capacity"]),
            int(state["input_dim"]),
            int(state["action_dim"]),
            device,
        )
        buffer.size = int(state["size"])
        buffer.seen = int(state["seen"])
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


class DeepCFRTrainer:
    """External-sampling Deep CFR with player-specific advantage and average networks."""

    CHECKPOINT_VERSION = 4

    def __init__(
        self,
        spec: GameSpec,
        *,
        hidden_sizes: Sequence[int] = (128, 128),
        device: str | torch.device = "cpu",
        seed: int = 0,
        advantage_buffer_capacity: int = 100_000,
        strategy_buffer_capacity: int = 100_000,
        learning_rate: float = 1e-3,
        batch_size: int = 256,
        advantage_train_steps: int = 100,
        strategy_train_steps: int = 100,
        advantage_positive_weight: float = 0.0,
        strategy_weighting: str = "linear",
        highest_regret_fallback: bool = True,
        alternating_updates: bool = True,
        retrain_advantage_from_scratch: bool = False,
        validation_fraction: float = 0.0,
        validation_buffer_capacity: int = 10_000,
        traversal_backend: str = "recursive",
        traversal_batch_size: int = 256,
        device_replay: bool = False,
        fused_optimizer: bool | None = None,
        amp_dtype: str | None = None,
        compile_models: bool = False,
    ) -> None:
        self.spec = spec
        self.rules = rules_for_spec(spec)
        self.encoder = InfosetEncoder(spec)
        self.hidden_sizes = tuple(int(size) for size in hidden_sizes)
        self.device = torch.device(device)
        self.seed = int(seed)
        self.rng = random.Random(seed)
        self.validation_rng = random.Random(seed + 1_000_003)
        torch.manual_seed(seed)

        self.learning_rate = float(learning_rate)
        self.batch_size = int(batch_size)
        self.advantage_train_steps = int(advantage_train_steps)
        self.strategy_train_steps = int(strategy_train_steps)
        self.advantage_positive_weight = float(advantage_positive_weight)
        if strategy_weighting not in {"linear", "uniform"}:
            raise ValueError("strategy_weighting must be 'linear' or 'uniform'.")
        self.strategy_weighting = strategy_weighting
        self.highest_regret_fallback = bool(highest_regret_fallback)
        self.alternating_updates = bool(alternating_updates)
        self.retrain_advantage_from_scratch = bool(retrain_advantage_from_scratch)
        self.validation_fraction = float(validation_fraction)
        self.validation_buffer_capacity = int(validation_buffer_capacity)
        if traversal_backend not in {"recursive", "batched", "gpu_native"}:
            raise ValueError(
                "traversal_backend must be 'recursive', 'batched', or 'gpu_native'."
            )
        self.traversal_backend = traversal_backend
        self.traversal_batch_size = int(traversal_batch_size)
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

        self.advantage_nets = [
            NeuralMLP(self.encoder.input_dim, self.encoder.action_dim, self.hidden_sizes).to(self.device)
            for _ in range(2)
        ]
        self.strategy_nets = [
            NeuralMLP(self.encoder.input_dim, self.encoder.action_dim, self.hidden_sizes).to(self.device)
            for _ in range(2)
        ]
        self.advantage_optimizers = [
            self._make_optimizer(model)
            for model in self.advantage_nets
        ]
        self.strategy_optimizers = [
            self._make_optimizer(model)
            for model in self.strategy_nets
        ]

        buffer_cls = DeviceReservoirBuffer if self.device_replay else ReservoirBuffer
        buffer_args = (self.device,) if self.device_replay else ()
        self.advantage_buffers = [
            buffer_cls(
                advantage_buffer_capacity,
                self.encoder.input_dim,
                self.encoder.action_dim,
                *buffer_args,
            )
            for _ in range(2)
        ]
        self.strategy_buffers = [
            buffer_cls(
                strategy_buffer_capacity,
                self.encoder.input_dim,
                self.encoder.action_dim,
                *buffer_args,
            )
            for _ in range(2)
        ]
        self.advantage_validation_buffers = [
            buffer_cls(
                validation_buffer_capacity,
                self.encoder.input_dim,
                self.encoder.action_dim,
                *buffer_args,
            )
            for _ in range(2)
        ]
        self.strategy_validation_buffers = [
            buffer_cls(
                validation_buffer_capacity,
                self.encoder.input_dim,
                self.encoder.action_dim,
                *buffer_args,
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
    def advantage_net_p1(self) -> NeuralMLP:
        return self.advantage_nets[0]

    @property
    def advantage_net_p2(self) -> NeuralMLP:
        return self.advantage_nets[1]

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

    def _strategy_from_features(
        self,
        pid: int,
        features: np.ndarray,
        legal: Tuple[int, ...],
    ) -> np.ndarray:
        x = torch.from_numpy(features).to(self.device)
        with torch.no_grad():
            with self._autocast():
                advantages = self._forward(self.advantage_nets[pid], x)
            advantages = advantages.float().cpu().numpy()

        strategy = np.zeros(self.encoder.action_dim, dtype=np.float32)
        cols = [self._action_col(action) for action in legal]
        positive = np.maximum(advantages[cols], 0.0)
        total = float(positive.sum())
        if total > 0.0:
            strategy[cols] = positive / total
        elif self.highest_regret_fallback:
            strategy[cols[int(np.argmax(advantages[cols]))]] = 1.0
        else:
            strategy[cols] = 1.0 / len(cols)
        return strategy

    def current_strategy(self, infoset: InfoSet) -> Dict[int, float]:
        legal = self.rules.legal_actions_for(infoset)
        features = self.encoder.encode(infoset.hand, infoset.history)
        strategy = self._strategy_from_features(infoset.pid, features, legal)
        return {action: float(strategy[self._action_col(action)]) for action in legal}

    def current_policy_dense(self, *, batch_size: int = 4096) -> DenseTabularPolicy:
        """Compile the current regret-matched strategy for exact evaluation."""

        dense = DenseTabularPolicy(self.spec)
        hands = tuple(possible_starting_hands(self.spec))

        for hid in range(1 << self.encoder.k):
            history = tuple(action for action in range(self.encoder.k) if hid & (1 << action))
            legal = dense.legal_actions[hid]
            cols = [self._action_col(action) for action in legal]
            model = self.advantage_nets[dense.pid_to_act(hid)]

            for start in range(0, len(hands), batch_size):
                stop = min(start + batch_size, len(hands))
                features = self.encoder.encode_hands(hands[start:stop], history)
                x = torch.from_numpy(features).to(self.device)
                with torch.no_grad():
                    with self._autocast():
                        advantages = self._forward(model, x)
                    advantages = advantages.float().cpu().numpy()

                positive = np.maximum(advantages[:, cols], 0.0)
                totals = positive.sum(axis=1, keepdims=True)
                probs = np.divide(
                    positive,
                    totals,
                    out=np.zeros_like(positive),
                    where=totals > 0.0,
                )
                fallback = totals[:, 0] <= 0.0
                if np.any(fallback):
                    if self.highest_regret_fallback:
                        best = np.argmax(advantages[fallback][:, cols], axis=1)
                        probs[fallback, best] = 1.0
                    else:
                        probs[fallback] = 1.0 / len(cols)
                block = dense.S[hid, start:stop]
                block.fill(0.0)
                block[:, cols] = probs

        dense.recompute_likelihoods()
        return dense

    def advantage_values(self, infoset: InfoSet) -> Dict[int, float]:
        legal = self.rules.legal_actions_for(infoset)
        x = torch.from_numpy(self.encoder.encode(infoset.hand, infoset.history)).to(self.device)
        with torch.no_grad():
            with self._autocast():
                values = self._forward(self.advantage_nets[infoset.pid], x)
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

    def _add_record(
        self,
        training_buffer,
        validation_buffer,
        features,
        targets,
        legal_mask,
        weight: float,
    ) -> None:
        if isinstance(training_buffer, DeviceReservoirBuffer):
            features = torch.as_tensor(features, device=self.device)
            targets = torch.as_tensor(targets, device=self.device)
            legal_mask = torch.as_tensor(legal_mask, device=self.device)
        if self.validation_fraction > 0.0 and self.validation_rng.random() < self.validation_fraction:
            validation_buffer.add(features, targets, legal_mask, weight, self.validation_rng)
        else:
            training_buffer.add(features, targets, legal_mask, weight, self.rng)

    def _add_records(
        self,
        training_buffer,
        validation_buffer,
        features,
        targets,
        legal_masks,
        weights,
    ) -> None:
        n = int(len(features))
        if n == 0:
            return

        if isinstance(training_buffer, DeviceReservoirBuffer):
            self._add_device_records(
                training_buffer,
                validation_buffer,
                torch.as_tensor(features, device=self.device),
                torch.as_tensor(targets, device=self.device),
                torch.as_tensor(legal_masks, device=self.device),
                weights,
            )
            return

        weights_arr = (
            np.full(n, float(weights), dtype=np.float32)
            if np.isscalar(weights)
            else np.asarray(weights, dtype=np.float32)
        )

        if self.validation_fraction > 0.0:
            use_validation = np.fromiter(
                (self.validation_rng.random() < self.validation_fraction for _ in range(n)),
                dtype=bool,
                count=n,
            )
            if np.any(use_validation):
                validation_buffer.add_many(
                    features[use_validation],
                    targets[use_validation],
                    legal_masks[use_validation],
                    weights_arr[use_validation],
                    self.validation_rng,
                )
            use_training = ~use_validation
            if np.any(use_training):
                training_buffer.add_many(
                    features[use_training],
                    targets[use_training],
                    legal_masks[use_training],
                    weights_arr[use_training],
                    self.rng,
                )
            return

        training_buffer.add_many(features, targets, legal_masks, weights_arr, self.rng)

    def _add_device_records(
        self,
        training_buffer: DeviceReservoirBuffer,
        validation_buffer: DeviceReservoirBuffer,
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
                device=self.device,
                dtype=torch.float32,
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
        strategy = self._strategy_from_features(to_play, features, legal)

        if to_play == traverser:
            action_values = np.zeros(self.encoder.action_dim, dtype=np.float32)
            node_value = 0.0
            for action in legal:
                col = self._action_col(action)
                value = self._traverse(history + (action,), p1_hand, p2_hand, traverser)
                action_values[col] = value
                node_value += float(strategy[col]) * value

            advantages = np.zeros(self.encoder.action_dim, dtype=np.float32)
            advantages[legal_mask] = action_values[legal_mask] - node_value
            self._add_record(
                self.advantage_buffers[traverser],
                self.advantage_validation_buffers[traverser],
                features,
                advantages,
                legal_mask,
                float(self.iteration),
            )
            return node_value

        # In the other player's traversal, this actor's own past actions are sampled,
        # so these records have the reach weighting needed for average strategy.
        self._add_record(
            self.strategy_buffers[to_play],
            self.strategy_validation_buffers[to_play],
            features,
            strategy,
            legal_mask,
            1.0 if self.strategy_weighting == "uniform" else float(self.iteration),
        )
        action = self._sample_action(legal, strategy)
        return self._traverse(history + (action,), p1_hand, p2_hand, traverser)

    def _train_advantage(self, pid: int) -> float:
        if (
            self.retrain_advantage_from_scratch
            and self.advantage_buffers[pid].size > 0
            and self.advantage_train_steps > 0
        ):
            self.advantage_nets[pid] = NeuralMLP(
                self.encoder.input_dim,
                self.encoder.action_dim,
                self.hidden_sizes,
            ).to(self.device)
            self.advantage_optimizers[pid] = self._make_optimizer(
                self.advantage_nets[pid]
            )
            self._compiled_forwards.clear()
        return self._train_model(
            self.advantage_nets[pid],
            self.advantage_optimizers[pid],
            self.advantage_buffers[pid],
            self.advantage_train_steps,
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
        buffer: ReservoirBuffer,
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
                if self.advantage_positive_weight:
                    entry_weight = mask_float * (
                        1.0 + self.advantage_positive_weight * (y > 1e-6).float()
                    )
                else:
                    entry_weight = mask_float
                squared = (pred - y).square() * entry_weight
                per_sample = squared.sum(dim=1) / entry_weight.sum(dim=1).clamp_min(1.0)
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
        if self.highest_regret_fallback:
            masked_values = values.masked_fill(mask <= 0.0, -torch.inf)
            fallback = torch.zeros_like(values).scatter_(
                1,
                masked_values.argmax(dim=1, keepdim=True),
                1.0,
            )
        else:
            fallback = mask / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        return torch.where(totals > 0.0, matched, fallback)

    def _validation_metrics_for(
        self,
        model: NeuralMLP,
        buffer: ReservoirBuffer,
        *,
        strategy_targets: bool,
        max_records: int,
    ) -> Dict[str, float]:
        size = min(buffer.size, max_records)
        if size == 0:
            return {"records": 0}

        if isinstance(buffer, DeviceReservoirBuffer):
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

        with torch.no_grad():
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

            squared = (pred - targets).square() * mask_float
            mse = squared.sum(dim=1) / mask_float.sum(dim=1).clamp_min(1.0)
            significant = mask & (targets.abs() > 1e-3)
            sign_total = int(significant.sum().item())
            sign_correct = ((pred > 0.0) == (targets > 0.0)) & significant
            support_correct = ((pred > 0.0) == (targets > 0.0)) & mask
            pred_strategy = self._regret_matching_tensor(pred, mask_float)
            target_strategy = self._regret_matching_tensor(targets, mask_float)
            tv = 0.5 * torch.abs(pred_strategy - target_strategy).sum(dim=1)
            return {
                "records": size,
                "mse": float((mse * weights).mean().cpu()),
                "sign_accuracy": float(sign_correct.sum().item() / sign_total) if sign_total else 1.0,
                "support_accuracy": float(support_correct.sum().item() / mask.sum().item()),
                "strategy_tv": float((tv * weights).mean().cpu()),
            }

    def validation_metrics(self, *, max_records: int = 2048) -> Dict[str, object]:
        return {
            "advantage": [
                self._validation_metrics_for(
                    self.advantage_nets[pid],
                    self.advantage_validation_buffers[pid],
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
        advantage_seen_before = [buffer.seen for buffer in self.advantage_buffers]
        strategy_seen_before = [buffer.seen for buffer in self.strategy_buffers]

        traversal_s = 0.0
        advantage_training_s = 0.0
        advantage_losses = [0.0, 0.0]
        if self.traversal_backend == "gpu_native":
            from liars_poker.algo.neural_cfr_gpu import GPUDeepCFRTraverser

            if self._gpu_traverser is None:
                self._gpu_traverser = GPUDeepCFRTraverser(self)
            for traverser in (0, 1):
                self._synchronize()
                start = time.perf_counter()
                remaining = int(traversals_per_player)
                while remaining > 0:
                    batch = min(self.traversal_batch_size, remaining)
                    self._gpu_traverser.run_traversals(traverser, batch)
                    remaining -= batch
                self._synchronize()
                traversal_s += time.perf_counter() - start

                if self.alternating_updates:
                    start = time.perf_counter()
                    advantage_losses[traverser] = self._train_advantage(traverser)
                    advantage_training_s += time.perf_counter() - start

            if not self.alternating_updates:
                start = time.perf_counter()
                advantage_losses = [self._train_advantage(pid) for pid in (0, 1)]
                advantage_training_s = time.perf_counter() - start
        elif self.traversal_backend == "batched":
            from liars_poker.algo.neural_cfr_batched import BatchedDeepCFRTraverser

            batched_traverser = BatchedDeepCFRTraverser(self)
            for traverser in (0, 1):
                start = time.perf_counter()
                remaining = int(traversals_per_player)
                while remaining > 0:
                    batch = min(self.traversal_batch_size, remaining)
                    batched_traverser.run_traversals(traverser, batch)
                    remaining -= batch
                traversal_s += time.perf_counter() - start

                if self.alternating_updates:
                    start = time.perf_counter()
                    advantage_losses[traverser] = self._train_advantage(traverser)
                    advantage_training_s += time.perf_counter() - start

            if not self.alternating_updates:
                start = time.perf_counter()
                advantage_losses = [self._train_advantage(pid) for pid in (0, 1)]
                advantage_training_s = time.perf_counter() - start
        elif self.alternating_updates:
            for traverser in (0, 1):
                start = time.perf_counter()
                for _ in range(traversals_per_player):
                    p1_hand, p2_hand = self._deal()
                    self._traverse((), p1_hand, p2_hand, traverser)
                traversal_s += time.perf_counter() - start

                start = time.perf_counter()
                advantage_losses[traverser] = self._train_advantage(traverser)
                advantage_training_s += time.perf_counter() - start
        else:
            start = time.perf_counter()
            for traverser in (0, 1):
                for _ in range(traversals_per_player):
                    p1_hand, p2_hand = self._deal()
                    self._traverse((), p1_hand, p2_hand, traverser)
            traversal_s = time.perf_counter() - start

            start = time.perf_counter()
            advantage_losses = [self._train_advantage(pid) for pid in (0, 1)]
            advantage_training_s = time.perf_counter() - start

        start = time.perf_counter()
        strategy_losses = [self._train_strategy(pid) for pid in (0, 1)]
        strategy_training_s = time.perf_counter() - start

        advantage_seen = [buffer.seen for buffer in self.advantage_buffers]
        strategy_seen = [buffer.seen for buffer in self.strategy_buffers]
        record = {
            "iteration": self.iteration,
            "advantage_loss": advantage_losses,
            "strategy_loss": strategy_losses,
            "advantage_buffer_sizes": [buffer.size for buffer in self.advantage_buffers],
            "strategy_buffer_sizes": [buffer.size for buffer in self.strategy_buffers],
            "advantage_records_seen": advantage_seen,
            "strategy_records_seen": strategy_seen,
            "new_advantage_records": [
                after - before for before, after in zip(advantage_seen_before, advantage_seen)
            ],
            "new_strategy_records": [
                after - before for before, after in zip(strategy_seen_before, strategy_seen)
            ],
            "timing": {
                "traversal_s": traversal_s,
                "advantage_training_s": advantage_training_s,
                "strategy_training_s": strategy_training_s,
            },
        }
        return record

    def average_policy(self) -> NeuralPolicy:
        policy = NeuralPolicy(
            self.spec,
            hidden_sizes=self.hidden_sizes,
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
                "hidden_sizes": self.hidden_sizes,
                "seed": self.seed,
                "advantage_buffer_capacity": self.advantage_buffers[0].capacity,
                "strategy_buffer_capacity": self.strategy_buffers[0].capacity,
                "learning_rate": self.learning_rate,
                "batch_size": self.batch_size,
                "advantage_train_steps": self.advantage_train_steps,
                "strategy_train_steps": self.strategy_train_steps,
                "advantage_positive_weight": self.advantage_positive_weight,
                "strategy_weighting": self.strategy_weighting,
                "highest_regret_fallback": self.highest_regret_fallback,
                "alternating_updates": self.alternating_updates,
                "retrain_advantage_from_scratch": self.retrain_advantage_from_scratch,
                "validation_fraction": self.validation_fraction,
                "validation_buffer_capacity": self.validation_buffer_capacity,
                "traversal_backend": self.traversal_backend,
                "traversal_batch_size": self.traversal_batch_size,
                "device_replay": self.device_replay,
                "fused_optimizer": self.fused_optimizer,
                "amp_dtype": self.amp_dtype,
                "compile_models": self.compile_models,
            },
            "iteration": self.iteration,
            "advantage_nets": [model.state_dict() for model in self.advantage_nets],
            "strategy_nets": [model.state_dict() for model in self.strategy_nets],
            "advantage_optimizers": [opt.state_dict() for opt in self.advantage_optimizers],
            "strategy_optimizers": [opt.state_dict() for opt in self.strategy_optimizers],
            "grad_scaler": self._grad_scaler.state_dict(),
            "advantage_buffers": [buffer.state_dict() for buffer in self.advantage_buffers],
            "strategy_buffers": [buffer.state_dict() for buffer in self.strategy_buffers],
            "advantage_validation_buffers": [
                buffer.state_dict() for buffer in self.advantage_validation_buffers
            ],
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
    ) -> "DeepCFRTrainer":
        state = torch.load(path, map_location=device, weights_only=False)
        config = dict(state["config"])
        config.setdefault("traversal_backend", "recursive")
        config.setdefault("traversal_batch_size", 256)
        config.setdefault("advantage_positive_weight", 0.0)
        config.setdefault("device_replay", False)
        config.setdefault("fused_optimizer", None)
        config.setdefault("amp_dtype", None)
        config.setdefault("compile_models", False)
        if int(state.get("version", 1)) < 2:
            config.setdefault("highest_regret_fallback", False)
            config.setdefault("alternating_updates", False)
            config.setdefault("retrain_advantage_from_scratch", False)
        trainer = cls(_spec_from_dict(state["spec"]), device=device, **config)
        trainer.iteration = int(state["iteration"])

        for model, model_state in zip(trainer.advantage_nets, state["advantage_nets"]):
            model.load_state_dict(model_state)
            model.eval()
        for model, model_state in zip(trainer.strategy_nets, state["strategy_nets"]):
            model.load_state_dict(model_state)
            model.eval()
        for optimizer, optimizer_state in zip(
            trainer.advantage_optimizers,
            state["advantage_optimizers"],
        ):
            optimizer.load_state_dict(optimizer_state)
        for optimizer, optimizer_state in zip(
            trainer.strategy_optimizers,
            state["strategy_optimizers"],
        ):
            optimizer.load_state_dict(optimizer_state)
        if "grad_scaler" in state:
            trainer._grad_scaler.load_state_dict(state["grad_scaler"])

        def restore_buffer(buffer_state):
            if trainer.device_replay:
                return DeviceReservoirBuffer.from_state_dict(
                    buffer_state,
                    device=trainer.device,
                )
            return ReservoirBuffer.from_state_dict(buffer_state)

        trainer.advantage_buffers = [
            restore_buffer(buffer_state) for buffer_state in state["advantage_buffers"]
        ]
        trainer.strategy_buffers = [
            restore_buffer(buffer_state) for buffer_state in state["strategy_buffers"]
        ]
        if "advantage_validation_buffers" in state:
            trainer.advantage_validation_buffers = [
                restore_buffer(buffer_state)
                for buffer_state in state["advantage_validation_buffers"]
            ]
            trainer.strategy_validation_buffers = [
                restore_buffer(buffer_state)
                for buffer_state in state["strategy_validation_buffers"]
            ]
        trainer.rng.setstate(state["random_state"])
        if "validation_random_state" in state:
            trainer.validation_rng.setstate(state["validation_random_state"])
        torch.set_rng_state(state["torch_random_state"])
        if (
            state.get("torch_cuda_random_state") is not None
            and torch.cuda.is_available()
        ):
            torch.cuda.set_rng_state_all(state["torch_cuda_random_state"])
        return trainer
