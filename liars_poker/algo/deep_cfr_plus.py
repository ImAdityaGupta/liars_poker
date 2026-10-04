from __future__ import annotations

from contextlib import nullcontext
import math
import random
import time
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import torch

from liars_poker.algo.cfr_plus_targets import make_regret_target
from liars_poker.algo.regret_readers import NetworkRegretReader
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


_CUDA_AGGREGATE_SPEC = GameSpec(
    ranks=4,
    suits=4,
    hand_size=2,
    claim_kinds=("RankHigh", "Pair", "TwoPair", "Trips"),
    suit_symmetry=True,
)
_CUDA_AGGREGATE_30_SPEC = GameSpec(
    ranks=5, suits=4, hand_size=3,
    claim_kinds=("RankHigh", "Pair", "TwoPair", "Trips", "Quads"),
    suit_symmetry=True,
)


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
        self.require_no_overwrite = False

    def add(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        legal_mask: np.ndarray,
        weight: float,
        rng: random.Random | None = None,
    ) -> None:
        _ = rng
        if self.require_no_overwrite and self.seen >= self.capacity:
            raise OverflowError("Regret buffer cannot hold all rows from this iteration")
        idx = self.cursor
        self.features[idx] = features
        self.targets[idx] = targets
        self.legal_masks[idx] = legal_mask
        self.weights[idx] = weight

        self.seen += 1
        self.size = min(self.size + 1, self.capacity)
        self.cursor = (self.cursor + 1) % self.capacity

    def add_many(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        legal_masks: np.ndarray,
        weights: np.ndarray | float,
        rng: random.Random | None = None,
    ) -> None:
        _ = rng
        features = np.asarray(features, dtype=np.float32)
        targets = np.asarray(targets, dtype=np.float32)
        legal_masks = np.asarray(legal_masks, dtype=bool)
        n = int(features.shape[0])
        if n == 0:
            return
        if self.require_no_overwrite and self.seen + n > self.capacity:
            raise OverflowError("Regret buffer cannot hold all rows from this iteration")

        weights_arr = (
            np.full(n, float(weights), dtype=np.float32)
            if np.isscalar(weights)
            else np.asarray(weights, dtype=np.float32)
        )

        if n >= self.capacity:
            self.features[:] = features[-self.capacity :]
            self.targets[:] = targets[-self.capacity :]
            self.legal_masks[:] = legal_masks[-self.capacity :]
            self.weights[:] = weights_arr[-self.capacity :]
            self.size = self.capacity
            self.seen += n
            self.cursor = 0
            return

        indices = (np.arange(n, dtype=np.int64) + self.cursor) % self.capacity
        self.features[indices] = features
        self.targets[indices] = targets
        self.legal_masks[indices] = legal_masks
        self.weights[indices] = weights_arr
        self.cursor = (self.cursor + n) % self.capacity
        self.size = min(self.capacity, self.size + n)
        self.seen += n

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
        self.require_no_overwrite = False

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
        if self.require_no_overwrite and self.seen + n > self.capacity:
            raise OverflowError("Regret buffer cannot hold all rows from this iteration")

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
    """External-sampling neural CFR+ with configurable regret targets."""

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
        regret_batch_size: int | None = None,
        regret_fit_schedule: str = "constant",
        regret_fit_learning_rate: float | None = None,
        regret_train_steps: int = 100,
        strategy_train_steps: int = 50,
        use_regret_network: bool = True,
        use_strategy_network: bool = True,
        strategy_weighting: str = "linear",
        regret_positive_weight: float = 0.5,
        regret_target_mode: str = "clip_each_record",
        regret_increment_reach_mode: str = "none",
        regret_accumulation_mode: str = "normalized",
        validation_fraction: float = 0.0,
        validation_buffer_capacity: int = 10_000,
        traversal_backend: str = "recursive",
        traversal_batch_size: int = 256,
        traverser_action_sample_count: int | None = None,
        traverser_action_sample_fraction: float | None = None,
        traverser_action_full_first: bool = False,
        traverser_action_sample_schedule: Sequence[int] | None = None,
        traverser_action_priority_count: int = 0,
        traverser_action_baseline: str = "none",
        traverser_action_sample_mode: str = "random",
        traversal_streaming: bool = False,
        traversal_live_row_budget: int | None = None,
        traverser_action_chunk_size: int | None = None,
        traversal_record_flush_size: int = 131_072,
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
        self.regret_batch_size = int(batch_size if regret_batch_size is None else regret_batch_size)
        self.regret_fit_schedule = regret_fit_schedule
        self.regret_fit_learning_rate = (None if regret_fit_learning_rate is None
                                         else float(regret_fit_learning_rate))
        if self.regret_batch_size <= 0 or regret_fit_schedule not in {"constant", "cosine"}:
            raise ValueError("Invalid regret fitting batch size or schedule")
        if self.regret_fit_learning_rate is not None and self.regret_fit_learning_rate <= 0:
            raise ValueError("Regret fitting learning rate must be positive")
        if self.regret_fit_schedule == "cosine" and self.regret_fit_learning_rate is None:
            raise ValueError("A cosine regret fit requires regret_fit_learning_rate")
        self.regret_train_steps = int(regret_train_steps)
        self.strategy_train_steps = int(strategy_train_steps)
        self.use_regret_network = bool(use_regret_network)
        self.use_strategy_network = bool(use_strategy_network)
        if strategy_weighting not in {"linear", "uniform", "quadratic"}:
            raise ValueError("strategy_weighting must be uniform, linear, or quadratic.")
        self.strategy_weighting = strategy_weighting
        self.regret_positive_weight = float(regret_positive_weight)
        self.regret_target_mode = regret_target_mode
        if regret_increment_reach_mode not in {"none", "visit_fraction", "visit_count"}:
            raise ValueError("Unknown regret_increment_reach_mode.")
        self.regret_increment_reach_mode = regret_increment_reach_mode
        if regret_accumulation_mode not in {"normalized", "cumulative"}:
            raise ValueError("Unknown regret_accumulation_mode.")
        self.regret_accumulation_mode = regret_accumulation_mode
        self.validation_fraction = float(validation_fraction)
        self.validation_buffer_capacity = int(validation_buffer_capacity)
        if traversal_backend not in {"recursive", "gpu_native"}:
            raise ValueError("traversal_backend must be 'recursive' or 'gpu_native'.")
        self.traversal_backend = traversal_backend
        self._validate_regret_target_mode(self.regret_target_mode, self.regret_positive_weight)
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
        self.traverser_action_sample_fraction = (
            None
            if traverser_action_sample_fraction is None
            else float(traverser_action_sample_fraction)
        )
        if (
            self.traverser_action_sample_fraction is not None
            and not 0.0 < self.traverser_action_sample_fraction <= 1.0
        ):
            raise ValueError(
                "traverser_action_sample_fraction must be in (0, 1]."
            )
        if (
            self.traverser_action_sample_count is not None
            and self.traverser_action_sample_fraction is not None
        ):
            raise ValueError(
                "Specify either traverser_action_sample_count or "
                "traverser_action_sample_fraction, not both."
            )
        self.traverser_action_full_first = bool(traverser_action_full_first)
        self.traverser_action_sample_schedule = (
            None
            if traverser_action_sample_schedule is None
            else tuple(int(count) for count in traverser_action_sample_schedule)
        )
        if self.traverser_action_sample_schedule is not None:
            if not self.traverser_action_sample_schedule:
                raise ValueError("traverser_action_sample_schedule cannot be empty.")
            if any(count <= 0 for count in self.traverser_action_sample_schedule):
                raise ValueError(
                    "traverser_action_sample_schedule entries must be positive."
                )
            if (
                self.traverser_action_sample_count is not None
                or self.traverser_action_sample_fraction is not None
            ):
                raise ValueError(
                    "Specify traverser_action_sample_schedule instead of "
                    "traverser_action_sample_count/sample_fraction."
                )
        self.traverser_action_priority_count = int(
            traverser_action_priority_count
        )
        if self.traverser_action_priority_count < 0:
            raise ValueError("traverser_action_priority_count cannot be negative.")
        if (
            self.traverser_action_priority_count
            and self.traverser_action_sample_count is None
            and self.traverser_action_sample_schedule is None
        ):
            raise ValueError(
                "Priority sampling requires traverser_action_sample_count "
                "or traverser_action_sample_schedule."
            )
        if (
            self.traverser_action_sample_count is not None
            and self.traverser_action_priority_count
            > self.traverser_action_sample_count
        ):
            raise ValueError(
                "traverser_action_priority_count cannot exceed "
                "traverser_action_sample_count."
            )
        if (
            self.traverser_action_sample_schedule is not None
            and self.traverser_action_priority_count
            > min(self.traverser_action_sample_schedule)
        ):
            raise ValueError(
                "traverser_action_priority_count cannot exceed the smallest "
                "traverser_action_sample_schedule entry."
            )
        if traverser_action_baseline not in {"none", "call"}:
            raise ValueError(
                "traverser_action_baseline must be 'none' or 'call'."
            )
        self.traverser_action_baseline = traverser_action_baseline
        if traverser_action_sample_mode not in {"random", "hash"}:
            raise ValueError(
                "traverser_action_sample_mode must be 'random' or 'hash'."
            )
        self.traverser_action_sample_mode = traverser_action_sample_mode
        if self.regret_increment_reach_mode in {"visit_fraction", "visit_count"} and (
            self.traverser_action_sample_count is not None
            or self.traverser_action_sample_fraction is not None
            or self.traverser_action_sample_schedule is not None
        ):
            raise ValueError("visit-based updates require full traverser-action expansion")
        self.traversal_streaming = bool(traversal_streaming)
        self.traversal_live_row_budget = (
            None
            if traversal_live_row_budget is None
            else int(traversal_live_row_budget)
        )
        if (
            self.traversal_live_row_budget is not None
            and self.traversal_live_row_budget <= 0
        ):
            raise ValueError("traversal_live_row_budget must be positive.")
        self.traverser_action_chunk_size = (
            None
            if traverser_action_chunk_size is None
            else int(traverser_action_chunk_size)
        )
        if (
            self.traverser_action_chunk_size is not None
            and self.traverser_action_chunk_size <= 0
        ):
            raise ValueError("traverser_action_chunk_size must be positive.")
        self.traversal_record_flush_size = int(traversal_record_flush_size)
        if self.traversal_record_flush_size <= 0:
            raise ValueError("traversal_record_flush_size must be positive.")
        if (
            self.traversal_backend != "gpu_native"
            and (
                self.traverser_action_sample_count is not None
                or self.traverser_action_sample_fraction is not None
                or self.traverser_action_sample_schedule is not None
                or self.traverser_action_baseline != "none"
                or self.traverser_action_sample_mode != "random"
                or self.traversal_streaming
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

        self.regret_nets = (
            [
                NeuralMLP(
                    self.encoder.input_dim,
                    self.encoder.action_dim,
                    self.regret_hidden_sizes,
                ).to(self.device)
                for _ in range(2)
            ]
            if self.use_regret_network else []
        )
        self.regret_reader = (
            NetworkRegretReader(self.regret_nets, self._forward)
            if self.use_regret_network else None
        )
        self.strategy_nets = (
            [
                NeuralMLP(
                    self.encoder.input_dim,
                    self.encoder.action_dim,
                    self.strategy_hidden_sizes,
                ).to(self.device)
                for _ in range(2)
            ]
            if self.use_strategy_network else []
        )
        self.regret_optimizers = [
            self._make_optimizer(model) for model in self.regret_nets
        ]
        self.strategy_optimizers = [
            self._make_optimizer(model) for model in self.strategy_nets
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
        for buffer in self.regret_buffers:
            buffer.require_no_overwrite = self.regret_target_mode in {
                "aggregate_then_clip", "aggregate_then_clip_on_read"
            }
        self.strategy_buffer_capacity = int(strategy_buffer_capacity)
        self.strategy_buffers = (
            [
                reservoir_cls(
                    strategy_buffer_capacity,
                    self.encoder.input_dim,
                    self.encoder.action_dim,
                    *reservoir_args,
                )
                for _ in range(2)
            ]
            if self.use_strategy_network else []
        )
        self.regret_validation_buffers = [
            recent_cls(
                validation_buffer_capacity,
                self.encoder.input_dim,
                self.encoder.action_dim,
                *recent_args,
            )
            for _ in range(2)
        ]
        self.strategy_validation_buffers = (
            [
                reservoir_cls(
                    validation_buffer_capacity,
                    self.encoder.input_dim,
                    self.encoder.action_dim,
                    *reservoir_args,
                )
                for _ in range(2)
            ]
            if self.use_strategy_network else []
        )

    def _validate_regret_target_mode(self, mode: str, positive_weight: float) -> None:
        if mode not in {
            "clip_each_record", "aggregate_then_clip", "clip_on_read",
            "aggregate_then_clip_on_read",
        }:
            raise ValueError("Unknown regret_target_mode.")
        if mode in {"clip_on_read", "aggregate_then_clip_on_read"} and positive_weight != 0.0:
            raise ValueError(f"{mode} requires regret_positive_weight=0 (plain MSE).")
        cuda_aggregate_supported = (
            self.device.type == "cuda"
            and self.spec in {_CUDA_AGGREGATE_SPEC, _CUDA_AGGREGATE_30_SPEC}
        )
        if mode in {"aggregate_then_clip", "aggregate_then_clip_on_read"} and (
            self.traversal_backend != "gpu_native"
            or (self.device.type != "cpu" and not cuda_aggregate_supported)
        ):
            raise ValueError(
                "aggregate_then_clip requires gpu_native traversal on CPU "
                "or CUDA with a validated 18- or 30-claim spec."
            )
        if self.regret_increment_reach_mode in {"visit_fraction", "visit_count"} and (
            mode != "aggregate_then_clip" or self.validation_fraction != 0.0
        ):
            raise ValueError(
                "visit-based updates require aggregate_then_clip and validation_fraction=0."
            )
        if (self.regret_increment_reach_mode == "visit_count"
                and self.regret_accumulation_mode != "cumulative"):
            raise ValueError("visit_count requires cumulative regret accumulation.")
        if self.regret_accumulation_mode == "cumulative" and mode not in {
            "aggregate_then_clip", "clip_on_read", "aggregate_then_clip_on_read"
        }:
            raise ValueError("cumulative regrets require aggregate_then_clip or clip_on_read.")

    def set_regret_target_mode(self, mode: str, *, regret_positive_weight: float) -> None:
        """Change the target/loss at an iteration boundary, preserving model state."""
        weight = float(regret_positive_weight)
        self._validate_regret_target_mode(mode, weight)
        self.regret_target_mode = mode
        self.regret_positive_weight = weight
        for buffer in self.regret_buffers:
            buffer.require_no_overwrite = mode in {
                "aggregate_then_clip", "aggregate_then_clip_on_read"
            }

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
        if not self.use_regret_network:
            raise RuntimeError("This trainer has no neural regret network")
        return self.regret_nets[0]

    @property
    def regret_net_p2(self) -> NeuralMLP:
        if not self.use_regret_network:
            raise RuntimeError("This trainer has no neural regret network")
        return self.regret_nets[1]

    @property
    def strategy_net_p1(self) -> NeuralMLP:
        if not self.use_strategy_network:
            raise RuntimeError("This trainer has no neural strategy network")
        return self.strategy_nets[0]

    @property
    def strategy_net_p2(self) -> NeuralMLP:
        if not self.use_strategy_network:
            raise RuntimeError("This trainer has no neural strategy network")
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
                values = self.regret_values_tensor(pid, x)
            values = values.float().cpu().numpy()
        return values.astype(np.float32, copy=False)

    def _snapshot_regret_values_from_features(self, pid: int, features: np.ndarray) -> np.ndarray:
        # Collection and fitting never overlap. The live network is therefore
        # already frozen for the complete traversal phase.
        return self._regret_values_from_features(pid, features)

    def make_regret_record(self, old_raw, advantage, legal_mask):
        """Convert a sampled advantage into the regret record for this trainer."""
        return make_regret_target(
            old_raw, advantage, legal_mask,
            iteration=self.iteration,
            accumulation_mode=self.regret_accumulation_mode,
            target_mode=self.regret_target_mode,
        )

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

    def regret_values_tensor(self, pid: int, features: torch.Tensor) -> torch.Tensor:
        """Read current raw regrets from the active network or table source."""
        if self.regret_reader is None:
            raise RuntimeError("No regret reader is active")
        return self.regret_reader.read(pid, features)

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
                        values = self.regret_values_tensor(pid, x)
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
                values = self.regret_values_tensor(infoset.pid, x)
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

    def _strategy_record_weight(self) -> float:
        if self.strategy_weighting == "uniform":
            return 1.0
        if self.strategy_weighting == "quadratic":
            return float(self.iteration) ** 2
        return float(self.iteration)

    def _add_strategy_record(
        self,
        pid: int,
        features: np.ndarray,
        strategy: np.ndarray,
        legal_mask: np.ndarray,
    ) -> None:
        if not self.use_strategy_network:
            return
        weight = self._strategy_record_weight()
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

        if not isinstance(training_buffer, (DeviceReservoirBuffer, DeviceRecentBuffer)):
            features_np = features.detach().cpu().numpy().astype(np.float32, copy=False)
            targets_np = targets.detach().cpu().numpy().astype(np.float32, copy=False)
            masks_np = legal_masks.detach().cpu().numpy().astype(bool, copy=False)
            if torch.is_tensor(weights):
                weights_np = (
                    weights.detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32, copy=False)
                )
                if weights_np.ndim == 0:
                    weights_np = np.full(n, float(weights_np), dtype=np.float32)
            else:
                weights_np = np.full(n, float(weights), dtype=np.float32)

            if self.validation_fraction <= 0.0:
                training_buffer.add_many(
                    features_np,
                    targets_np,
                    masks_np,
                    weights_np,
                    self.rng,
                )
                return

            use_validation = (
                np.fromiter(
                    (self.validation_rng.random() for _ in range(n)),
                    dtype=np.float64,
                    count=n,
                )
                < self.validation_fraction
            )
            if np.any(use_validation):
                validation_buffer.add_many(
                    features_np[use_validation],
                    targets_np[use_validation],
                    masks_np[use_validation],
                    weights_np[use_validation],
                    self.validation_rng,
                )
            use_training = ~use_validation
            if np.any(use_training):
                training_buffer.add_many(
                    features_np[use_training],
                    targets_np[use_training],
                    masks_np[use_training],
                    weights_np[use_training],
                    self.rng,
                )
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

            old_raw = self._snapshot_regret_values_from_features(traverser, features)
            target = self.make_regret_record(old_raw, instant_regret, legal_mask)
            self._add_regret_record(traverser, features, target, legal_mask)
            return node_value

        self._add_strategy_record(to_play, features, strategy, legal_mask)
        action = self._sample_action(legal, strategy)
        return self._traverse(history + (action,), p1_hand, p2_hand, traverser)

    def _train_regret(self, pid: int, traversals_per_player: int) -> float:
        if self.regret_target_mode in {"aggregate_then_clip", "aggregate_then_clip_on_read"}:
            reach_weighted = self.regret_increment_reach_mode in {"visit_fraction", "visit_count"}
            self._aggregate_regret_targets(
                self.regret_buffers[pid],
                model=self.regret_nets[pid] if reach_weighted else None,
                iteration=self.iteration,
                roots=traversals_per_player if reach_weighted else None,
                accumulation_mode=self.regret_accumulation_mode,
                reach_mode=self.regret_increment_reach_mode,
                clip_result=self.regret_target_mode == "aggregate_then_clip",
            )
            if not reach_weighted:
                self._aggregate_regret_targets(
                    self.regret_validation_buffers[pid],
                    clip_result=self.regret_target_mode == "aggregate_then_clip",
                )
        return self._train_model(
            self.regret_nets[pid],
            self.regret_optimizers[pid],
            self.regret_buffers[pid],
            self.regret_train_steps,
            strategy_loss=False,
        )

    @staticmethod
    def _aggregate_regret_targets(
        buffer: DeviceRecentBuffer,
        *,
        model: NeuralMLP | None = None,
        iteration: int = 1,
        roots: int | None = None,
        accumulation_mode: str = "normalized",
        reach_mode: str = "visit_fraction",
        clip_result: bool = True,
    ) -> None:
        """Mean raw updates per infoset; optionally scale fresh regret by visits or visits/K.

        Repeating the aggregate target on every visit keeps the production
        replay sampling and importance weights unchanged during fitting.
        """
        n = buffer.size
        if not n:
            buffer.last_group_count = 0
            return
        if roots is not None and (roots <= 0 or buffer.seen != n or model is None):
            raise ValueError("visit-based updates require all records from this iteration")
        if roots is not None and reach_mode not in {"visit_fraction", "visit_count"}:
            raise ValueError("Unknown visit-based reach mode")
        unique, inverse = torch.unique(
            buffer.features[:n], dim=0, return_inverse=True
        )
        groups = int(inverse.max().item()) + 1
        buffer.last_group_count = groups
        weights = buffer.weights[:n].double()
        totals = torch.zeros(groups, dtype=torch.float64, device=buffer.device)
        totals.index_add_(0, inverse, weights)
        weighted = torch.zeros(
            (groups, buffer.action_dim), dtype=torch.float64, device=buffer.device
        )
        weighted.index_add_(
            0, inverse, buffer.targets[:n].double() * weights[:, None]
        )
        grouped_raw = weighted / totals.clamp_min(1e-12)[:, None]
        if roots is not None:
            counts = torch.bincount(inverse, minlength=groups)
            if int(counts.max().item()) > roots:
                raise ValueError("An infoset has more visits than sampled roots")
            old_parts = []
            with torch.inference_mode():
                for start in range(0, groups, 8192):
                    old_parts.append(torch.relu(model(unique[start:start + 8192])).double())
            old = torch.cat(old_parts, dim=0)
            visit_multiplier = counts.double()
            if reach_mode == "visit_fraction":
                visit_multiplier = visit_multiplier / roots
            if accumulation_mode == "cumulative":
                if iteration <= 1:
                    old.zero_()
                fresh = grouped_raw - old
                grouped_raw = old + visit_multiplier[:, None] * fresh
            else:
                previous_scale = (iteration - 1.0) / iteration
                fresh = grouped_raw - previous_scale * old
                grouped_raw = previous_scale * old + visit_multiplier[:, None] * fresh
        grouped = torch.relu(grouped_raw) if clip_result else grouped_raw
        buffer.targets[:n] = grouped.index_select(0, inverse).float() * buffer.legal_masks[:n]

    def _train_strategy(self, pid: int) -> float:
        if not self.use_strategy_network:
            return 0.0
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
        for step in range(steps):
            if not strategy_loss and self.regret_fit_learning_rate is not None:
                lr = self.regret_fit_learning_rate
                if self.regret_fit_schedule == "cosine":
                    fraction = step / max(steps - 1, 1)
                    lr = 1e-4 + 0.5 * (lr - 1e-4) * (1 + math.cos(math.pi * fraction))
                for group in optimizer.param_groups:
                    group["lr"] = lr
            sample_size = self.batch_size if strategy_loss else self.regret_batch_size
            features, targets, masks, weights = buffer.sample(sample_size, self.rng)
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
                # clip_on_read requires zero positive weight: otherwise the
                # sign of a noisy sample biases its fitted conditional mean.
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
        result: Dict[str, object] = {"regret": [], "strategy": []}
        if self.use_regret_network:
            result["regret"] = [
                self._validation_metrics_for(
                    self.regret_nets[pid],
                    self.regret_validation_buffers[pid],
                    strategy_targets=False,
                    max_records=max_records,
                )
                for pid in (0, 1)
            ]
        if self.use_strategy_network:
            result["strategy"] = [
                self._validation_metrics_for(
                    self.strategy_nets[pid],
                    self.strategy_validation_buffers[pid],
                    strategy_targets=True,
                    max_records=max_records,
                )
                for pid in (0, 1)
            ]
        return result

    def run_iteration(self, *, traversals_per_player: int = 100) -> Dict[str, object]:
        self.iteration += 1
        strategy_seen_before = (
            [buffer.seen for buffer in self.strategy_buffers]
            if self.use_strategy_network else []
        )

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
            "streamed_edge_chunks": 0,
            "streamed_row_splits": 0,
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
                        "streamed_edge_chunks",
                        "streamed_row_splits",
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
            regret_losses[traverser] = self._train_regret(traverser, traversals_per_player)
            self._synchronize()
            regret_training_s += time.perf_counter() - start

        start = time.perf_counter()
        strategy_losses = (
            [self._train_strategy(pid) for pid in (0, 1)]
            if self.use_strategy_network else []
        )
        self._synchronize()
        strategy_training_s = time.perf_counter() - start

        regret_seen = [buffer.seen for buffer in self.regret_buffers]
        strategy_seen = (
            [buffer.seen for buffer in self.strategy_buffers]
            if self.use_strategy_network else []
        )
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
            "visited_infosets": [getattr(buffer, "last_group_count", None)
                                 for buffer in self.regret_buffers],
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
        if not self.use_strategy_network:
            raise RuntimeError("This trainer has no neural average-policy network")
        policy = NeuralPolicy(
            self.spec,
            hidden_sizes=self.strategy_hidden_sizes,
            device=self.device,
        )
        policy.model_p1.load_state_dict(self.strategy_nets[0].state_dict())
        policy.model_p2.load_state_dict(self.strategy_nets[1].state_dict())
        return policy.eval()

    def checkpoint_dict(self) -> Dict[str, object]:
        state = {
            "version": self.CHECKPOINT_VERSION,
            "spec": _spec_to_dict(self.spec),
            "config": {
                "regret_hidden_sizes": self.regret_hidden_sizes,
                "strategy_hidden_sizes": self.strategy_hidden_sizes,
                "seed": self.seed,
                "regret_buffer_capacity": self.regret_buffers[0].capacity,
                "strategy_buffer_capacity": (
                    self.strategy_buffers[0].capacity
                    if self.use_strategy_network else 0
                ),
                "learning_rate": self.learning_rate,
                "batch_size": self.batch_size,
                "regret_batch_size": self.regret_batch_size,
                "regret_fit_schedule": self.regret_fit_schedule,
                "regret_fit_learning_rate": self.regret_fit_learning_rate,
                "regret_train_steps": self.regret_train_steps,
                "strategy_train_steps": self.strategy_train_steps,
                "use_regret_network": self.use_regret_network,
                "use_strategy_network": self.use_strategy_network,
                "strategy_weighting": self.strategy_weighting,
                "regret_positive_weight": self.regret_positive_weight,
                "regret_target_mode": self.regret_target_mode,
                "regret_increment_reach_mode": self.regret_increment_reach_mode,
                "regret_accumulation_mode": self.regret_accumulation_mode,
                "validation_fraction": self.validation_fraction,
                "validation_buffer_capacity": self.validation_buffer_capacity,
                "traversal_backend": self.traversal_backend,
                "traversal_batch_size": self.traversal_batch_size,
                "traverser_action_sample_count": self.traverser_action_sample_count,
                "traverser_action_sample_fraction": self.traverser_action_sample_fraction,
                "traverser_action_full_first": self.traverser_action_full_first,
                "traverser_action_sample_schedule": self.traverser_action_sample_schedule,
                "traverser_action_priority_count": self.traverser_action_priority_count,
                "traverser_action_baseline": self.traverser_action_baseline,
                "traverser_action_sample_mode": self.traverser_action_sample_mode,
                "traversal_streaming": self.traversal_streaming,
                "traversal_live_row_budget": self.traversal_live_row_budget,
                "traverser_action_chunk_size": self.traverser_action_chunk_size,
                "traversal_record_flush_size": self.traversal_record_flush_size,
                "device_replay": self.device_replay,
                "fused_optimizer": self.fused_optimizer,
                "amp_dtype": self.amp_dtype,
                "compile_models": self.compile_models,
            },
            "iteration": self.iteration,
            "grad_scaler": self._grad_scaler.state_dict(),
            "random_state": self.rng.getstate(),
            "validation_random_state": self.validation_rng.getstate(),
            "torch_random_state": torch.get_rng_state(),
            "torch_cuda_random_state": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
        }
        if self.use_regret_network:
            state["regret_nets"] = [model.state_dict() for model in self.regret_nets]
            state["regret_optimizers"] = [opt.state_dict() for opt in self.regret_optimizers]
        if self.use_strategy_network:
            state["strategy_nets"] = [model.state_dict() for model in self.strategy_nets]
            state["strategy_optimizers"] = [opt.state_dict() for opt in self.strategy_optimizers]
            state["strategy_buffers"] = [buffer.state_dict() for buffer in self.strategy_buffers]
            state["strategy_validation_buffers"] = [
                buffer.state_dict() for buffer in self.strategy_validation_buffers
            ]
        return state

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
        # Keep the serialized replay on CPU while constructing device buffers.
        # Loading a large checkpoint directly onto CUDA temporarily duplicates
        # the reservoir and can OOM before the old allocation is released.
        state = torch.load(path, map_location="cpu", weights_only=False)
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
        config.setdefault("regret_target_mode", "clip_each_record")
        config.setdefault("regret_increment_reach_mode", "none")
        config.setdefault("regret_accumulation_mode", "normalized")
        config.setdefault("traversal_batch_size", 256)
        config.setdefault("traverser_action_sample_count", None)
        config.setdefault("traverser_action_sample_fraction", None)
        config.setdefault("traverser_action_full_first", False)
        config.setdefault("traverser_action_sample_schedule", None)
        config.setdefault("traverser_action_priority_count", 0)
        config.setdefault("traverser_action_baseline", "none")
        config.setdefault("traverser_action_sample_mode", "random")
        config.setdefault("traversal_streaming", False)
        config.setdefault("traversal_live_row_budget", None)
        config.setdefault("traverser_action_chunk_size", None)
        config.setdefault("traversal_record_flush_size", 131_072)
        config.setdefault("device_replay", False)
        config.setdefault("fused_optimizer", None)
        config.setdefault("amp_dtype", None)
        config.setdefault("compile_models", False)
        config.setdefault("use_regret_network", True)
        config.setdefault("use_strategy_network", True)
        trainer = cls(_spec_from_dict(state["spec"]), device=device, **config)
        trainer.iteration = int(state["iteration"])

        if trainer.use_regret_network:
            for model, model_state in zip(trainer.regret_nets, state["regret_nets"]):
                model.load_state_dict(model_state)
                model.eval()
            for optimizer, optimizer_state in zip(trainer.regret_optimizers, state["regret_optimizers"]):
                optimizer.load_state_dict(optimizer_state)
        if trainer.use_strategy_network:
            for model, model_state in zip(trainer.strategy_nets, state["strategy_nets"]):
                model.load_state_dict(model_state)
                model.eval()
            for optimizer, optimizer_state in zip(
                trainer.strategy_optimizers,
                state["strategy_optimizers"],
            ):
                optimizer.load_state_dict(optimizer_state)
        if "grad_scaler" in state:
            trainer._grad_scaler.load_state_dict(state["grad_scaler"])

        if trainer.use_strategy_network:
            def restore_into(existing, saved):
                if int(saved["capacity"]) != existing.capacity:
                    raise ValueError("Reservoir capacity differs from checkpoint")
                size = int(saved["size"])
                if trainer.device_replay:
                    # The constructor has already allocated the full GPU buffer.
                    # Copy into it directly, in chunks: constructing a second
                    # buffer or staging all saved rows on CUDA doubles peak use.
                    for name in ("features", "targets", "legal_masks", "weights"):
                        destination = getattr(existing, name)
                        source = saved[name]
                        for start in range(0, size, 131_072):
                            end = min(start + 131_072, size)
                            destination[start:end].copy_(source[start:end])
                    existing.size = size
                    existing.seen = int(saved["seen"])
                    return existing
                return ReservoirBuffer.from_state_dict(saved)

            trainer.strategy_buffers = [
                restore_into(existing, saved) for existing, saved in
                zip(trainer.strategy_buffers, state["strategy_buffers"])
            ]
            trainer.strategy_validation_buffers = [
                restore_into(existing, saved) for existing, saved in
                zip(trainer.strategy_validation_buffers,
                    state.get("strategy_validation_buffers", []))
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
