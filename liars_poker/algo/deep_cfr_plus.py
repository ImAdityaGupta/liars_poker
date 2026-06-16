from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import torch

from liars_poker.algo.deep_cfr import ReservoirBuffer, _spec_from_dict, _spec_to_dict
from liars_poker.core import GameSpec, generate_deck, possible_starting_hands
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
            "capacity": self.capacity,
            "input_dim": self.input_dim,
            "action_dim": self.action_dim,
            "features": self.features.copy(),
            "targets": self.targets.copy(),
            "legal_masks": self.legal_masks.copy(),
            "weights": self.weights.copy(),
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
        buffer.features[:] = state["features"]
        buffer.targets[:] = state["targets"]
        buffer.legal_masks[:] = state["legal_masks"]
        buffer.weights[:] = state["weights"]
        buffer.size = int(state["size"])
        buffer.seen = int(state["seen"])
        buffer.cursor = int(state["cursor"])
        return buffer


class FixedValidationBuffer:
    """Append-only validation buffer that stops changing once full."""

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
        rng: random.Random | None = None,
    ) -> None:
        _ = rng
        self.seen += 1
        if self.size >= self.capacity:
            return
        idx = self.size
        self.features[idx] = features
        self.targets[idx] = targets
        self.legal_masks[idx] = legal_mask
        self.weights[idx] = weight
        self.size += 1

    def sample(self, batch_size: int, rng: random.Random) -> Tuple[np.ndarray, ...]:
        n = min(batch_size, self.size)
        indices = np.fromiter((rng.randrange(self.size) for _ in range(n)), dtype=np.int64)
        return (
            self.features[indices],
            self.targets[indices],
            self.legal_masks[indices],
            self.weights[indices],
        )

    def state_dict(self) -> Dict[str, object]:
        return {
            "capacity": self.capacity,
            "input_dim": self.input_dim,
            "action_dim": self.action_dim,
            "features": self.features.copy(),
            "targets": self.targets.copy(),
            "legal_masks": self.legal_masks.copy(),
            "weights": self.weights.copy(),
            "size": self.size,
            "seen": self.seen,
        }

    @classmethod
    def from_state_dict(cls, state: Dict[str, object]) -> "FixedValidationBuffer":
        buffer = cls(
            int(state["capacity"]),
            int(state["input_dim"]),
            int(state["action_dim"]),
        )
        buffer.features[:] = state["features"]
        buffer.targets[:] = state["targets"]
        buffer.legal_masks[:] = state["legal_masks"]
        buffer.weights[:] = state["weights"]
        buffer.size = int(state["size"])
        buffer.seen = int(state["seen"])
        return buffer


class DeepCFRPlusTrainer:
    """External-sampling neural CFR+ with clipped cumulative-regret targets."""

    CHECKPOINT_VERSION = 1

    def __init__(
        self,
        spec: GameSpec,
        *,
        hidden_sizes: Sequence[int] = (256, 256),
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
        self.regret_train_steps = int(regret_train_steps)
        self.strategy_train_steps = int(strategy_train_steps)
        if strategy_weighting not in {"linear", "uniform"}:
            raise ValueError("strategy_weighting must be 'linear' or 'uniform'.")
        self.strategy_weighting = strategy_weighting
        self.regret_positive_weight = float(regret_positive_weight)
        self.validation_fraction = float(validation_fraction)
        self.validation_buffer_capacity = int(validation_buffer_capacity)
        self.iteration = 0

        self.regret_nets = [
            NeuralMLP(self.encoder.input_dim, self.encoder.action_dim, self.hidden_sizes).to(self.device)
            for _ in range(2)
        ]
        self.regret_snapshot_nets = [
            NeuralMLP(self.encoder.input_dim, self.encoder.action_dim, self.hidden_sizes).to(self.device)
            for _ in range(2)
        ]
        self.strategy_nets = [
            NeuralMLP(self.encoder.input_dim, self.encoder.action_dim, self.hidden_sizes).to(self.device)
            for _ in range(2)
        ]
        self.regret_optimizers = [
            torch.optim.Adam(model.parameters(), lr=self.learning_rate)
            for model in self.regret_nets
        ]
        self.strategy_optimizers = [
            torch.optim.Adam(model.parameters(), lr=self.learning_rate)
            for model in self.strategy_nets
        ]

        self.regret_buffers = [
            RecentBuffer(regret_buffer_capacity, self.encoder.input_dim, self.encoder.action_dim)
            for _ in range(2)
        ]
        self.strategy_buffers = [
            ReservoirBuffer(strategy_buffer_capacity, self.encoder.input_dim, self.encoder.action_dim)
            for _ in range(2)
        ]
        self.regret_validation_buffers = [
            FixedValidationBuffer(validation_buffer_capacity, self.encoder.input_dim, self.encoder.action_dim)
            for _ in range(2)
        ]
        self.strategy_validation_buffers = [
            ReservoirBuffer(validation_buffer_capacity, self.encoder.input_dim, self.encoder.action_dim)
            for _ in range(2)
        ]

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

    def _refresh_regret_snapshots(self) -> None:
        for snapshot, live in zip(self.regret_snapshot_nets, self.regret_nets):
            snapshot.load_state_dict(live.state_dict())
            snapshot.eval()

    def _regret_values_from_features(self, pid: int, features: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(features).to(self.device)
        with torch.no_grad():
            values = self.regret_nets[pid](x).cpu().numpy()
        return values.astype(np.float32, copy=False)

    def _snapshot_regret_values_from_features(self, pid: int, features: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(features).to(self.device)
        with torch.no_grad():
            values = self.regret_snapshot_nets[pid](x).cpu().numpy()
        return values.astype(np.float32, copy=False)

    def _strategy_from_features(
        self,
        pid: int,
        features: np.ndarray,
        legal: Tuple[int, ...],
        *,
        use_snapshot: bool = False,
    ) -> np.ndarray:
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

    def current_policy_dense(self, *, batch_size: int = 4096) -> DenseTabularPolicy:
        dense = DenseTabularPolicy(self.spec)
        hands = tuple(possible_starting_hands(self.spec))

        for hid in range(1 << self.encoder.k):
            history = tuple(action for action in range(self.encoder.k) if hid & (1 << action))
            legal = dense.legal_actions[hid]
            cols = [self._action_col(action) for action in legal]
            model = self.regret_nets[dense.pid_to_act(hid)]

            for start in range(0, len(hands), batch_size):
                stop = min(start + batch_size, len(hands))
                features = self.encoder.encode_hands(hands[start:stop], history)
                x = torch.from_numpy(features).to(self.device)
                with torch.no_grad():
                    values = model(x).cpu().numpy()

                positive = np.maximum(values[:, cols], 0.0)
                totals = positive.sum(axis=1, keepdims=True)
                probs = np.divide(
                    positive,
                    totals,
                    out=np.zeros_like(positive),
                    where=totals > 0.0,
                )
                fallback = totals[:, 0] <= 0.0
                if np.any(fallback):
                    probs[fallback] = 1.0 / len(cols)

                block = dense.S[hid, start:stop]
                block.fill(0.0)
                block[:, cols] = probs

        dense.recompute_likelihoods()
        return dense

    def regret_values(self, infoset: InfoSet) -> Dict[int, float]:
        legal = self.rules.legal_actions_for(infoset)
        x = torch.from_numpy(self.encoder.encode(infoset.hand, infoset.history)).to(self.device)
        with torch.no_grad():
            values = self.regret_nets[infoset.pid](x).cpu().numpy()
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
        total_loss = 0.0
        for _ in range(steps):
            features, targets, masks, weights = buffer.sample(self.batch_size, self.rng)
            x = torch.from_numpy(features).to(self.device)
            y = torch.from_numpy(targets).to(self.device)
            mask = torch.from_numpy(masks).to(self.device)
            weight = torch.from_numpy(weights).to(self.device)
            weight = weight / weight.mean().clamp_min(1e-8)

            pred = model(x)
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

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu())

        model.eval()
        return total_loss / steps

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

        x = torch.from_numpy(buffer.features[:size]).to(self.device)
        targets = torch.from_numpy(buffer.targets[:size]).to(self.device)
        mask = torch.from_numpy(buffer.legal_masks[:size]).to(self.device)
        mask_float = mask.float()
        weights = torch.from_numpy(buffer.weights[:size]).to(self.device)
        weights = weights / weights.mean().clamp_min(1e-8)

        with torch.no_grad():
            pred = model(x)
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
        for traverser in (0, 1):
            self.regret_buffers[traverser].clear()
            self._refresh_regret_snapshots()
            start = time.perf_counter()
            for _ in range(traversals_per_player):
                p1_hand, p2_hand = self._deal()
                self._traverse((), p1_hand, p2_hand, traverser)
            traversal_s += time.perf_counter() - start

            start = time.perf_counter()
            regret_losses[traverser] = self._train_regret(traverser)
            regret_training_s += time.perf_counter() - start

        start = time.perf_counter()
        strategy_losses = [self._train_strategy(pid) for pid in (0, 1)]
        strategy_training_s = time.perf_counter() - start

        regret_seen = [buffer.seen for buffer in self.regret_buffers]
        strategy_seen = [buffer.seen for buffer in self.strategy_buffers]
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
            "timing": {
                "traversal_s": traversal_s,
                "regret_training_s": regret_training_s,
                "strategy_training_s": strategy_training_s,
            },
        }

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
            },
            "iteration": self.iteration,
            "regret_nets": [model.state_dict() for model in self.regret_nets],
            "strategy_nets": [model.state_dict() for model in self.strategy_nets],
            "regret_optimizers": [opt.state_dict() for opt in self.regret_optimizers],
            "strategy_optimizers": [opt.state_dict() for opt in self.strategy_optimizers],
            "regret_buffers": [buffer.state_dict() for buffer in self.regret_buffers],
            "strategy_buffers": [buffer.state_dict() for buffer in self.strategy_buffers],
            "regret_validation_buffers": [
                buffer.state_dict() for buffer in self.regret_validation_buffers
            ],
            "strategy_validation_buffers": [
                buffer.state_dict() for buffer in self.strategy_validation_buffers
            ],
            "random_state": self.rng.getstate(),
            "validation_random_state": self.validation_rng.getstate(),
            "torch_random_state": torch.get_rng_state(),
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
        trainer = cls(_spec_from_dict(state["spec"]), device=device, **dict(state["config"]))
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

        trainer.regret_buffers = [
            RecentBuffer.from_state_dict(buffer_state)
            for buffer_state in state["regret_buffers"]
        ]
        trainer.strategy_buffers = [
            ReservoirBuffer.from_state_dict(buffer_state)
            for buffer_state in state["strategy_buffers"]
        ]
        trainer.regret_validation_buffers = [
            FixedValidationBuffer.from_state_dict(buffer_state)
            for buffer_state in state["regret_validation_buffers"]
        ]
        trainer.strategy_validation_buffers = [
            ReservoirBuffer.from_state_dict(buffer_state)
            for buffer_state in state["strategy_validation_buffers"]
        ]
        trainer.rng.setstate(state["random_state"])
        trainer.validation_rng.setstate(state["validation_random_state"])
        torch.set_rng_state(state["torch_random_state"])
        return trainer
