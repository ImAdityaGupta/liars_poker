from __future__ import annotations

import time
from typing import Dict, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from liars_poker.algo.br_neural import NeuralBRTrainer
from liars_poker.core import GameSpec
from liars_poker.policies.base import Policy


class DeviceReturnReplay:
    """Device-resident ring buffer of complete-return action targets."""

    def __init__(
        self,
        capacity: int,
        input_dim: int,
        device: torch.device,
    ) -> None:
        self.capacity = int(capacity)
        self.input_dim = int(input_dim)
        self.device = device
        self.features = torch.empty(
            (capacity, input_dim),
            dtype=torch.float32,
            device=device,
        )
        self.actions = torch.empty(capacity, dtype=torch.long, device=device)
        self.returns = torch.empty(capacity, dtype=torch.float32, device=device)
        self.size = 0
        self.position = 0
        self.seen = 0

    def add_many(
        self,
        features: torch.Tensor,
        actions: torch.Tensor,
        returns: torch.Tensor,
    ) -> None:
        n = int(features.shape[0])
        if n <= 0:
            return
        self.seen += n
        if n >= self.capacity:
            features = features[-self.capacity :]
            actions = actions[-self.capacity :]
            returns = returns[-self.capacity :]
            n = self.capacity

        first = min(n, self.capacity - self.position)
        second = n - first
        sl = slice(self.position, self.position + first)
        self.features[sl].copy_(features[:first])
        self.actions[sl].copy_(actions[:first])
        self.returns[sl].copy_(returns[:first])
        if second:
            sl = slice(0, second)
            self.features[sl].copy_(features[first:])
            self.actions[sl].copy_(actions[first:])
            self.returns[sl].copy_(returns[first:])

        self.position = (self.position + n) % self.capacity
        self.size = min(self.capacity, self.size + n)

    def sample(
        self,
        batch_size: int,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        indices = torch.randint(
            self.size,
            (int(batch_size),),
            device=self.device,
            generator=generator,
        )
        return (
            self.features.index_select(0, indices),
            self.actions.index_select(0, indices),
            self.returns.index_select(0, indices),
        )


class FittedReturnBRTrainer(NeuralBRTrainer):
    """Greedy fitted response trained on complete terminal-return targets."""

    def __init__(
        self,
        spec: GameSpec,
        opponent: Policy,
        *,
        hidden_sizes: Sequence[int] = (512, 512),
        device: str | torch.device | None = None,
        replay_capacity: int = 1_000_000,
        batch_size: int = 4096,
        learning_rate: float = 1e-3,
        train_steps: int = 100,
        warmup_transitions: int = 20_000,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_decisions: int = 500_000,
        rollouts_per_action: int = 1,
        seed: int = 0,
    ) -> None:
        super().__init__(
            spec,
            opponent,
            hidden_sizes=hidden_sizes,
            device=device,
            expansion="sampled",
            replay_capacity=1,
            batch_size=batch_size,
            learning_rate=learning_rate,
            train_steps=train_steps,
            warmup_transitions=warmup_transitions,
            target_update_every=2**62,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_decay_decisions=epsilon_decay_decisions,
            seed=seed,
        )
        if rollouts_per_action <= 0:
            raise ValueError("rollouts_per_action must be positive.")
        self.rollouts_per_action = int(rollouts_per_action)
        self.replay = [
            DeviceReturnReplay(
                replay_capacity,
                self.encoder.input_dim,
                self.device,
            )
            for _ in range(2)
        ]

    def _complete_returns(
        self,
        role: int,
        deal_idx: torch.Tensor,
        histories: torch.Tensor,
        last_claim: torch.Tensor,
        action_cols: torch.Tensor,
        p1_counts: torch.Tensor,
        p2_counts: torch.Tensor,
        total_counts: torch.Tensor,
    ) -> torch.Tensor:
        repeats = self.rollouts_per_action
        base_n = len(action_cols)
        if repeats > 1:
            deal_idx = deal_idx.repeat_interleave(repeats)
            histories = self.history.repeat_interleave(histories, repeats)
            last_claim = last_claim.repeat_interleave(repeats)
            action_cols = action_cols.repeat_interleave(repeats)

        (
            rewards,
            dones,
            next_histories,
            next_last,
            _,
            _,
        ) = self._advance_actions(
            role,
            deal_idx,
            histories,
            last_claim,
            action_cols,
            p1_counts,
            p2_counts,
            total_counts,
        )
        returns = torch.empty(len(action_cols), dtype=torch.float32, device=self.device)
        returns[dones] = rewards[dones]

        active_rows = (~dones).nonzero(as_tuple=False).squeeze(1)
        active_deals = deal_idx.index_select(0, active_rows)
        active_histories = self.history.select(next_histories, active_rows)
        active_last = next_last.index_select(0, active_rows)

        while active_rows.numel():
            features = self._features(
                role,
                active_deals,
                active_histories,
                p1_counts,
                p2_counts,
            )
            legal = self.legal_masks.index_select(0, active_last + 1)
            chosen = self._select_actions(role, features, legal, epsilon=0.0)
            (
                rewards,
                dones,
                next_histories,
                next_last,
                _,
                _,
            ) = self._advance_actions(
                role,
                active_deals,
                active_histories,
                active_last,
                chosen,
                p1_counts,
                p2_counts,
                total_counts,
            )
            returns[active_rows[dones]] = rewards[dones]
            keep = ~dones
            active_rows = active_rows[keep]
            active_deals = active_deals[keep]
            keep_rows = keep.nonzero(as_tuple=False).squeeze(1)
            active_histories = self.history.select(next_histories, keep_rows)
            active_last = next_last[keep]

        if repeats > 1:
            returns = returns.reshape(base_n, repeats).mean(dim=1)
        return returns

    def _collect_batch_returns(self, role: int, episodes: int) -> Dict[str, int]:
        p1_counts, p2_counts, total_counts = self._sample_deals(episodes)
        deal_idx = torch.arange(episodes, device=self.device)
        histories = self.history.zeros(episodes)
        last_claim = torch.full(
            (episodes,),
            -1,
            dtype=torch.long,
            device=self.device,
        )

        if role == 1:
            opponent_cols = self._opponent_actions(
                0,
                deal_idx,
                histories,
                last_claim,
                p1_counts,
                p2_counts,
            )
            claims = opponent_cols - 1
            histories = self.history.from_claims(claims)
            last_claim = claims

        wins = 0
        decisions = 0
        targets = 0
        epsilon = self.epsilon(role)
        active_deals = deal_idx
        active_histories = histories
        active_last = last_claim

        while active_deals.numel():
            features = self._features(
                role,
                active_deals,
                active_histories,
                p1_counts,
                p2_counts,
            )
            legal = self.legal_masks.index_select(0, active_last + 1)
            chosen = self._select_actions(role, features, legal, epsilon)
            decisions += len(features)

            edges = legal.nonzero(as_tuple=False)
            parent_rows = edges[:, 0]
            action_cols = edges[:, 1]
            branch_features = features.index_select(0, parent_rows)
            returns = self._complete_returns(
                role,
                active_deals.index_select(0, parent_rows),
                self.history.select(active_histories, parent_rows),
                active_last.index_select(0, parent_rows),
                action_cols,
                p1_counts,
                p2_counts,
                total_counts,
            )
            self.replay[role].add_many(branch_features, action_cols, returns)
            targets += len(action_cols)

            (
                rewards,
                dones,
                next_histories,
                next_last,
                _,
                _,
            ) = self._advance_actions(
                role,
                active_deals,
                active_histories,
                active_last,
                chosen,
                p1_counts,
                p2_counts,
                total_counts,
            )
            wins += int((dones & (rewards > 0)).sum().item())
            keep = ~dones
            active_deals = active_deals[keep]
            keep_rows = keep.nonzero(as_tuple=False).squeeze(1)
            active_histories = self.history.select(next_histories, keep_rows)
            active_last = next_last[keep]

        self.decisions_seen[role] += decisions
        return {
            "episodes": episodes,
            "wins": wins,
            "decisions": decisions,
            "targets": targets,
        }

    def collect_role(
        self,
        role: int,
        episodes: int,
        *,
        rollout_batch_size: int = 1024,
    ) -> Dict[str, int | float]:
        start = time.perf_counter()
        totals = {"episodes": 0, "wins": 0, "decisions": 0, "targets": 0}
        remaining = int(episodes)
        while remaining:
            batch = min(remaining, int(rollout_batch_size))
            stats = self._collect_batch_returns(role, batch)
            for key in totals:
                totals[key] += stats[key]
            remaining -= batch
        return {
            **totals,
            "epsilon": self.epsilon(role),
            "collect_s": time.perf_counter() - start,
        }

    def train_role(self, role: int, *, steps: int | None = None) -> float:
        buffer = self.replay[role]
        if buffer.size < max(self.warmup_transitions, self.batch_size):
            return float("nan")

        model = self.q_nets[role]
        optimizer = self.optimizers[role]
        losses = []
        model.train()
        for _ in range(self.train_steps if steps is None else int(steps)):
            features, actions, returns = buffer.sample(self.batch_size, self.generator)
            predicted = model(features).gather(1, actions[:, None]).squeeze(1)
            loss = F.smooth_l1_loss(predicted, returns)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            self.optimizer_steps[role] += 1
            losses.append(float(loss.detach()))
        model.eval()
        return float(np.mean(losses))

    def run_iteration(
        self,
        *,
        episodes_per_role: int = 1024,
        rollout_batch_size: int = 1024,
    ) -> Dict[str, object]:
        self.iteration += 1
        role_records = []
        for role in (0, 1):
            collected = self.collect_role(
                role,
                episodes_per_role,
                rollout_batch_size=rollout_batch_size,
            )
            start = time.perf_counter()
            loss = self.train_role(role)
            collected["fit_s"] = time.perf_counter() - start
            collected["loss"] = loss
            collected["replay_size"] = self.replay[role].size
            collected["replay_seen"] = self.replay[role].seen
            role_records.append(collected)
        return {"iter": self.iteration, "roles": role_records}
