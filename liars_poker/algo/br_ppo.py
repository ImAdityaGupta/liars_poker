from __future__ import annotations

import time
from typing import Dict, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from liars_poker.algo.br_neural import NeuralBRTrainer
from liars_poker.core import GameSpec
from liars_poker.policies.base import Policy
from liars_poker.policies.neural import NeuralMLP, NeuralPolicy


class PPOBRTrainer(NeuralBRTrainer):
    """On-policy PPO responder against a fixed opponent."""

    def __init__(
        self,
        spec: GameSpec,
        opponent: Policy,
        *,
        hidden_sizes: Sequence[int] = (512, 512),
        value_hidden_sizes: Sequence[int] | None = None,
        device: str | torch.device | None = None,
        learning_rate: float = 3e-4,
        value_learning_rate: float | None = None,
        ppo_epochs: int = 4,
        minibatch_size: int = 4096,
        clip_ratio: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        seed: int = 0,
    ) -> None:
        super().__init__(
            spec,
            opponent,
            hidden_sizes=(8,),
            device=device,
            replay_capacity=1,
            batch_size=1,
            learning_rate=learning_rate,
            train_steps=1,
            warmup_transitions=1,
            seed=seed,
        )
        self.actor_hidden_sizes = tuple(int(size) for size in hidden_sizes)
        self.value_hidden_sizes = tuple(
            int(size)
            for size in (
                hidden_sizes if value_hidden_sizes is None else value_hidden_sizes
            )
        )
        self.ppo_epochs = int(ppo_epochs)
        self.minibatch_size = int(minibatch_size)
        self.clip_ratio = float(clip_ratio)
        self.entropy_coef = float(entropy_coef)
        self.value_coef = float(value_coef)

        self.actor_nets = [
            NeuralMLP(
                self.encoder.input_dim,
                self.encoder.action_dim,
                self.actor_hidden_sizes,
            ).to(self.device)
            for _ in range(2)
        ]
        self.value_nets = [
            NeuralMLP(
                self.encoder.input_dim,
                1,
                self.value_hidden_sizes,
            ).to(self.device)
            for _ in range(2)
        ]
        self.actor_optimizers = [
            torch.optim.Adam(model.parameters(), lr=learning_rate)
            for model in self.actor_nets
        ]
        value_lr = learning_rate if value_learning_rate is None else value_learning_rate
        self.value_optimizers = [
            torch.optim.Adam(model.parameters(), lr=value_lr)
            for model in self.value_nets
        ]

    def _actor_actions(
        self,
        role: int,
        features: torch.Tensor,
        legal: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.actor_nets[role](features).float().masked_fill(
            ~legal,
            -torch.inf,
        )
        probs = torch.softmax(logits, dim=1)
        actions = torch.multinomial(
            probs,
            1,
            generator=self.generator,
        ).squeeze(1)
        log_probs = torch.log(
            probs.gather(1, actions[:, None]).squeeze(1).clamp_min(1e-12)
        )
        return actions, log_probs, logits

    def _collect_batch_ppo(
        self,
        role: int,
        episodes: int,
    ) -> tuple[Dict[str, int], Dict[str, torch.Tensor]]:
        p1_counts, p2_counts, total_counts = self._sample_deals(episodes)
        deal_idx = torch.arange(episodes, device=self.device)
        hids = torch.zeros(episodes, dtype=torch.long, device=self.device)
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
                hids,
                last_claim,
                p1_counts,
                p2_counts,
            )
            claims = opponent_cols - 1
            hids = torch.ones_like(claims) << claims
            last_claim = claims

        episode_returns = torch.empty(
            episodes,
            dtype=torch.float32,
            device=self.device,
        )
        feature_parts = []
        action_parts = []
        legal_parts = []
        log_prob_parts = []
        value_parts = []
        deal_parts = []

        wins = 0
        decisions = 0
        active_deals = deal_idx
        active_hids = hids
        active_last = last_claim

        while active_deals.numel():
            features = self._features(
                role,
                active_deals,
                active_hids,
                p1_counts,
                p2_counts,
            )
            legal = self.legal_masks.index_select(0, active_last + 1)
            with torch.no_grad():
                actions, log_probs, _ = self._actor_actions(role, features, legal)
                values = self.value_nets[role](features).squeeze(1)

            feature_parts.append(features)
            action_parts.append(actions)
            legal_parts.append(legal)
            log_prob_parts.append(log_probs)
            value_parts.append(values)
            deal_parts.append(active_deals)
            decisions += len(features)

            (
                rewards,
                dones,
                next_hids,
                next_last,
                _,
                _,
            ) = self._advance_actions(
                role,
                active_deals,
                active_hids,
                active_last,
                actions,
                p1_counts,
                p2_counts,
                total_counts,
            )
            episode_returns[active_deals[dones]] = rewards[dones]
            wins += int((dones & (rewards > 0)).sum().item())
            keep = ~dones
            active_deals = active_deals[keep]
            active_hids = next_hids[keep]
            active_last = next_last[keep]

        record_deals = torch.cat(deal_parts)
        batch = {
            "features": torch.cat(feature_parts),
            "actions": torch.cat(action_parts),
            "legal": torch.cat(legal_parts),
            "old_log_probs": torch.cat(log_prob_parts),
            "old_values": torch.cat(value_parts),
            "returns": episode_returns.index_select(0, record_deals),
        }
        return {
            "episodes": episodes,
            "wins": wins,
            "decisions": decisions,
        }, batch

    def collect_role(
        self,
        role: int,
        episodes: int,
        *,
        rollout_batch_size: int = 8192,
    ) -> tuple[Dict[str, int | float], Dict[str, torch.Tensor]]:
        start = time.perf_counter()
        totals = {"episodes": 0, "wins": 0, "decisions": 0}
        batches: Dict[str, list[torch.Tensor]] = {}
        remaining = int(episodes)
        while remaining:
            batch_size = min(remaining, int(rollout_batch_size))
            stats, batch = self._collect_batch_ppo(role, batch_size)
            for key in totals:
                totals[key] += stats[key]
            for key, value in batch.items():
                batches.setdefault(key, []).append(value)
            remaining -= batch_size
        combined = {key: torch.cat(parts) for key, parts in batches.items()}
        return {
            **totals,
            "collect_s": time.perf_counter() - start,
        }, combined

    def train_role(
        self,
        role: int,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        features = batch["features"]
        actions = batch["actions"]
        legal = batch["legal"]
        old_log_probs = batch["old_log_probs"]
        old_values = batch["old_values"]
        returns = batch["returns"]
        advantages = returns - old_values
        advantages = (advantages - advantages.mean()) / advantages.std().clamp_min(1e-6)

        actor = self.actor_nets[role]
        critic = self.value_nets[role]
        actor_optimizer = self.actor_optimizers[role]
        value_optimizer = self.value_optimizers[role]
        actor.train()
        critic.train()
        actor_losses = []
        value_losses = []
        entropies = []
        n = len(features)

        for _ in range(self.ppo_epochs):
            order = torch.randperm(n, device=self.device, generator=self.generator)
            for start in range(0, n, self.minibatch_size):
                indices = order[start : start + self.minibatch_size]
                x = features.index_select(0, indices)
                mask = legal.index_select(0, indices)
                chosen = actions.index_select(0, indices)
                old_lp = old_log_probs.index_select(0, indices)
                adv = advantages.index_select(0, indices)
                target_values = returns.index_select(0, indices)

                logits = actor(x).float().masked_fill(~mask, -torch.inf)
                log_all = torch.log_softmax(logits, dim=1)
                probs = torch.softmax(logits, dim=1)
                new_lp = log_all.gather(1, chosen[:, None]).squeeze(1)
                ratio = torch.exp(new_lp - old_lp)
                clipped = ratio.clamp(1.0 - self.clip_ratio, 1.0 + self.clip_ratio)
                policy_loss = -torch.minimum(ratio * adv, clipped * adv).mean()
                safe_log_all = torch.where(mask, log_all, torch.zeros_like(log_all))
                entropy = -(probs * safe_log_all).sum(dim=1).mean()
                values = critic(x).squeeze(1)
                value_loss = F.mse_loss(values, target_values)
                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy
                )

                actor_optimizer.zero_grad(set_to_none=True)
                value_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                actor_optimizer.step()
                value_optimizer.step()
                actor_losses.append(float(policy_loss.detach()))
                value_losses.append(float(value_loss.detach()))
                entropies.append(float(entropy.detach()))

        actor.eval()
        critic.eval()
        return {
            "policy_loss": float(np.mean(actor_losses)),
            "value_loss": float(np.mean(value_losses)),
            "entropy": float(np.mean(entropies)),
        }

    def run_iteration(
        self,
        *,
        episodes_per_role: int = 8192,
        rollout_batch_size: int = 8192,
    ) -> Dict[str, object]:
        self.iteration += 1
        role_records = []
        for role in (0, 1):
            collected, batch = self.collect_role(
                role,
                episodes_per_role,
                rollout_batch_size=rollout_batch_size,
            )
            start = time.perf_counter()
            losses = self.train_role(role, batch)
            collected["fit_s"] = time.perf_counter() - start
            collected.update(losses)
            role_records.append(collected)
        return {"iter": self.iteration, "roles": role_records}

    def _evaluate_batch(self, role: int, episodes: int) -> Dict[str, int]:
        p1_counts, p2_counts, total_counts = self._sample_deals(episodes)
        deal_idx = torch.arange(episodes, device=self.device)
        hids = torch.zeros(episodes, dtype=torch.long, device=self.device)
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
                hids,
                last_claim,
                p1_counts,
                p2_counts,
            )
            claims = opponent_cols - 1
            hids = torch.ones_like(claims) << claims
            last_claim = claims

        wins = 0
        active_deals = deal_idx
        active_hids = hids
        active_last = last_claim
        while active_deals.numel():
            features = self._features(
                role,
                active_deals,
                active_hids,
                p1_counts,
                p2_counts,
            )
            legal = self.legal_masks.index_select(0, active_last + 1)
            with torch.no_grad():
                actions, _, _ = self._actor_actions(role, features, legal)
            (
                rewards,
                dones,
                next_hids,
                next_last,
                _,
                _,
            ) = self._advance_actions(
                role,
                active_deals,
                active_hids,
                active_last,
                actions,
                p1_counts,
                p2_counts,
                total_counts,
            )
            wins += int((dones & (rewards > 0)).sum().item())
            keep = ~dones
            active_deals = active_deals[keep]
            active_hids = next_hids[keep]
            active_last = next_last[keep]
        return {"episodes": episodes, "wins": wins}

    def evaluate_role(
        self,
        role: int,
        episodes: int,
        *,
        rollout_batch_size: int = 8192,
    ) -> Dict[str, float]:
        total_wins = 0
        remaining = int(episodes)
        while remaining:
            batch = min(remaining, int(rollout_batch_size))
            result = self._evaluate_batch(role, batch)
            total_wins += result["wins"]
            remaining -= batch
        return {
            "episodes": int(episodes),
            "wins": total_wins,
            "win_rate": total_wins / max(1, int(episodes)),
        }

    def policy(self) -> NeuralPolicy:
        policy = NeuralPolicy(
            self.spec,
            hidden_sizes=self.actor_hidden_sizes,
            device=self.device,
        )
        policy.model_p1.load_state_dict(self.actor_nets[0].state_dict())
        policy.model_p2.load_state_dict(self.actor_nets[1].state_dict())
        return policy.eval()
