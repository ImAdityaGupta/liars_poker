from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from liars_poker.algo.br_fitted_return import FittedReturnBRTrainer
from liars_poker.core import GameSpec
from liars_poker.policies.action_conditioned import (
    ActionConditionedQPolicy,
    ActionConditionedScorer,
    ActionFeatureEncoder,
)
from liars_poker.policies.base import Policy


class ActionConditionedFittedReturnBRTrainer(FittedReturnBRTrainer):
    """Complete-return responder with a shared state-action value scorer."""

    def __init__(
        self,
        spec: GameSpec,
        opponent: Policy,
        *,
        state_hidden_sizes: Sequence[int] = (512, 512),
        action_hidden_sizes: Sequence[int] = (128, 128),
        embedding_dim: int = 256,
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
        fused_optimizer: bool | None = None,
        seed: int = 0,
    ) -> None:
        super().__init__(
            spec,
            opponent,
            hidden_sizes=(8,),
            device=device,
            replay_capacity=replay_capacity,
            batch_size=batch_size,
            learning_rate=learning_rate,
            train_steps=train_steps,
            warmup_transitions=warmup_transitions,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_decay_decisions=epsilon_decay_decisions,
            rollouts_per_action=rollouts_per_action,
            seed=seed,
        )
        self.state_hidden_sizes = tuple(int(size) for size in state_hidden_sizes)
        self.action_hidden_sizes = tuple(int(size) for size in action_hidden_sizes)
        self.embedding_dim = int(embedding_dim)
        self.action_encoder = ActionFeatureEncoder(spec)
        self.action_features = self.action_encoder.tensor(self.device)
        self.fused_optimizer = (
            self.device.type == "cuda"
            if fused_optimizer is None
            else bool(fused_optimizer)
        )

        self.q_nets = [self._new_model().to(self.device) for _ in range(2)]
        for model in self.q_nets:
            model.initialize_action_neutral()
            model.eval()
            model.cache_actions(self.action_features)
        self.target_nets = []
        optimizer_kwargs = {"lr": self.learning_rate}
        if self.fused_optimizer and self.device.type == "cuda":
            optimizer_kwargs["fused"] = True
        try:
            self.optimizers = [
                torch.optim.Adam(model.parameters(), **optimizer_kwargs)
                for model in self.q_nets
            ]
        except TypeError:
            optimizer_kwargs.pop("fused", None)
            self.optimizers = [
                torch.optim.Adam(model.parameters(), **optimizer_kwargs)
                for model in self.q_nets
            ]

    def _new_model(self) -> ActionConditionedScorer:
        return ActionConditionedScorer(
            self.encoder.input_dim,
            self.action_encoder.feature_dim,
            state_hidden_sizes=self.state_hidden_sizes,
            action_hidden_sizes=self.action_hidden_sizes,
            embedding_dim=self.embedding_dim,
        )

    def _select_actions(
        self,
        role: int,
        features: torch.Tensor,
        legal_mask: torch.Tensor,
        epsilon: float,
    ) -> torch.Tensor:
        with torch.inference_mode():
            q_values = self.q_nets[role].score_all_cached(features)
        greedy = q_values.masked_fill(~legal_mask, -torch.inf).argmax(dim=1)
        if epsilon <= 0.0:
            return greedy

        random_probs = legal_mask.float()
        random_probs /= random_probs.sum(dim=1, keepdim=True)
        random_cols = torch.multinomial(
            random_probs,
            1,
            generator=self.generator,
        ).squeeze(1)
        explore = torch.rand(
            len(features),
            device=self.device,
            generator=self.generator,
        ) < epsilon
        return torch.where(explore, random_cols, greedy)

    def train_role(self, role: int, *, steps: int | None = None) -> float:
        buffer = self.replay[role]
        if buffer.size < max(self.warmup_transitions, self.batch_size):
            return float("nan")

        model = self.q_nets[role]
        optimizer = self.optimizers[role]
        losses = []
        model.train()
        for _ in range(self.train_steps if steps is None else int(steps)):
            features, actions, returns = buffer.sample(
                self.batch_size,
                self.generator,
            )
            predicted = model.score_selected(
                features,
                self.action_features,
                actions,
            )
            loss = F.smooth_l1_loss(predicted, returns)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            self.optimizer_steps[role] += 1
            losses.append(float(loss.detach()))
        model.eval()
        model.cache_actions(self.action_features)
        return float(np.mean(losses))

    def policy(self) -> ActionConditionedQPolicy:
        policy = ActionConditionedQPolicy(
            self.spec,
            state_hidden_sizes=self.state_hidden_sizes,
            action_hidden_sizes=self.action_hidden_sizes,
            embedding_dim=self.embedding_dim,
            device=self.device,
        )
        policy.model_p1.load_state_dict(self.q_nets[0].state_dict())
        policy.model_p2.load_state_dict(self.q_nets[1].state_dict())
        return policy.eval()
