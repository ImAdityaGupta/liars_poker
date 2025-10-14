from __future__ import annotations

import random
from typing import Dict, List, Tuple

from liars_poker.algo.br_mc import best_response_mc
from liars_poker.core import GameSpec
from liars_poker.env import rules_for_spec
from liars_poker.io.run_manager import RunManager
from liars_poker.policies.base import Policy
from liars_poker.policies.commit_once import CommitOnceMixture
from liars_poker.training.configs import FSPConfig
from liars_poker.training.schedules import constant_eta, harmonic_eta


class FSPTrainer:
    def __init__(self, run: RunManager, config: FSPConfig) -> None:
        self.run = run
        self.config = config
        self.spec: GameSpec = run.spec
        self.rules = rules_for_spec(self.spec)
        self._rng = random.Random(config.seed)

    def step(self, opponent_policy: Policy, iter_index: int):
        if iter_index > self.config.max_iters:
            raise ValueError("Iteration index exceeds configured max_iters.")

        avg_id = self.run.current_policy_id()
        avg_policy = self.run.current_policy()

        opponent_policy.bind_rules(self.rules)
        avg_policy.bind_rules(self.rules)

        br_seed = self.config.seed + iter_index + 1
        br_policy = best_response_mc(
            self.spec,
            opponent_policy,
            episodes=self.config.episodes0,
            epsilon=self.config.epsilon,
            min_visits_per_action=self.config.min_visits,
            alternate_seats=True,
            seed=br_seed,
            annotate="memory",
        )

        br_id = self.run.log_policy(
            br_policy,
            role="best_response",
            parents=[{"id": avg_id, "role": "avg", "weight": 1.0}],
            mixing=None,
            seed=br_seed,
            train={
                "algo": "best_response_mc",
                "episodes": self.config.episodes0,
                "epsilon": self.config.epsilon,
                "min_visits": self.config.min_visits,
            },
        )

        eta = self._eta(iter_index)
        new_avg_policy = self._mix_policies(avg_policy, br_policy, eta)
        mixing_meta = {"impl": self.config.mix_impl, "eta": eta, "iter": iter_index}
        avg_train_meta = {
            "algo": "fsp_average",
            "eta_schedule": self.config.eta_schedule,
            "iter": iter_index,
        }
        new_avg_id = self.run.log_policy(
            new_avg_policy,
            role="average",
            parents=[
                {"id": avg_id, "role": "avg", "weight": max(0.0, 1.0 - eta)},
                {"id": br_id, "role": "br", "weight": eta},
            ],
            mixing=mixing_meta,
            seed=self.config.seed,
            train=avg_train_meta,
        )

        metrics = {
            "eta": eta,
            "br_states": len(br_policy.values()),
            "avg_components": len(new_avg_policy.policies) if isinstance(new_avg_policy, CommitOnceMixture) else 1,
        }

        self.run.log_event(
            "fsp.iteration",
            iter=iter_index,
            br_id=br_id,
            avg_id=new_avg_id,
            eta=eta,
        )

        return br_policy, new_avg_policy, br_id, new_avg_id, metrics

    def _eta(self, iter_index: int) -> float:
        schedule = self.config.eta_schedule
        if schedule == "harmonic":
            return harmonic_eta(iter_index + 1, c=self.config.eta_c)
        if schedule == "constant":
            return constant_eta(self.config.eta_constant)
        raise ValueError(f"Unknown eta schedule: {schedule}")

    def _mix_policies(self, base_policy: Policy, br_policy: Policy, eta: float) -> CommitOnceMixture:
        if self.config.mix_impl != "commit_once":
            raise ValueError(f"Unsupported mix impl: {self.config.mix_impl}")

        base_components = self._flatten_commit_once(base_policy)
        br_components = self._flatten_commit_once(br_policy)

        combined_policies: List[Policy] = []
        combined_weights: List[float] = []

        for policy, weight in base_components:
            scaled = (1.0 - eta) * weight
            if scaled > 0:
                combined_policies.append(policy)
                combined_weights.append(scaled)

        for policy, weight in br_components:
            scaled = eta * weight
            if scaled > 0:
                combined_policies.append(policy)
                combined_weights.append(scaled)

        return CommitOnceMixture(combined_policies, combined_weights, rng=self._rng)

    @staticmethod
    def _flatten_commit_once(policy: Policy) -> List[Tuple[Policy, float]]:
        if isinstance(policy, CommitOnceMixture):
            return list(zip(policy.policies, policy.weights))
        return [(policy, 1.0)]

