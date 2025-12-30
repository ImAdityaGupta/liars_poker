from __future__ import annotations

from liars_poker.core import GameSpec, possible_starting_hands
from liars_poker.policies.tabular import Policy, TabularPolicy
from liars_poker.policies import CommitOnceMixture
from liars_poker.infoset import InfoSet
from liars_poker.env import resolve_call_winner, Rules

from typing import Dict, Tuple, List
import random


def flatten_commit_once(policy: Policy) -> List[Tuple[Policy, float]]:
    if isinstance(policy, CommitOnceMixture):
        return list(zip(policy.policies, policy.weights))
    return [(policy, 1.0)]

def mix_policies(base_policy: Policy, br_policy: Policy, eta: float, rng: random.Random | None = None) -> CommitOnceMixture:
    base_components = flatten_commit_once(base_policy)
    br_components = flatten_commit_once(br_policy)

    combined_policies: List[Policy] = []
    combined_weights: List[float] = []

    for policy, weight in base_components:
        scaled = (1.0 - eta) * weight
        combined_policies.append(policy)
        combined_weights.append(scaled)

    for policy, weight in br_components:
        scaled = eta * weight
        combined_policies.append(policy)
        combined_weights.append(scaled)

    mixed_policy = CommitOnceMixture(combined_policies, combined_weights, rng=rng)
    mixed_policy.bind_rules(base_policy._rules)

    return mixed_policy


def fsp_loop(spec: GameSpec, initial_pol: Policy | None, ):
    pass




