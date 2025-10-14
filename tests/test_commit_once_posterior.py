import random

from liars_poker.core import GameSpec
from liars_poker.env import rules_for_spec
from liars_poker.infoset import InfoSet, NO_CLAIM
from liars_poker.policies.base import Policy
from liars_poker.policies.commit_once import CommitOnceMixture


class OneHotPolicy(Policy):
    def __init__(self, action: int) -> None:
        super().__init__()
        self._action = action

    def action_probs(self, infoset: InfoSet):
        return {self._action: 1.0}

    def prob_dist_at_infoset(self, infoset: InfoSet):
        return {self._action: 1.0}


def test_commit_once_posterior_updates_with_history() -> None:
    spec = GameSpec(ranks=3, suits=1, hand_size=1)
    rules = rules_for_spec(spec)

    base_infoset = InfoSet(pid=0, hand=(0,), history=())
    observed_infoset = InfoSet(pid=0, hand=(0,), history=(0,))

    policy_a = OneHotPolicy(0)
    policy_b = OneHotPolicy(1)

    policy_a.bind_rules(rules)
    policy_b.bind_rules(rules)

    mix = CommitOnceMixture([policy_a, policy_b], [0.5, 0.5], rng=random.Random(0))
    mix.bind_rules(rules)

    prior = mix.prob_dist_at_infoset(base_infoset)
    assert prior[0] == prior[1] == 0.5

    posterior = mix.prob_dist_at_infoset(observed_infoset)
    assert abs(posterior[0] - 1.0) < 1e-9
    assert posterior.get(1, 0.0) < 1e-9
