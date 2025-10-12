import random

from liars_poker.core import GameSpec
from liars_poker.env import rules_for_spec
from liars_poker.policy import CommitOnceMixture, PerDecisionMixture, Policy


class OneHotPolicy(Policy):
    def __init__(self, pick):
        super().__init__()
        self.pick = pick

    def action_probs(self, infoset_key):
        legal = self._legal(infoset_key)
        if self.pick not in legal:
            # default to first legal
            return {legal[0]: 1.0} if legal else {}
        return {a: (1.0 if a == self.pick else 0.0) for a in legal}


def test_per_decision_mixture_weights_and_shape():
    spec = GameSpec(ranks=4, suits=1, hand_size=1, starter="random", claim_kinds=("RankHigh",))
    rules = rules_for_spec(spec)
    infoset = (0, -2, tuple(), tuple())
    pi = OneHotPolicy(1)
    be = OneHotPolicy(2)
    mix = PerDecisionMixture(pi, be, w=0.25)
    mix.bind_rules(rules)
    la = rules.legal_actions_for(infoset)
    p = mix.action_probs(infoset)
    assert set(p.keys()) == set(la)
    # mass split between picks only
    assert p[0] == 0.0
    assert p[3] == 0.0
    assert abs(p[1] - 0.75) < 1e-9
    assert abs(p[2] - 0.25) < 1e-9


def test_commit_once_flips_once_per_episode():
    # Force deterministic flip via RNG
    rng = random.Random(0)
    spec = GameSpec(ranks=3, suits=1, hand_size=1, starter="random", claim_kinds=("RankHigh",))
    rules = rules_for_spec(spec)
    infoset = (0, -2, tuple(), tuple())
    pi = OneHotPolicy(0)
    be = OneHotPolicy(1)
    mix = CommitOnceMixture(pi, be, w=0.5, rng=rng)
    mix.bind_rules(rules)
    mix.begin_episode(rng)
    la = rules.legal_actions_for(infoset)
    p1 = mix.action_probs(infoset)
    p2 = mix.action_probs(infoset)
    # Within one episode, distribution stays on same component
    assert p1 == p2
    assert sum(p1.values()) == 1.0
    # New episode may change
    mix.begin_episode(rng)
    p3 = mix.action_probs(infoset)
    # Not necessarily different, but is a valid dist
    assert sum(p3.values()) == 1.0
