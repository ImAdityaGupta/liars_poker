import random

from liars_poker.policy import Policy, PerDecisionMixture, CommitOnceMixture


class OneHotPolicy(Policy):
    def __init__(self, pick):
        self.pick = pick

    def action_probs(self, infoset_key, legal_actions):
        if self.pick not in legal_actions:
            # default to first legal
            return {legal_actions[0]: 1.0} if legal_actions else {}
        return {a: (1.0 if a == self.pick else 0.0) for a in legal_actions}


def test_per_decision_mixture_weights_and_shape():
    pi = OneHotPolicy(1)
    be = OneHotPolicy(2)
    mix = PerDecisionMixture(pi, be, w=0.25)
    la = [1, 2, 3]
    p = mix.action_probs(("dummy",), la)
    assert set(p.keys()) == set(la)
    # mass split between 1 and 2 only
    assert p[3] == 0.0
    assert abs(p[1] - 0.75) < 1e-9
    assert abs(p[2] - 0.25) < 1e-9


def test_commit_once_flips_once_per_episode():
    # Force deterministic flip via RNG
    rng = random.Random(0)
    pi = OneHotPolicy(5)
    be = OneHotPolicy(7)
    mix = CommitOnceMixture(pi, be, w=0.5, rng=rng)
    mix.begin_episode(rng)
    la = [5, 7]
    p1 = mix.action_probs(("x",), la)
    p2 = mix.action_probs(("x",), la)
    # Within one episode, distribution stays on same component
    assert p1 == p2
    assert sum(p1.values()) == 1.0
    # New episode may change
    mix.begin_episode(rng)
    p3 = mix.action_probs(("x",), la)
    # Not necessarily different, but is a valid dist
    assert sum(p3.values()) == 1.0
