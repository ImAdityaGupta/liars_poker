from liars_poker.core import GameSpec
from liars_poker.env import Env


def test_infoset_identity_same_private_and_public():
    spec = GameSpec(ranks=5, suits=2, hand_size=2, starter="P1", claim_kinds=("RankHigh", "Pair"))
    env1 = Env(spec, seed=123)
    env2 = Env(spec, seed=123)

    obs1 = env1.reset()
    obs2 = env2.reset()

    # Same starter and same deals -> same infoset keys at start
    k1 = env1.infoset_key(env1.current_player())
    k2 = env2.infoset_key(env2.current_player())
    assert k1 == k2

    # After identical actions, still same
    a1 = obs1["legal_actions"][0]
    obs1 = env1.step(a1)
    obs2 = env2.step(a1)
    k1b = env1.infoset_key(env1.current_player())
    k2b = env2.infoset_key(env2.current_player())
    assert k1b == k2b


def test_infoset_difference_when_history_differs():
    spec = GameSpec(ranks=5, suits=2, hand_size=2, starter="P1", claim_kinds=("RankHigh", "Pair"))
    env1 = Env(spec, seed=456)
    env2 = Env(spec, seed=456)
    env1.reset()
    env2.reset()
    # Take different first actions (first and second legal)
    a1 = env1.legal_actions()[0]
    a2 = env2.legal_actions()[1]
    env1.step(a1)
    env2.step(a2)
    k1 = env1.infoset_key(env1.current_player())
    k2 = env2.infoset_key(env2.current_player())
    assert k1 != k2

