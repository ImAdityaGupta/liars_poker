from liars_poker.core import GameSpec
from liars_poker.env import Env


def test_seed_reproducibility_same_seed_same_deal():
    spec = GameSpec(ranks=6, suits=2, hand_size=2, starter="P1", claim_kinds=("RankHigh", "Pair"))
    env = Env(spec, seed=99)
    obs1 = env.reset()
    hand1 = obs1["hand"] if env.current_player() == "P1" else env._p1_hand  # type: ignore[attr-defined]
    obs2 = env.reset(seed=99)
    hand2 = obs2["hand"] if env.current_player() == "P1" else env._p1_hand  # type: ignore[attr-defined]
    assert hand1 == hand2


def test_seed_reproducibility_different_seeds_often_different():
    spec = GameSpec(ranks=6, suits=2, hand_size=2, starter="P1", claim_kinds=("RankHigh", "Pair"))
    env = Env(spec, seed=0)
    deals = set()
    for s in [1, 2, 3, 4, 5]:
        env.reset(seed=s)
        deals.add((env._p1_hand, env._p2_hand))  # type: ignore[attr-defined]
    # Probabilistic: expect more than one unique deal
    assert len(deals) > 1

