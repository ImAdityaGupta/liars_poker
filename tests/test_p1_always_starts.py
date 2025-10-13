from liars_poker.core import GameSpec
from liars_poker.env import Env


def test_p1_always_starts() -> None:
    spec = GameSpec(ranks=4, suits=1, hand_size=1)
    env = Env(spec, seed=123)

    for offset in range(100):
        env.reset(seed=123 + offset)
        assert env.current_player() == "P1"
