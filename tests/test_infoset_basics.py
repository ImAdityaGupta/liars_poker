from liars_poker.core import GameSpec
from liars_poker.env import Env
from liars_poker.infoset import CALL, InfoSet, NO_CLAIM


def test_infoset_type_and_history() -> None:
    spec = GameSpec(ranks=3, suits=1, hand_size=1)
    env = Env(spec, seed=0)

    obs = env.reset()
    iset_p1 = env.infoset_key("P1")
    assert isinstance(iset_p1, InfoSet)
    assert iset_p1.pid == 0
    assert iset_p1.last_idx == NO_CLAIM
    assert iset_p1.history == ()

    first_action = env.legal_actions()[0]
    env.step(first_action)

    iset_p2 = env.infoset_key("P2")
    assert isinstance(iset_p2, InfoSet)
    assert len(iset_p2.history) == len(iset_p1.history) + 1

    # Force terminal by calling immediately
    if CALL in env.legal_actions():
        env.step(CALL)
        iset_terminal = env.infoset_key("P1")
        assert iset_terminal.history and iset_terminal.history[-1] == CALL


def test_infoset_hashability() -> None:
    spec = GameSpec(ranks=3, suits=1, hand_size=1)
    env = Env(spec, seed=1)

    env.reset()
    iset = env.infoset_key("P1")
    seen = {iset}
    assert iset in seen
