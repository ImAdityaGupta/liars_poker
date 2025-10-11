from liars_poker.core import GameSpec
from liars_poker.env import Env, CALL


def test_first_move_forbids_call():
    spec = GameSpec(ranks=5, suits=2, hand_size=2, starter="P1", claim_kinds=("RankHigh", "Pair"))
    env = Env(spec, seed=42)
    obs = env.reset()
    assert CALL not in obs["legal_actions"]


def test_call_allowed_after_first_move_and_strict_raise():
    spec = GameSpec(ranks=5, suits=2, hand_size=1, starter="P1", claim_kinds=("RankHigh", "Pair"))
    env = Env(spec, seed=1)
    obs = env.reset()
    # First player makes the first claim (lowest claim index)
    first_claim = obs["legal_actions"][0]
    assert first_claim != CALL
    obs = env.step(first_claim)
    # Now CALL should be legal
    assert CALL in obs["legal_actions"]
    # Any raise must be strictly higher than first_claim
    raises = [a for a in obs["legal_actions"] if a != CALL]
    assert all(a > first_claim for a in raises)


def test_env_parse_render_roundtrip():
    spec = GameSpec(ranks=5, suits=2, hand_size=1, starter="P1", claim_kinds=("RankHigh", "Pair"))
    env = Env(spec, seed=5)
    obs = env.reset()
    first = obs["legal_actions"][0]
    text = env.render_action(first)
    assert env.parse_action(text) == first
    obs = env.step(first)
    assert env.parse_action("CALL", obs["legal_actions"]) == CALL
