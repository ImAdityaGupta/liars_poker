from liars_poker.algo.br_mc import best_response_mc
from liars_poker.core import GameSpec
from liars_poker.policies.random import RandomPolicy


def test_best_response_annotations_present() -> None:
    spec = GameSpec(ranks=3, suits=1, hand_size=1)
    opponent = RandomPolicy()
    br_policy = best_response_mc(
        spec,
        opponent,
        episodes=200,
        epsilon=0.05,
        min_visits_per_action=1,
        seed=7,
    )

    values = br_policy.values()
    assert values, "Expected state values annotations."

    for state, value in values.items():
        assert br_policy.get_value(state) == value

    visits = br_policy.visits()
    assert visits, "Expected visit annotations."
    for state, count in visits.items():
        assert count >= 0
        assert br_policy.get_visits(state) == count
