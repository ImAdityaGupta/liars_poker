from liars_poker.core import GameSpec, card_display, card_rank, generate_deck
from liars_poker.env import Env


def test_generate_deck_with_symmetry():
    spec = GameSpec(ranks=4, suits=3, hand_size=2, suit_symmetry=True)
    deck = generate_deck(spec)
    assert len(deck) == spec.ranks * spec.suits
    for rank in range(1, spec.ranks + 1):
        assert deck.count(rank) == spec.suits


def test_env_deals_rank_multisets_under_symmetry():
    spec = GameSpec(ranks=3, suits=2, hand_size=2, suit_symmetry=True)
    env = Env(spec, seed=123)
    obs = env.reset()
    hand = obs["hand"]
    assert len(hand) == spec.hand_size
    assert all(1 <= card <= spec.ranks for card in hand)
    # simulate a few resets ensure counts update
    seen_pairs = set()
    for seed in range(10):
        obs = env.reset(seed=seed)
        seen_pairs.add(tuple(obs["hand"]))
    assert seen_pairs  # non-empty


def test_card_display_and_rank_symmetry():
    spec = GameSpec(ranks=5, suits=2, hand_size=1, suit_symmetry=True)
    deck = generate_deck(spec)
    for card in deck:
        assert card_rank(card, spec) == card
        assert card_display(card, spec) == str(card)


def test_card_helpers_without_symmetry():
    spec = GameSpec(ranks=3, suits=2, hand_size=1, suit_symmetry=False)
    deck = generate_deck(spec)
    assert len(deck) == 6
    assert len({card_display(card, spec) for card in deck}) == 6
