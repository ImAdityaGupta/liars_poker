from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import os
from itertools import combinations
from typing import List, Tuple


@dataclass(slots=True, frozen=True)
class GameSpec:
    """Immutable game specification (config).

    - ranks: number of ranks (1..r)
    - suits: number of suits per rank (1..4)
    - hand_size: cards per player
    - claim_kinds: tuple of claim family names in priority order
    - suit_symmetry: if True, suits are interchangeable; hands are rank multisets
    """

    ranks: int
    suits: int
    hand_size: int
    claim_kinds: Tuple[str, ...] = ("RankHigh", "Pair")
    suit_symmetry: bool = False

    def to_json(self) -> str:
        return json.dumps(
            {
                "ranks": self.ranks,
                "suits": self.suits,
                "hand_size": self.hand_size,
                "claim_kinds": list(self.claim_kinds),
                "suit_symmetry": self.suit_symmetry,
            },
            sort_keys=True,
        )

    def to_short_str(self) -> str:
        kinds = ""
        kinds += "h" if "RankHigh" in self.claim_kinds else ""
        kinds += "p" if "Pair" in self.claim_kinds else ""
        kinds += "2p" if "TwoPair" in self.claim_kinds else ""
        kinds += "t" if "Trips" in self.claim_kinds else ""
        kinds += "fh" if "FullHouse" in self.claim_kinds else ""
        kinds += "q" if "Quads" in self.claim_kinds else ""
        sym = "ss" if self.suit_symmetry else "nss"
        return f"r{self.ranks}_s{self.suits}_h{self.hand_size}_{kinds}_{sym}"


def env_hash(spec: GameSpec) -> str:
    """Deterministic short fingerprint for the environment spec."""
    h = hashlib.sha1(spec.to_json().encode("utf-8")).hexdigest()
    return h[:12]


# --- Repo helpers ---

@lru_cache()
def repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


ARTIFACTS_ROOT = os.path.join(repo_root(), "artifacts")


# --- Card helpers ---

def generate_deck(spec: GameSpec) -> Tuple[int, ...]:
    """Return the canonical deck for the spec.

    - suit_symmetry False: cards are encoded rank/suit pairs as ints.
    - suit_symmetry True and suits > 1: deck is a multiset of ranks, one entry per suit copy.
    """

    if spec.suit_symmetry and spec.suits > 1:
        deck = []
        for rank in range(1, spec.ranks + 1):
            deck.extend([rank] * spec.suits)
        return tuple(deck)

    deck = []
    for rank in range(1, spec.ranks + 1):
        for suit in range(spec.suits):
            card_id = (rank - 1) * spec.suits + suit
            deck.append(card_id)
    return tuple(deck)


def card_rank(card: int, spec: GameSpec) -> int:
    if spec.suit_symmetry and spec.suits > 1:
        return card
    return card // spec.suits + 1


def card_display(card: int, spec: GameSpec) -> str:
    if spec.suit_symmetry and spec.suits > 1:
        return str(card)
    rank = card_rank(card, spec)
    suit = card % spec.suits
    if spec.suits <= 1:
        return str(rank)
    suffix = chr(ord("A") + suit)
    return f"{rank}{suffix}"

def hand_display(hand: list[int] | tuple[int, ...], spec: GameSpec) -> list[str]:
    return [card_display(c, spec) for c in hand]

def possible_starting_hands(spec: GameSpec) -> List[Tuple[int, ...]]:
    """Enumerate all unique starting hands consistent with the spec."""

    deck = generate_deck(spec)
    hands = {
        tuple(sorted(hand))
        for hand in combinations(deck, spec.hand_size)
    }
    return sorted(hands)
