from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import os
from typing import Tuple


@dataclass(slots=True, frozen=True)
class GameSpec:
    """Immutable game specification (config).

    - ranks: number of ranks (1..r)
    - suits: number of suits per rank (1..4)
    - hand_size: cards per player
    - claim_kinds: tuple of claim family names in priority order
    """

    ranks: int
    suits: int
    hand_size: int
    claim_kinds: Tuple[str, ...] = ("RankHigh", "Pair")

    def to_json(self) -> str:
        return json.dumps(
            {
                "ranks": self.ranks,
                "suits": self.suits,
                "hand_size": self.hand_size,
                "claim_kinds": list(self.claim_kinds),
            },
            sort_keys=True,
        )


def env_hash(spec: GameSpec) -> str:
    """Deterministic short fingerprint for the environment spec."""
    h = hashlib.sha1(spec.to_json().encode("utf-8")).hexdigest()
    return h[:12]


# --- Repo helpers ---

@lru_cache()
def repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


ARTIFACTS_ROOT = os.path.join(repo_root(), "artifacts")


# --- Card helpers (ints) ---

def encode_card(rank: int, suit: int, suits: int) -> int:
    """Encode card to int id. rank in [1..R], suit in [0..S-1]."""
    return (rank - 1) * suits + suit


def decode_card(card_id: int, suits: int) -> Tuple[int, int]:
    """Decode int id to (rank, suit)."""
    rank = card_id // suits + 1
    suit = card_id % suits
    return rank, suit


def card_rank(card_id: int, suits: int) -> int:
    return card_id // suits + 1


def build_deck(ranks: int, suits: int) -> Tuple[int, ...]:
    return tuple(encode_card(r, s, suits) for r in range(1, ranks + 1) for s in range(suits))
