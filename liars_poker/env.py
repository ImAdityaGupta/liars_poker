from __future__ import annotations

import random
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

from .core import GameSpec, generate_deck, card_rank
from .infoset import CALL, NO_CLAIM, InfoSet


class Rules:
    """Immutable action ordering and legality."""

    __slots__ = ("spec", "_claims", "_next_higher_indices")

    def __init__(self, spec: GameSpec):
        self.spec = spec
        self._claims: Tuple[Tuple[str, int], ...] = self._build_claims()
        self._next_higher_indices: Tuple[Tuple[int, ...], ...] = tuple(
            tuple(range(i + 1, len(self._claims))) for i in range(len(self._claims))
        )

    @property
    def claims(self) -> Tuple[Tuple[str, int], ...]:
        return self._claims

    def legal_actions_from_last(self, last_claim_idx: Optional[int]) -> Tuple[int, ...]:
        if last_claim_idx is None:
            candidates = range(len(self._claims))
        else:
            candidates = self._next_higher_indices[last_claim_idx]

        raises = tuple(idx for idx in candidates if self._claim_possible(idx))
        if last_claim_idx is None:
            return raises
        return (CALL,) + raises

    def legal_actions_for(self, infoset: InfoSet) -> Tuple[int, ...]:
        last_idx = InfoSet.last_claim_idx(infoset.history)
        if last_idx == NO_CLAIM:
            ref = None
        else:
            ref = last_idx
        return self.legal_actions_from_last(ref)

    def parse_action(self, text: str, legal: Sequence[int] | None = None) -> int:
        text = text.strip()
        if text.upper() == "CALL":
            action = CALL
        elif ":" in text:
            kind_str, value_str = text.split(":", 1)
            kind_norm = kind_str.strip().lower()
            try:
                rank_value = int(value_str.strip())
            except ValueError as exc:
                raise ValueError(f"Invalid rank value in action '{text}'") from exc
            action = None
            for idx, (kind, rank) in enumerate(self._claims):
                if kind.lower() == kind_norm and rank == rank_value:
                    action = idx
                    break
            if action is None:
                raise ValueError(f"Unknown claim: {text}")
        else:
            raise ValueError(f"Unrecognized action string: {text}")

        if legal is not None and action not in legal:
            raise ValueError(f"Action {text} not in legal set {legal}")
        return action

    def render_action(self, idx: int) -> str:
        if idx == CALL:
            return "CALL"
        kind, rank = self._claims[idx]
        return f"{kind}:{rank}"

    def _build_claims(self) -> Tuple[Tuple[str, int], ...]:
        claims: List[Tuple[str, int]] = []
        R = self.spec.ranks
        for kind in self.spec.claim_kinds:
            if kind == "RankHigh":
                for r in range(1, R + 1):
                    claims.append(("RankHigh", r))
            elif kind == "Pair":
                for r in range(1, R + 1):
                    claims.append(("Pair", r))
            else:
                raise ValueError(f"Unsupported claim kind: {kind}")
        return tuple(claims)

    def _claim_possible(self, idx: int) -> bool:
        kind, _ = self._claims[idx]
        S = self.spec.suits
        if kind == "RankHigh":
            return S >= 1
        if kind == "Pair":
            return S >= 2 and (self.spec.hand_size * 2) >= 2
        return False


@lru_cache(maxsize=128)
def rules_for_spec(spec: GameSpec) -> Rules:
    return Rules(spec)


class Env:
    """Minimal Liar's Poker environment with deterministic starting seat (P1)."""

    def __init__(self, spec: GameSpec, seed: Optional[int] = None):
        self.spec = spec
        self.rules = rules_for_spec(spec)
        self._rng = random.Random(seed)
        self._seed = seed

        self._p1_hand: Tuple[int, ...] = tuple()
        self._p2_hand: Tuple[int, ...] = tuple()
        self._to_play: int = 0
        self._last_claim_idx: Optional[int] = None
        self._history: List[int] = []
        self._done: bool = False
        self._winner: Optional[int] = None

    def reset(
        self,
        seed: Optional[int] = None,
        hands: Optional[Tuple[Tuple[int, ...], Tuple[int, ...]]] = None,
    ) -> Dict:
        if seed is not None:
            self._rng = random.Random(seed)
            self._seed = seed

        if hands is None:
            deck = list(generate_deck(self.spec))
            self._rng.shuffle(deck)
            k = self.spec.hand_size
            self._p1_hand = tuple(sorted(deck[:k]))
            self._p2_hand = tuple(sorted(deck[k : 2 * k]))
        else:
            if len(hands) != 2:
                raise ValueError("hands must be a tuple of (p1_hand, p2_hand)")
            self._p1_hand = tuple(sorted(hands[0]))
            self._p2_hand = tuple(sorted(hands[1]))

        self._to_play = 0  # P1 always starts
        self._last_claim_idx = None
        self._history = []
        self._done = False
        self._winner = None

        return self.observation_for(self.current_player())

    def current_player(self) -> str:
        return "P1" if self._to_play == 0 else "P2"

    def legal_actions(self) -> List[int]:
        if self._done:
            return []
        return list(self.rules.legal_actions_from_last(self._last_claim_idx))

    def step(self, action: int) -> Dict:
        if self._done:
            raise RuntimeError("Episode already done")
        legal = self.legal_actions()
        if action not in legal:
            raise ValueError(f"Illegal action {action}; legal={legal}")

        if action == CALL:
            self._resolve_call()
            self._history.append(CALL)
            self._done = True
            return self.observation_for(self.current_player())

        self._last_claim_idx = action
        self._history.append(action)
        self._to_play = 1 - self._to_play
        return self.observation_for(self.current_player())

    def infoset_key(self, for_player: str) -> InfoSet:
        pid = 0 if for_player == "P1" else 1
        hand = self._p1_hand if pid == 0 else self._p2_hand
        return InfoSet(pid=pid, hand=hand, history=tuple(self._history))

    def observation_for(self, player: str) -> Dict:
        pid = 0 if player == "P1" else 1
        hand = self._p1_hand if pid == 0 else self._p2_hand
        legal = self.legal_actions() if player == self.current_player() else []
        return {
            "to_play": self.current_player(),
            "hand": hand,
            "last_claim_idx": self._last_claim_idx,
            "legal_actions": legal,
            "history": tuple(self._history),
            "terminal": self._done,
            "winner": None if self._winner is None else ("P1" if self._winner == 0 else "P2"),
            "infoset_key": self.infoset_key(player),
        }

    def parse_action(self, text: str, legal: Sequence[int] | None = None) -> int:
        return self.rules.parse_action(text, legal)

    def render_action(self, idx: int) -> str:
        return self.rules.render_action(idx)

    def _resolve_call(self) -> None:
        assert self._last_claim_idx is not None, "CALL cannot occur on first move"
        joint_counts = self._joint_rank_counts()
        last_true = self._satisfied(self._last_claim_idx, joint_counts)
        caller = self._to_play
        if last_true:
            self._winner = 1 - caller
        else:
            self._winner = caller

    def _joint_rank_counts(self) -> List[int]:
        R = self.spec.ranks
        counts = [0] * (R + 1)
        for card in self._p1_hand:
            counts[card_rank(card, self.spec)] += 1
        for card in self._p2_hand:
            counts[card_rank(card, self.spec)] += 1
        return counts

    def _satisfied(self, idx: int, joint_counts: List[int]) -> bool:
        kind, r = self.rules.claims[idx]
        if kind == "RankHigh":
            return joint_counts[r] >= 1
        if kind == "Pair":
            return joint_counts[r] >= 2
        return False
