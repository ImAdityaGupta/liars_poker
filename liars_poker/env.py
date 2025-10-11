from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence, Tuple

from .core import GameSpec, build_deck, card_rank


# Special action id for CALL
CALL = -1


class Env:
    """Minimal Liar's Poker environment.

    Hot path uses ints/tuples; precomputes claim order and next-higher map.
    """

    def __init__(self, spec: GameSpec, seed: Optional[int] = None):
        self.spec = spec
        self._rng = random.Random(seed)
        self._seed = seed

        # Precompute claims for this spec
        self.claims: List[Tuple[str, int]] = self._build_claims()
        self.next_higher_indices: List[List[int]] = [
            list(range(i + 1, len(self.claims))) for i in range(len(self.claims))
        ]

        # Mutable episode state
        self._p1_hand: Tuple[int, ...] = tuple()
        self._p2_hand: Tuple[int, ...] = tuple()
        self._to_play: int = 0  # 0 -> P1, 1 -> P2
        self._last_claim_idx: Optional[int] = None
        self._history: List[int] = []
        self._done: bool = False
        self._winner: Optional[int] = None

    # --- Core API ---
    def reset(
        self,
        seed: Optional[int] = None,
        hands: Optional[Tuple[Tuple[int, ...], Tuple[int, ...]]] = None,
        starter: Optional[str] = None,
    ) -> Dict:
        if seed is not None:
            self._rng = random.Random(seed)
            self._seed = seed

        if hands is None:
            deck = list(build_deck(self.spec.ranks, self.spec.suits))
            self._rng.shuffle(deck)
            k = self.spec.hand_size
            self._p1_hand = tuple(sorted(deck[:k]))
            self._p2_hand = tuple(sorted(deck[k : 2 * k]))
        else:
            if len(hands) != 2:
                raise ValueError("hands must be a tuple of (p1_hand, p2_hand)")
            self._p1_hand = tuple(sorted(hands[0]))
            self._p2_hand = tuple(sorted(hands[1]))

        starter_choice = starter if starter is not None else self.spec.starter
        if starter_choice == "random":
            self._to_play = self._rng.choice([0, 1])
        elif starter_choice == "P1":
            self._to_play = 0
        elif starter_choice == "P2":
            self._to_play = 1
        else:
            raise ValueError(f"Invalid starter: {starter_choice}")

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

        actions: List[int] = []
        # Raises (strictly higher than last claim, or any claim if none)
        if self._last_claim_idx is None:
            candidate_indices = range(len(self.claims))
        else:
            candidate_indices = self.next_higher_indices[self._last_claim_idx]

        for idx in candidate_indices:
            if self._claim_possible(idx):
                actions.append(idx)

        # CALL allowed after first move
        if self._last_claim_idx is not None:
            actions.insert(0, CALL)

        return actions

    def step(self, action: int) -> Dict:
        if self._done:
            raise RuntimeError("Episode already done")
        la = self.legal_actions()
        if action not in la:
            raise ValueError(f"Illegal action {action}; legal={la}")

        if action == CALL:
            # Resolve winner
            assert self._last_claim_idx is not None, "CALL cannot occur on first move"
            joint_counts = self._joint_rank_counts()
            last_true = self._satisfied(self._last_claim_idx, joint_counts)
            caller = self._to_play
            if last_true:
                # Caller loses
                self._winner = 1 - caller
            else:
                self._winner = caller
            self._history.append(CALL)
            self._done = True
            return self.observation_for(self.current_player())

        # Raise with claim index
        self._last_claim_idx = action
        self._history.append(action)
        self._to_play = 1 - self._to_play
        return self.observation_for(self.current_player())

    # --- Infosets ---
    def infoset_key(self, for_player: str) -> Tuple:
        pid = 0 if for_player == "P1" else 1
        hand = self._p1_hand if pid == 0 else self._p2_hand
        last_idx = -2 if self._last_claim_idx is None else self._last_claim_idx
        return (
            pid,
            last_idx,
            hand,
            tuple(self._history),
        )

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
        text = text.strip()
        if text.upper() == "CALL":
            action = CALL
        elif ":" in text:
            kind_str, value_str = text.split(":", 1)
            kind_norm = kind_str.strip().lower()
            rank_value = int(value_str.strip())
            action = None
            for idx, (kind, rank) in enumerate(self.claims):
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
        kind, rank = self.claims[idx]
        return f"{kind}:{rank}"

    # --- Internals ---
    def _build_claims(self) -> List[Tuple[str, int]]:
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
        return claims

    def _claim_possible(self, idx: int) -> bool:
        kind, r = self.claims[idx]
        S = self.spec.suits
        if kind == "RankHigh":
            return S >= 1
        if kind == "Pair":
            return S >= 2 and (self.spec.hand_size * 2) >= 2
        # TODO: tighten feasibility checks based on remaining unseen cards.
        return False

    def _joint_rank_counts(self) -> List[int]:
        R, S = self.spec.ranks, self.spec.suits
        counts = [0] * (R + 1)  # 1-based ranks
        for c in self._p1_hand:
            counts[card_rank(c, S)] += 1
        for c in self._p2_hand:
            counts[card_rank(c, S)] += 1
        return counts

    def _satisfied(self, idx: int, joint_counts: List[int]) -> bool:
        kind, r = self.claims[idx]
        if kind == "RankHigh":
            return joint_counts[r] >= 1
        if kind == "Pair":
            return joint_counts[r] >= 2
        return False
