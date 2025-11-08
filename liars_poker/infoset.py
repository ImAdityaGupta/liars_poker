from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

# Action ids
CALL = -1
NO_CLAIM = -2


@dataclass(frozen=True, slots=True)
class InfoSet:
    """Hashable state supplied to policies when it is their turn."""

    pid: int  # 0 for the policy playing as P1, 1 for P2
    hand: Tuple[int, ...]  # sorted private cards for this player
    history: Tuple[int, ...]  # public action history (claim indices / CALL)

    def __post_init__(self):
        if not isinstance(self.history, tuple):
            raise TypeError("InfoSet.history must be a tuple of actions.")
        if not isinstance(self.hand, tuple):
            raise TypeError("InfoSet.hand must be a tuple of cards.")

    @staticmethod
    def last_claim_idx(history: Tuple[int, ...]) -> int:
        for action in reversed(history):
            if action == CALL:
                continue
            return action
        return NO_CLAIM
