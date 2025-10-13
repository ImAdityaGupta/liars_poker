from __future__ import annotations

from typing import NamedTuple, Tuple

# Action ids
CALL = -1
NO_CLAIM = -2


class InfoSet(NamedTuple):
    """Hashable state supplied to policies when it is their turn."""

    pid: int  # 0 for the policy playing as P1, 1 for P2
    last_idx: int  # index of last claim, or NO_CLAIM when no claim yet
    hand: Tuple[int, ...]  # sorted private cards for this player
    history: Tuple[int, ...]  # public action history (claim indices / CALL)

