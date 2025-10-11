from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .core import GameSpec
from .policy import Policy, RandomPolicy

if TYPE_CHECKING:
    from .env import Env


def best_response_exact(spec: GameSpec, opponent: Policy, who: str) -> Policy:
    """Placeholder for exact BR on tiny games.

    Args:
        spec: Game specification for the environment.
        opponent: Opponent policy to respond to.
        who: Which player to compute the BR for ("P1" or "P2").

    Returns:
        Currently returns a RandomPolicy placeholder.
    """

    _ = (spec, opponent, who)
    return RandomPolicy()


class RLLearner:
    """RL learner base stub (no training yet)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        _ = (args, kwargs)

    def fit_off_policy(self, opponent: Policy, env: "Env", **params: Any) -> None:
        """Train against a fixed opponent in the given environment."""

        raise NotImplementedError

    def export_policy(self) -> Policy:
        """Return the learned policy."""

        raise NotImplementedError
