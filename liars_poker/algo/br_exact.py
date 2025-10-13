from __future__ import annotations

from liars_poker.core import GameSpec
from liars_poker.policies.tabular import TabularPolicy


def best_response_exact(spec: GameSpec) -> TabularPolicy:
    """Placeholder for an exact best response implementation."""

    _ = spec
    raise NotImplementedError("Exact best response not implemented.")

