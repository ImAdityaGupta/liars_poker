from __future__ import annotations

import math
from typing import Dict

from liars_poker.algo.br_neural import NeuralBRTrainer


def wilson_lower_bound(wins: int, episodes: int, *, z: float = 1.96) -> float:
    """One-sided use of the Wilson score formula at the supplied z value."""

    if episodes <= 0:
        return 0.0
    p = wins / episodes
    z2 = z * z
    denominator = 1.0 + z2 / episodes
    centre = p + z2 / (2.0 * episodes)
    radius = z * math.sqrt(
        (p * (1.0 - p) + z2 / (4.0 * episodes)) / episodes
    )
    return (centre - radius) / denominator


def evaluate_approximate_br(
    trainer: NeuralBRTrainer,
    *,
    episodes_per_role: int = 200_000,
    rollout_batch_size: int = 8192,
    z: float = 1.96,
) -> Dict[str, float]:
    """Held-out response values and a conservative exploitability lower bound."""

    first = trainer.evaluate_role(
        0,
        episodes_per_role,
        rollout_batch_size=rollout_batch_size,
    )
    second = trainer.evaluate_role(
        1,
        episodes_per_role,
        rollout_batch_size=rollout_batch_size,
    )
    first_lcb = wilson_lower_bound(
        int(first["wins"]),
        int(first["episodes"]),
        z=z,
    )
    second_lcb = wilson_lower_bound(
        int(second["wins"]),
        int(second["episodes"]),
        z=z,
    )
    return {
        "p_first": float(first["win_rate"]),
        "p_second": float(second["win_rate"]),
        "exploitability_estimate": (
            float(first["win_rate"]) + float(second["win_rate"]) - 1.0
        ),
        "p_first_lcb": first_lcb,
        "p_second_lcb": second_lcb,
        "exploitability_lower_bound": first_lcb + second_lcb - 1.0,
        "episodes_per_role": int(episodes_per_role),
        "z": float(z),
    }
