from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class FSPConfig:
    episodes0: int
    epsilon: float
    min_visits: int
    max_iters: int
    eta_schedule: str = "harmonic"
    eta_c: float = 1.0
    eta_constant: float = 0.1
    mix_impl: str = "commit_once"
    seed: int = 0

