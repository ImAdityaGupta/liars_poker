from __future__ import annotations


def harmonic_eta(t: int, c: float = 1.0) -> float:
    if t < 0:
        raise ValueError("harmonic_eta expects non-negative iteration index.")
    return c / (t + 1.0)


def constant_eta(value: float) -> float:
    if not 0.0 <= value <= 1.0:
        raise ValueError("Constant eta must be within [0, 1].")
    return value

