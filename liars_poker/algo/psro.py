from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from liars_poker.core import GameSpec
from liars_poker.policies.base import Policy


def _require_scipy():
    try:
        from scipy.optimize import linprog
    except Exception as exc:  # pragma: no cover - import guard
        raise ImportError("SciPy is required for the PSRO Nash solver.") from exc
    return linprog


def nash_solver(U: np.ndarray, *, entropy: float = 0.0, **kwargs) -> Tuple[np.ndarray, np.ndarray, float]:
    """Solve a zero-sum matrix game using linear programming."""

    _ = entropy
    if U.ndim != 2:
        raise ValueError("Payoff matrix must be 2D.")
    m, n = U.shape
    if m == 0 or n == 0:
        raise ValueError("Payoff matrix must be non-empty.")

    linprog = _require_scipy()

    # Row player (first): maximize v
    c = np.zeros(m + 1, dtype=float)
    c[-1] = -1.0
    A_ub = np.hstack([-U.T, np.ones((n, 1), dtype=float)])
    b_ub = np.zeros(n, dtype=float)
    A_eq = np.zeros((1, m + 1), dtype=float)
    A_eq[0, :m] = 1.0
    b_eq = np.array([1.0], dtype=float)
    bounds = [(0.0, None)] * m + [(None, None)]
    res_row = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res_row.success:
        raise RuntimeError(f"Row LP failed: {res_row.message}")
    sigma_first = np.clip(res_row.x[:m], 0.0, None)
    if sigma_first.sum() <= 0:
        raise RuntimeError("Row strategy is degenerate.")
    sigma_first /= sigma_first.sum()

    # Column player (second): minimize v
    c2 = np.zeros(n + 1, dtype=float)
    c2[-1] = 1.0
    A_ub2 = np.hstack([U, -np.ones((m, 1), dtype=float)])
    b_ub2 = np.zeros(m, dtype=float)
    A_eq2 = np.zeros((1, n + 1), dtype=float)
    A_eq2[0, :n] = 1.0
    b_eq2 = np.array([1.0], dtype=float)
    bounds2 = [(0.0, None)] * n + [(None, None)]
    res_col = linprog(c2, A_ub=A_ub2, b_ub=b_ub2, A_eq=A_eq2, b_eq=b_eq2, bounds=bounds2, method="highs")
    if not res_col.success:
        raise RuntimeError(f"Column LP failed: {res_col.message}")
    sigma_second = np.clip(res_col.x[:n], 0.0, None)
    if sigma_second.sum() <= 0:
        raise RuntimeError("Column strategy is degenerate.")
    sigma_second /= sigma_second.sum()

    value = float(sigma_first @ U @ sigma_second)
    return sigma_first, sigma_second, value


@dataclass
class PSROState:
    spec: GameSpec
    pop_first: List[Policy]
    pop_second: List[Policy]
    ids_first: List[int]
    ids_second: List[int]
    next_id_first: int
    next_id_second: int
    n: np.ndarray
    mean: np.ndarray
    M2: np.ndarray

    @classmethod
    def create(cls, spec: GameSpec, pop_first: List[Policy], pop_second: List[Policy]) -> "PSROState":
        ids_first = list(range(len(pop_first)))
        ids_second = list(range(len(pop_second)))
        n = np.zeros((len(pop_first), len(pop_second)), dtype=np.int64)
        mean = np.zeros((len(pop_first), len(pop_second)), dtype=float)
        M2 = np.zeros((len(pop_first), len(pop_second)), dtype=float)
        return cls(
            spec=spec,
            pop_first=pop_first,
            pop_second=pop_second,
            ids_first=ids_first,
            ids_second=ids_second,
            next_id_first=len(ids_first),
            next_id_second=len(ids_second),
            n=n,
            mean=mean,
            M2=M2,
        )

    def ensure_shape(self) -> None:
        rows = len(self.pop_first)
        cols = len(self.pop_second)
        if self.n.shape == (rows, cols):
            return
        new_n = np.zeros((rows, cols), dtype=np.int64)
        new_mean = np.zeros((rows, cols), dtype=float)
        new_M2 = np.zeros((rows, cols), dtype=float)
        old_rows, old_cols = self.n.shape
        new_n[:old_rows, :old_cols] = self.n
        new_mean[:old_rows, :old_cols] = self.mean
        new_M2[:old_rows, :old_cols] = self.M2
        self.n, self.mean, self.M2 = new_n, new_mean, new_M2

    def add_first(self, policy: Policy) -> int:
        self.pop_first.append(policy)
        self.ids_first.append(self.next_id_first)
        self.next_id_first += 1
        self.ensure_shape()
        return len(self.pop_first) - 1

    def add_second(self, policy: Policy) -> int:
        self.pop_second.append(policy)
        self.ids_second.append(self.next_id_second)
        self.next_id_second += 1
        self.ensure_shape()
        return len(self.pop_second) - 1

    def get_first_id(self, idx: int) -> int:
        return self.ids_first[idx]

    def get_second_id(self, idx: int) -> int:
        return self.ids_second[idx]

    def update_entry(self, i: int, j: int, returns: Sequence[float]) -> None:
        for value in returns:
            n = int(self.n[i, j]) + 1
            delta = value - self.mean[i, j]
            mean = self.mean[i, j] + delta / n
            delta2 = value - mean
            M2 = self.M2[i, j] + delta * delta2
            self.n[i, j] = n
            self.mean[i, j] = mean
            self.M2[i, j] = M2

    def update_entry_counts(self, i: int, j: int, wins_first: int, wins_second: int) -> None:
        batch_n = int(wins_first + wins_second)
        if batch_n <= 0:
            return
        batch_mean = (wins_first - wins_second) / batch_n
        batch_M2 = wins_first * (1.0 - batch_mean) ** 2 + wins_second * (-1.0 - batch_mean) ** 2

        n_old = float(self.n[i, j])
        mean_old = float(self.mean[i, j])
        M2_old = float(self.M2[i, j])

        n_new = n_old + batch_n
        delta = batch_mean - mean_old
        mean_new = mean_old + delta * batch_n / n_new
        M2_new = M2_old + batch_M2 + delta * delta * n_old * batch_n / n_new

        self.n[i, j] = int(n_new)
        self.mean[i, j] = mean_new
        self.M2[i, j] = M2_new

    def U_mean(self) -> np.ndarray:
        return self.mean
