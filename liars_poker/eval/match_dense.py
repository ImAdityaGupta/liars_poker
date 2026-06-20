from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from liars_poker.algo.br_exact_dense_to_dense import BestResponseComputerDense
from liars_poker.core import GameSpec
from liars_poker.infoset import CALL
from liars_poker.policies.tabular_dense import DenseTabularPolicy


class _FixedResponseComputerDense(BestResponseComputerDense):
    """Evaluate a fixed dense responder using the exact BR traversal machinery."""

    def __init__(
        self,
        spec: GameSpec,
        opponent: DenseTabularPolicy,
        responder: DenseTabularPolicy,
    ) -> None:
        if responder.spec != spec:
            raise ValueError("DenseTabularPolicy spec mismatch.")
        if responder.S.shape != opponent.S.shape:
            raise ValueError("Dense policy tensor shape mismatch.")
        super().__init__(spec, opponent, store_state_values=False)
        self.responder = responder
        self.br_policy = None

    def _solve_node(
        self,
        hid: int,
        opp_reach: np.ndarray,
        *,
        to_play: int,
        my_id: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if to_play != my_id:
            return super()._solve_node(
                hid,
                opp_reach,
                to_play=to_play,
                my_id=my_id,
            )

        actions = self.opponent.legal_actions[hid]
        if not actions:
            zeros = np.zeros(self.n_hands, dtype=float)
            return zeros, zeros

        total_wm = np.zeros(self.n_hands, dtype=float)
        total_m = np.zeros(self.n_hands, dtype=float)
        next_to_play = 1 - to_play
        strategy = self.responder.S[hid]

        for action in actions:
            col = 0 if action == CALL else action + 1
            if action == CALL:
                wm, mass = self._solve_terminal(
                    hid,
                    opp_reach,
                    caller=to_play,
                    my_id=my_id,
                )
            else:
                wm, mass = self._solve_node(
                    hid | (1 << action),
                    opp_reach,
                    to_play=next_to_play,
                    my_id=my_id,
                )

            probabilities = strategy[:, col].astype(float, copy=False)
            total_wm += probabilities * wm
            total_m += probabilities * mass

        return total_wm, total_m

    def solve(self) -> None:
        root_reach = np.ones(self.n_hands, dtype=float)

        wm0, m0 = self._solve_node(0, root_reach, to_play=0, my_id=0)
        self._root_values[0] = np.where(m0 > 0.0, wm0 / m0, 0.0)

        wm1, m1 = self._solve_node(0, root_reach, to_play=0, my_id=1)
        self._root_values[1] = np.where(m1 > 0.0, wm1 / m1, 0.0)


def evaluate_dense_response(
    spec: GameSpec,
    opponent: DenseTabularPolicy,
    responder: DenseTabularPolicy,
) -> Tuple[float, float]:
    """Return responder win rates when it plays first and second."""

    computer = _FixedResponseComputerDense(spec, opponent, responder)
    computer.solve()
    return computer.exploitability()


def evaluate_dense_match(
    spec: GameSpec,
    p1: DenseTabularPolicy,
    p2: DenseTabularPolicy,
) -> Dict[str, float]:
    """Return exact fixed-seat win probabilities for dense P1 versus dense P2."""

    p1_win, _ = evaluate_dense_response(spec, opponent=p2, responder=p1)
    return {"P1": p1_win, "P2": 1.0 - p1_win}
