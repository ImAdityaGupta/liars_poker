from __future__ import annotations

import random
from typing import Callable, Dict, Optional, Tuple

import scipy.stats as stats

from liars_poker.core import GameSpec
from liars_poker.env import Rules
from liars_poker import eval_seats_split
from liars_poker.policies import Policy, DenseTabularPolicy
from liars_poker.policies.tabular_dense import mix_dense
from liars_poker.algo.br_exact_dense_to_dense import best_response_dense
from liars_poker.training.fsp_utils import basic_eta_control


def dense_fsp_loop(
    spec: GameSpec,
    episodes: int,
    *,
    initial_pol: Optional[Policy | None] = None,
    eta_control: Optional[Callable[[int], float]] | None = None,
    episodes_test: Optional[int] = 10_000,
    efficient: bool = False,
    debug: bool = False,
) -> Tuple[Policy, Dict]:
    if initial_pol is None:
        initial_pol = DenseTabularPolicy(spec)

    if eta_control is None:
        eta_control = basic_eta_control

    if episodes_test is None:
        episodes_test = 100

    rules = Rules(spec)
    initial_pol.bind_rules(rules)

    curr_av = initial_pol

    logs: Dict[str, object] = {
        "exploitability_series": [],
        "p_values": [],
    }

    for i in range(episodes):
        if debug:
            print(i)

        eta = eta_control(i)

        b_i, meta = best_response_dense(
            spec=spec,
            policy=curr_av,
            debug=debug,
            store_state_values=not efficient,
        )
        br_computer = meta["computer"] if isinstance(meta, dict) else None
        p_first, p_second = br_computer.exploitability() if br_computer is not None else (None, None)
        predicted = 0.5 * (p_first + p_second) if p_first is not None else None
        if efficient:
            br_computer = None
            meta = None

        seat_results = eval_seats_split(
            spec, b_i, curr_av,
            episodes=episodes_test,
            seed=random.randint(1, 2_147_483_647),
        )
        obs_seat1 = seat_results["A_seat1"]
        obs_seat2 = seat_results["A_seat2"]
        observed_avg = 0.5 * (obs_seat1 + obs_seat2)

        chi2_stat = 0.0
        p_value = None
        if p_first is not None and p_second is not None:
            expected = [episodes_test * p_first * 0.5, episodes_test * p_second * 0.5]
            observed = [episodes_test * obs_seat1 * 0.5, episodes_test * obs_seat2 * 0.5]
            if all(e > 0 for e in expected):
                chi2_stat = sum(((o - e) ** 2) / e for o, e in zip(observed, expected))
                p_value = 1 - stats.chi2.cdf(chi2_stat, 1)

        logs["p_values"].append(p_value)

        print(f"Predicted exploitability: avg={predicted:.9f} (first={p_first:.4f}, second={p_second:.4f})")
        print(
            f"Sampled exploitability: avg={observed_avg:.4f} "
            f"(BR as P1={obs_seat1:.4f}, BR as P2={obs_seat2:.4f}), "
            f"chi2 p-value={(p_value if p_value is not None else float('nan')):.4g}"
        )
        print()

        logs["exploitability_series"].append(
            {
                "predicted_avg": predicted,
                "p_first": p_first,
                "p_second": p_second,
                "rollout_avg": observed_avg,
                "rollout_seat1": obs_seat1,
                "rollout_seat2": obs_seat2,
            }
        )

        curr_av = mix_dense(b_i, curr_av, eta)
        if efficient:
            b_i = None

    return curr_av, logs
