from __future__ import annotations

from liars_poker.core import GameSpec, possible_starting_hands
from liars_poker.policies.tabular import Policy, TabularPolicy
from liars_poker.policies import CommitOnceMixture, RandomPolicy
from liars_poker.infoset import InfoSet
from liars_poker.env import resolve_call_winner, Rules
from liars_poker import eval_seats_split

from typing import Dict, Tuple, List, Callable, Optional, Iterable
import random

def flatten_commit_once(policy: Policy) -> List[Tuple[Policy, float]]:
    if isinstance(policy, CommitOnceMixture):
        return list(zip(policy.policies, policy.weights))
    return [(policy, 1.0)]

def mix_policies(base_policy: Policy, br_policy: Policy, eta: float, rng: random.Random | None = None) -> CommitOnceMixture:
    base_components = flatten_commit_once(base_policy)
    br_components = flatten_commit_once(br_policy)

    combined_policies: List[Policy] = []
    combined_weights: List[float] = []

    for policy, weight in base_components:
        scaled = (1.0 - eta) * weight
        combined_policies.append(policy)
        combined_weights.append(scaled)

    for policy, weight in br_components:
        scaled = eta * weight
        combined_policies.append(policy)
        combined_weights.append(scaled)

    mixed_policy = CommitOnceMixture(combined_policies, combined_weights, rng=rng)
    mixed_policy.bind_rules(base_policy._rules)

    return mixed_policy

def basic_eta_control(episodes: int) -> float:
    return 1 / (episodes + 2)

def fsp_loop(
    spec: GameSpec,
    br_fn: Callable[..., Tuple[Policy, Dict]],
    episodes: int,
    *,
    initial_pol: Policy = None,
    eta_control: Callable[[int], float] = None,
    episodes_test: int = 100,
    # Passed directly to br_fn each iteration (choose per run).
    br_kwargs: dict = None,
    # If br_meta does not contain an analytic prediction, do a larger rollout eval once.
    rollout_episodes_if_no_pred: int = 100,
    debug: bool = False
) -> Tuple[TabularPolicy, Dict]:

    if initial_pol is None:
        initial_pol = RandomPolicy()
    if eta_control is None:
        eta_control = basic_eta_control
    if br_kwargs is None:
        br_kwargs = {}

    rules = Rules(spec)
    initial_pol.bind_rules(rules)
    logs = {"exploitability_series": [], "p_values": [], "br_meta": []}

    curr_av = initial_pol

    for i in range(episodes):
        if debug:
            print(i)

        eta = eta_control(i)

        # br_fn must accept **br_kwargs for this run
        b_i, meta = br_fn(spec=spec, policy=curr_av, debug=debug, **br_kwargs)
        if meta is None:
            meta = {}

        logs["br_meta"].append(meta)

        if meta['computes_exploitability']:
            p_first, p_second = meta['computer'].exploitability()
            predicted = 0.5 * (p_first + p_second)
            seat_results = eval_seats_split(
                spec, b_i, curr_av,
                episodes=episodes_test,
                seed=random.randint(1, 2_147_483_647),
            )

            # Observed seat-wise win rates for BR (b_i) and avg (curr_av)
            obs_a_seat1 = seat_results["A_seat1"]
            obs_a_seat2 = seat_results["A_seat2"]
            obs_avg = 0.5 * (obs_a_seat1 + obs_a_seat2)

            # Chi-square against predicted p_first/p_second
            chi2_stat = 0.0
            p_value = None
            expected = [episodes_test * p_first * 0.5, episodes_test * p_second * 0.5]
            observed = [episodes_test * obs_a_seat1 * 0.5, episodes_test * obs_a_seat2 * 0.5]
            # Two bins: BR as P1, BR as P2
            exp_success = expected
            obs_success = observed
            exp_total = [episodes_test * 0.5, episodes_test * 0.5]
            obs_total = [episodes_test * 0.5, episodes_test * 0.5]
            if all(e > 0 for e in exp_success):
                chi2_stat = sum(((o - e) ** 2) / e for o, e in zip(obs_success, exp_success))
                p_value = 1 - stats.chi2.cdf(chi2_stat, 2 - 1)  # df=1 for two bins

            logs["p_values"].append(p_value)

            print(f"Predicted exploitability: avg={predicted:.4f} (first={p_first:.4f}, second={p_second:.4f})")
            print(f"Sampled   exploitability: avg={obs_avg:.4f} (first={obs_a_seat1:.4f}, second={obs_a_seat2:.4f}), chi2 p-value={(p_value if p_value is not None else float('nan')):.4g}\n")

            logs["exploitability_series"].append(
                {
                    "predicted_avg": predicted,
                    "p_first": p_first,
                    "p_second": p_second,
                    "rollout_avg": obs_avg,
                    "rollout_seat1": obs_a_seat1,
                    "rollout_seat2": obs_a_seat2,
                }
            )

        else:
            eval_eps = max(episodes_test, rollout_episodes_if_no_pred)
            seat_results = eval_seats_split(
                spec, b_i, curr_av,
                episodes=eval_eps,
                seed=random.randint(1, 2_147_483_647),
            )
            obs_avg = 0.5 * (seat_results["A_seat1"] + seat_results["A_seat2"])

            logs["p_values"].append(None)

            print(f"Exploitability (rollout only): avg={obs_avg:.4f} over {eval_eps} episodes\n")
            logs["exploitability_series"].append(
                {
                    "predicted_avg": None,
                    "p_first": None,
                    "p_second": None,
                    "rollout_avg": obs_avg,
                    "rollout_seat1": seat_results["A_seat1"],
                    "rollout_seat2": seat_results["A_seat2"],
                }
            )

        # Mix into average policy
        curr_av = mix_policies(curr_av, b_i, eta)

    return curr_av, logs
