from __future__ import annotations

from liars_poker.core import GameSpec, possible_starting_hands
from liars_poker.policies.tabular import Policy, TabularPolicy
from liars_poker.policies import CommitOnceMixture, RandomPolicy
from liars_poker.infoset import InfoSet
from liars_poker.env import resolve_call_winner, Rules
from liars_poker import eval_both_seats


from typing import Dict, Tuple, List, Callable, Optional
import random
import scipy.stats as stats

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
    initial_pol: Policy | None, 
    br_fn: Callable[[GameSpec, Policy, Optional[bool]], Policy],
    episodes: int,
    *,
    eta_control: Optional[Callable[[int], float]] | None = None,
    episodes_test: Optional[int] = 10_000,
    debug: bool = False

) -> TabularPolicy:

    if initial_pol is None:
        initial_pol = RandomPolicy()

    if eta_control is None:
        eta_control = basic_eta_control

    rules = Rules(spec)
    initial_pol.bind_rules(rules)

    all_averages = [initial_pol]
    all_brs = []

    curr_av = initial_pol



    last_exploitability = 1
    for i in range(episodes):
        if debug:
            print(i)

        eta = eta_control(i)

        b_i, br_computer = br_fn(spec=spec, policy=curr_av, debug=False)
        p_first, p_second = br_computer.exploitability()
        predicted = 0.5 * (p_first + p_second)

        eval_results = eval_both_seats(spec, b_i, curr_av, episodes=episodes_test, seed=random.randint(1,1000))
        observed_rate = eval_results['P1']

        expected_successes = episodes_test * predicted
        expected_failures = episodes_test * (1 - predicted)
        observed_failures = episodes_test * (1 - observed_rate)

        chi2_stat = 0.0
        if expected_successes > 0 and expected_failures > 0:
            chi2_stat = (((episodes_test*observed_rate) - expected_successes) ** 2) / expected_successes + ((observed_failures - expected_failures) ** 2) / expected_failures
        p_value = 1 - stats.chi2.cdf(chi2_stat, 1)


        all_brs.append(b_i)

        last_exploitablity = observed_rate
        print(f"Predicted exploitability: avg={predicted:.4f} (first={p_first:.4f}, second={p_second:.4f})")
        print(f"Sampled exploitability: avg={observed_rate:.4f}, chi2 p-value={p_value:.4g}")
        print()

        curr_av = mix_policies(curr_av, b_i, eta)
    all_averages.append(curr_av)

    return curr_av

