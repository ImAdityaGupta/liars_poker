from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import Dict, Tuple

from liars_poker.core import GameSpec
from liars_poker.env import Env, rules_for_spec
from liars_poker.infoset import InfoSet
from liars_poker.policies.base import Policy
from liars_poker.policies.tabular import TabularPolicy

State = InfoSet
StateAction = Tuple[State, int]
_TERM = object()


def best_response_mc(
    spec: GameSpec,
    opponent: Policy,
    *,
    episodes: int = 10_000,
    epsilon: float = 0.1,
    min_visits_per_action: int = 1,
    alternate_seats: bool = True,
    seed: int = 0,
    annotate: str = "memory",
) -> TabularPolicy:
    """Monte-Carlo best response against a black-box opponent."""

    rng = random.Random(seed)
    rules = rules_for_spec(spec)
    opponent.bind_rules(rules)
    env = Env(spec)

    # Empirical model containers
    transition_counts: Dict[StateAction, Counter] = defaultdict(Counter)
    visits_sa: Dict[StateAction, int] = defaultdict(int)
    reward_sum: Dict[StateAction, float] = defaultdict(float)

    def seat_for_episode(ep_index: int) -> str:
        if alternate_seats:
            return "P1" if ep_index % 2 == 0 else "P2"
        return "P1" if rng.random() < 0.5 else "P2"

    def pick_action(state: State, legal) -> int:
        need = [a for a in legal if visits_sa[(state, a)] < min_visits_per_action]
        if need:
            return rng.choice(need)
        if rng.random() < epsilon:
            return rng.choice(legal)
        min_visit = min(visits_sa[(state, a)] for a in legal)
        candidates = [a for a in legal if visits_sa[(state, a)] == min_visit]
        return min(candidates)

    for ep in range(episodes):
        obs = env.reset(seed=rng.randint(0, 2_147_483_647))
        me = seat_for_episode(ep)
        opp = "P2" if me == "P1" else "P1"

        opponent.begin_episode(rng)

        while not obs["terminal"]:
            current = env.current_player()
            if current != me:
                opp_infoset = env.infoset_key(current)
                action = opponent.sample(opp_infoset, rng)
                obs = env.step(action)
                continue

            state = env.infoset_key(me)
            legal = tuple(env.legal_actions())
            action = pick_action(state, legal)
            obs = env.step(action)

            visits_sa[(state, action)] += 1

            if obs["terminal"]:
                reward = 1.0 if obs["winner"] == me else -1.0
                transition_counts[(state, action)][_TERM] += 1
                reward_sum[(state, action)] += reward
                break

            # Opponent acts once
            opp_infoset = env.infoset_key(opp)
            opp_action = opponent.sample(opp_infoset, rng)
            obs = env.step(opp_action)

            if obs["terminal"]:
                reward = 1.0 if obs["winner"] == me else -1.0
                transition_counts[(state, action)][_TERM] += 1
                reward_sum[(state, action)] += reward
                break

            next_state = env.infoset_key(me)
            transition_counts[(state, action)][next_state] += 1

    # Backward induction on empirical DAG
    states = set()
    for (state, _), counter in transition_counts.items():
        states.add(state)
        for successor in counter:
            if successor is not _TERM:
                states.add(successor)

    ordered_states = sorted(states, key=lambda s: len(s.history), reverse=True)
    state_values: Dict[State, float] = {}
    state_visit_totals: Dict[State, int] = defaultdict(int)
    best_action: Dict[State, int] = {}

    # Aggregate state visit totals
    for (state, action), count in visits_sa.items():
        state_visit_totals[state] += count

    for state in ordered_states:
        actions_here = sorted({a for (s, a) in visits_sa.keys() if s == state})
        if not actions_here:
            state_values[state] = 0.0
            continue

        best_val = float("-inf")
        chosen_action = None

        for action in actions_here:
            n_sa = visits_sa[(state, action)]
            if n_sa <= 0:
                continue

            immediate = reward_sum[(state, action)] / n_sa
            total = immediate

            for successor, count in transition_counts[(state, action)].items():
                if successor is _TERM:
                    continue
                total += (count / n_sa) * state_values.get(successor, 0.0)

            if (total > best_val) or (total == best_val and (chosen_action is None or action < chosen_action)):
                best_val = total
                chosen_action = action

        if chosen_action is None:
            state_values[state] = 0.0
        else:
            state_values[state] = best_val
            best_action[state] = chosen_action

    policy = TabularPolicy()
    policy.bind_rules(rules)
    for state, action in best_action.items():
        policy.set(state, {action: 1.0})

    if annotate == "memory":
        policy.set_annotations(values=state_values, visits=state_visit_totals)
    elif annotate != "none":
        raise ValueError(f"Unknown annotate mode: {annotate}")

    return policy
