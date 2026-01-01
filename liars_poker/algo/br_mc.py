from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import Dict, Tuple

from liars_poker.core import GameSpec
from liars_poker.env import Env, rules_for_spec
from liars_poker.infoset import InfoSet
from liars_poker.policies.base import Policy
from liars_poker.policies.tabular import TabularPolicy

from typing import List, Optional

State = InfoSet
StateAction = Tuple[State, int]
_TERM = object()


def best_response_mc(
    spec: GameSpec,
    policy: Policy,
    *,
    episodes: int = 10_000,
    epsilon: float = 0.1,
    min_visits_per_action: int = 1,
    alternate_seats: bool = True,
    seed: int = 0,
    annotate: str = "memory",
    debug: bool = False
) -> TabularPolicy:
    """Monte-Carlo best response against a black-box opponent."""

    rng = random.Random(seed)
    rules = rules_for_spec(spec)
    policy.bind_rules(rules)
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

        policy.begin_episode(rng)

        while not obs["terminal"]:
            current = env.current_player()
            if current != me:
                opp_infoset = env.infoset_key(current)
                action = policy.sample(opp_infoset, rng)
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
            opp_action = policy.sample(opp_infoset, rng)
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

    new_policy = TabularPolicy()
    new_policy.bind_rules(rules)
    for state, action in best_action.items():
        new_policy.set(state, {action: 1.0})

    if annotate == "memory":
        new_policy.set_annotations(values=state_values, visits=state_visit_totals)
    elif annotate != "none":
        raise ValueError(f"Unknown annotate mode: {annotate}")
    
    dict_log = {"computes_exploitability": False, "computer": None}

    return new_policy, dict_log


def efficient_best_response_mc(
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
    """Monte-Carlo best response against a black-box opponent (optimized Phase 2).

    Same signature/return as best_response_mc. Builds an empirical MDP over OUR
    infosets, then solves it via a backward pass using dense integer indexing.
    """
    rng = random.Random(seed)
    rules = rules_for_spec(spec)
    opponent.bind_rules(rules)
    env = Env(spec)

    # --- Phase 1: collect empirical MDP over OUR infosets ---
    transition_counts: Dict[StateAction, Counter] = defaultdict(Counter)   # (s,a) -> Counter({s': n, _TERM: m})
    visits_sa: Dict[StateAction, int] = {}                                 # (s,a) -> N(s,a)   (plain dict; use .get)
    reward_sum: Dict[StateAction, float] = {}                              # (s,a) -> sum of terminal rewards

    def seat_for_episode(ep_index: int) -> str:
        if alternate_seats:
            return "P1" if (ep_index % 2 == 0) else "P2"
        return "P1" if rng.random() < 0.5 else "P2"

    def pick_action(state: State, legal) -> int:
        # Avoid mutating visits_sa on reads: use .get
        need = [a for a in legal if visits_sa.get((state, a), 0) < min_visits_per_action]
        if need:
            return rng.choice(need)
        if rng.random() < epsilon:
            return rng.choice(legal)
        min_visit = min(visits_sa.get((state, a), 0) for a in legal)
        candidates = [a for a in legal if visits_sa.get((state, a), 0) == min_visit]
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
                a_opp = opponent.sample(opp_infoset, rng)
                obs = env.step(a_opp)
                continue

            state: State = env.infoset_key(me)
            legal = tuple(env.legal_actions())
            a = pick_action(state, legal)
            obs = env.step(a)

            # Increment AFTER action is taken
            key = (state, a)
            visits_sa[key] = visits_sa.get(key, 0) + 1

            if obs["terminal"]:
                reward = 1.0 if obs["winner"] == me else -1.0
                transition_counts[key][_TERM] += 1
                reward_sum[key] = reward_sum.get(key, 0.0) + reward
                break

            # Opponent acts once (strict alternation)
            opp_infoset = env.infoset_key(opp)
            a_opp = opponent.sample(opp_infoset, rng)
            obs = env.step(a_opp)

            if obs["terminal"]:
                reward = 1.0 if obs["winner"] == me else -1.0
                transition_counts[key][_TERM] += 1
                reward_sum[key] = reward_sum.get(key, 0.0) + reward
                break

            # Next OUR infoset
            s_prime: State = env.infoset_key(me)
            transition_counts[key][s_prime] += 1

    # --- Phase 2: efficient backward induction on the empirical DAG ---

    # Collect all states encountered
    states = set()
    for (state, _), counter in transition_counts.items():
        states.add(state)
        for successor in counter:
            if successor is not _TERM:
                states.add(successor)

    ordered_states = sorted(states, key=lambda s: len(s.history), reverse=True)
    state_index = {state: idx for idx, state in enumerate(ordered_states)}

    n_states = len(ordered_states)

    policy = TabularPolicy()
    policy.bind_rules(rules)

    if n_states == 0:
        if annotate == "memory":
            policy.set_annotations(values={}, visits={})
        elif annotate != "none":
            raise ValueError(f"Unknown annotate mode: {annotate}")
        return policy

    # Prepare per-state structures
    state_actions: List[List[int]] = [[] for _ in range(n_states)]
    n_sa = defaultdict(dict)
    r_sa = defaultdict(dict)
    succ_ids = defaultdict(dict)
    succ_probs = defaultdict(dict)

    record_visits = annotate == "memory"
    state_visit_totals = [0] * n_states if record_visits else None

    for (state, action), count in visits_sa.items():
        if count <= 0:
            continue
        sid = state_index[state]
        state_actions[sid].append(action)
        if record_visits:
            state_visit_totals[sid] += count

        n_sa[sid][action] = count
        reward_total = reward_sum.get((state, action), 0.0)
        r_sa[sid][action] = reward_total / count

        counter = transition_counts.get((state, action), {})
        successor_ids: List[int] = []
        successor_probs: List[float] = []
        for successor, succ_count in counter.items():
            if successor is _TERM:
                continue
            successor_ids.append(state_index[successor])
            successor_probs.append(succ_count / count)
        succ_ids[sid][action] = successor_ids
        succ_probs[sid][action] = successor_probs

    V = [0.0] * n_states
    best_action = [-1] * n_states

    for state in ordered_states:
        sid = state_index[state]
        actions = state_actions[sid]
        if not actions:
            V[sid] = 0.0
            continue

        best_value = float("-inf")
        chosen = -1

        for action in actions:
            total = r_sa[sid][action]
            for successor_id, prob in zip(succ_ids[sid][action], succ_probs[sid][action]):
                total += prob * V[successor_id]

            if (total > best_value) or (total == best_value and (chosen == -1 or action < chosen)):
                best_value = total
                chosen = action

        V[sid] = best_value if chosen != -1 else 0.0
        best_action[sid] = chosen

    for state, sid in state_index.items():
        action = best_action[sid]
        if action != -1:
            policy.set(state, {action: 1.0})

    if annotate == "memory":
        values_map = {state: V[state_index[state]] for state in ordered_states}
        visits_map = (
            {state: state_visit_totals[state_index[state]] for state in ordered_states}
            if state_visit_totals is not None
            else {}
        )
        policy.set_annotations(values=values_map, visits=visits_map)
    elif annotate != "none":
        raise ValueError(f"Unknown annotate mode: {annotate}")

    return policy


def efficient_best_response_mc_v2(spec, opponent, *, episodes=10_000, epsilon=0.1,
                               min_visits_per_action=1, alternate_seats=True,
                               seed=0, annotate="memory") -> TabularPolicy:
    rng = random.Random(seed)
    rules = rules_for_spec(spec)
    opponent.bind_rules(rules)
    env = Env(spec)

    class SAStats:
        __slots__ = ("visits", "reward_sum", "term_count", "next_counts")
        def __init__(self):
            self.visits = 0; self.reward_sum = 0.0; self.term_count = 0
            self.next_counts = {}

    sa = defaultdict(dict)                 # Dict[State, Dict[int, SAStats]]
    state_visit_totals = defaultdict(int)  # Dict[State, int]

    def seat_for(ep):
        if alternate_seats: return "P1" if ep % 2 == 0 else "P2"
        return "P1" if rng.random() < 0.5 else "P2"

    for ep in range(episodes):
        env.reset(seed=rng.randint(0, 2_147_483_647))
        me = seat_for(ep); opp = "P2" if me == "P1" else "P1"
        opponent.begin_episode(rng)

        while not env._done:
            current = env.current_player()
            if current != me:
                opp_s = env.infoset_key(current)
                opp_a = opponent.sample(opp_s, rng)
                env.step(opp_a)
                continue

            s = env.infoset_key(me)
            legal = rules.legal_actions_for(s)

            need = [a for a in legal if sa[s].get(a, None) is None or sa[s][a].visits < min_visits_per_action]
            if need:
                a = rng.choice(need)
            elif rng.random() < epsilon:
                a = rng.choice(legal)
            else:
                minv = min((sa[s].get(a, None).visits if a in sa[s] else 0) for a in legal)
                candidates = [a for a in legal if (sa[s].get(a, None).visits if a in sa[s] else 0) == minv]
                a = min(candidates)

            st = sa[s].get(a)
            if st is None:
                st = sa[s][a] = SAStats()
            st.visits += 1
            state_visit_totals[s] += 1

            env.step(a)
            if env._done:
                st.term_count += 1
                st.reward_sum += (1.0 if (env._winner == 0 and me == "P1") or (env._winner == 1 and me == "P2") else -1.0)
                break

            opp_s = env.infoset_key(opp)
            opp_a = opponent.sample(opp_s, rng)
            env.step(opp_a)

            if env._done:
                st.term_count += 1
                st.reward_sum += (1.0 if (env._winner == 0 and me == "P1") or (env._winner == 1 and me == "P2") else -1.0)
                break

            ns = env.infoset_key(me)
            st.next_counts[ns] = st.next_counts.get(ns, 0) + 1

    # DP over empirical DAG (group by depth)
    states = set(sa.keys())
    for s, actions in sa.items():
        for st in actions.values():
            states.update(st.next_counts.keys())

    by_depth = defaultdict(list)
    for s in states:
        by_depth[len(s.history)].append(s)
    depths = sorted(by_depth.keys(), reverse=True)

    V = {}
    best = {}
    for d in depths:
        for s in by_depth[d]:
            actions = sa.get(s)
            if not actions:
                V[s] = 0.0
                continue
            best_val = float("-inf"); best_a = None
            for a in sorted(actions.keys()):
                st = actions[a]
                n = st.visits
                if n <= 0: continue
                total = (st.reward_sum / n)
                for s2, c in st.next_counts.items():
                    total += (c / n) * V.get(s2, 0.0)
                if (total > best_val) or (total == best_val and (best_a is None or a < best_a)):
                    best_val, best_a = total, a
            if best_a is None:
                V[s] = 0.0
            else:
                V[s] = best_val
                best[s] = best_a

    policy = TabularPolicy()
    policy.bind_rules(rules)
    for s, a in best.items():
        policy.set(s, {a: 1.0})
    if annotate == "memory":
        policy.set_annotations(values=V, visits=state_visit_totals)
    elif annotate != "none":
        raise ValueError(f"Unknown annotate mode: {annotate}")
    return policy


def efficient_best_response_mc_v3(
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
    rng = random.Random(seed)
    rules = rules_for_spec(spec)
    opponent.bind_rules(rules)
    env = Env(spec)

    # --- Interning for InfoSet objects ---
    intern: Dict[Tuple[int, Tuple[int, ...], Tuple[int, ...]], InfoSet] = {}
    def intern_iset(iset: InfoSet) -> InfoSet:
        key = (iset.pid, iset.hand, iset.history)
        cached = intern.get(key)
        if cached is None:
            intern[key] = iset
            return iset
        return cached

    # --- Empirical model (leaner structures) ---
    transitions: Dict[Tuple[State, int], Dict[object, int]] = {}
    visits_sa: Dict[Tuple[State, int], int] = {}
    reward_sum: Dict[Tuple[State, int], float] = {}
    actions_by_state: Dict[State, set[int]] = {}
    # (Optionally) state_visit_totals if you want to keep the annotation:
    state_visit_totals: Dict[State, int] = {}

    _TERM = object()

    def seat_for_episode(ep_index: int) -> str:
        if alternate_seats:
            return "P1" if ep_index % 2 == 0 else "P2"
        return "P1" if rng.random() < 0.5 else "P2"

    def pick_action(state: State, legal: Tuple[int, ...]) -> int:
        # IDENTICAL logic to best_response_mc
        need = [a for a in legal if visits_sa.get((state, a), 0) < min_visits_per_action]
        if need:
            return rng.choice(need)
        if rng.random() < epsilon:
            return rng.choice(legal)
        min_visit = min(visits_sa.get((state, a), 0) for a in legal)
        candidates = [a for a in legal if visits_sa.get((state, a), 0) == min_visit]
        return min(candidates)

    for ep in range(episodes):
        obs = env.reset(seed=rng.randint(0, 2_147_483_647))
        me = seat_for_episode(ep)
        opp = "P2" if me == "P1" else "P1"

        opponent.begin_episode(rng)

        while not obs["terminal"]:
            current = env.current_player()
            if current != me:
                opp_iset = intern_iset(env.infoset_key(current))
                opp_action = opponent.sample(opp_iset, rng)
                obs = env.step(opp_action)
                continue

            # Our turn
            s = intern_iset(env.infoset_key(me))
            legal = tuple(env.legal_actions())
            a = pick_action(s, legal)
            obs = env.step(a)

            key = (s, a)
            visits_sa[key] = visits_sa.get(key, 0) + 1
            actions_by_state.setdefault(s, set()).add(a)
            state_visit_totals[s] = state_visit_totals.get(s, 0) + 1

            if obs["terminal"]:
                reward = 1.0 if obs["winner"] == me else -1.0
                bucket = transitions.get(key)
                if bucket is None:
                    bucket = {}
                    transitions[key] = bucket
                bucket[_TERM] = bucket.get(_TERM, 0) + 1
                reward_sum[key] = reward_sum.get(key, 0.0) + reward
                break

            # Opponent acts once
            opp_iset = intern_iset(env.infoset_key(opp))
            opp_action = opponent.sample(opp_iset, rng)
            obs = env.step(opp_action)

            if obs["terminal"]:
                reward = 1.0 if obs["winner"] == me else -1.0
                bucket = transitions.get(key)
                if bucket is None:
                    bucket = {}
                    transitions[key] = bucket
                bucket[_TERM] = bucket.get(_TERM, 0) + 1
                reward_sum[key] = reward_sum.get(key, 0.0) + reward
                break

            # Successor: our next decision state
            succ = intern_iset(env.infoset_key(me))
            bucket = transitions.get(key)
            if bucket is None:
                bucket = {}
                transitions[key] = bucket
            bucket[succ] = bucket.get(succ, 0) + 1

    # ----- Backward induction (same math, cheaper lookups) -----
    # Gather all states
    states: set[State] = set()
    for (s, _), succs in transitions.items():
        states.add(s)
        for t in succs:
            if t is not _TERM:
                states.add(t)

    ordered_states = sorted(states, key=lambda st: len(st.history), reverse=True)

    state_values: Dict[State, float] = {}
    best_action: Dict[State, int] = {}

    for s in ordered_states:
        actions_here = actions_by_state.get(s)
        if not actions_here:
            state_values[s] = 0.0
            continue

        best_val = float("-inf")
        chosen_action: Optional[int] = None

        for a in actions_here:
            key = (s, a)
            n_sa = visits_sa.get(key, 0)
            if n_sa <= 0:
                continue

            total = (reward_sum.get(key, 0.0) / n_sa)
            for t, cnt in transitions.get(key, {}).items():
                if t is _TERM:
                    continue
                total += (cnt / n_sa) * state_values.get(t, 0.0)

            if (total > best_val) or (total == best_val and (chosen_action is None or a < chosen_action)):
                best_val = total
                chosen_action = a

        if chosen_action is None:
            state_values[s] = 0.0
        else:
            state_values[s] = best_val
            best_action[s] = chosen_action

    policy = TabularPolicy()
    policy.bind_rules(rules)
    for s, a in best_action.items():
        policy.set(s, {a: 1.0})

    if annotate == "memory":
        policy.set_annotations(values=state_values, visits=state_visit_totals)
    elif annotate != "none":
        raise ValueError(f"Unknown annotate mode: {annotate}")

    return policy
