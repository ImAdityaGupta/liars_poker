# br.py
from __future__ import annotations

import math
import random
from collections import defaultdict, Counter
from typing import Any, Dict, Optional, Tuple

from .core import GameSpec
from .env import Env, rules_for_spec
from .policy import Policy, RandomPolicy, TabularPolicy

# Type alias for our infoset keys (hashable tuples from Env.infoset_key)
State = Tuple[Any, ...]
# Private sentinel for absorbing terminal within this module
_TERM = object()


def best_response_exact(spec: GameSpec, opponent: Policy, who: str) -> Policy:
    """Placeholder for exact BR on tiny games."""
    rules = rules_for_spec(spec)
    opponent.bind_rules(rules)
    br = RandomPolicy()
    br.bind_rules(rules)
    _ = who
    return br


def best_response_mc(
    spec: GameSpec,
    opponent: Policy,
    *,
    episodes: int = 10_000,
    epsilon: float = 0.1,
    min_visits_per_action: int = 1,
    alternate_seats: bool = True,
    seed: int = 0,
) -> TabularPolicy:
    """
    Monte-Carlo expectimax best response vs a black-box opponent (seat-agnostic).

    Phase 1 (model): sample episodes vs `opponent` while WE control one seat
    (alternating by default). At our infosets, explore actions and record a
    transition to either the next OUR infoset (after one opponent move) or
    terminal with reward (+1/-1 from our perspective).

    Phase 2 (solve): single backward induction on the empirical DAG to get V(s)
    and argmax actions, then return a deterministic TabularPolicy containing
    actions for both seats' infosets.
    """
    rng = random.Random(seed)
    rules = rules_for_spec(spec)
    opponent.bind_rules(rules)
    env = Env(spec)

    # --- Phase 1: collect empirical MDP over OUR infosets ---
    counts: Dict[Tuple[State, int], Counter] = defaultdict(Counter)   # (s,a) -> Counter({s': n, _TERM: m})
    visits: Dict[Tuple[State, int], int] = defaultdict(int)           # (s,a) -> N(s,a)
    rew_sum: Dict[Tuple[State, int], float] = defaultdict(float)      # (s,a) -> sum of terminal rewards

    def seat_for_episode(ep: int) -> str:
        if alternate_seats:
            return "P1" if (ep % 2 == 0) else "P2"
        return "P1" if rng.random() < 0.5 else "P2"

    def pick_action(s: State, legal: Tuple[int, ...]) -> int:
        # Ensure minimum coverage per action at this state
        need = [a for a in legal if visits[(s, a)] < min_visits_per_action]
        if need:
            return rng.choice(need)
        # ε-greedy exploration
        if rng.random() < epsilon:
            return rng.choice(legal)
        # Otherwise prefer least-visited; deterministic tie-break
        lv = min(visits[(s, a)] for a in legal)
        cands = [a for a in legal if visits[(s, a)] == lv]
        return min(cands)

    for ep in range(episodes):
        obs = env.reset(seed=rng.randint(0, 2_147_483_647))
        me = seat_for_episode(ep)
        opp = "P2" if me == "P1" else "P1"

        # Per-episode init (commit-once flips here; others no-op)
        opponent.begin_episode(rng)

        # Main loop
        while not obs["terminal"]:
            if env.current_player() != me:
                # This will only trigger if opponent plays first move.
                # If they are starting, they can't call so we won't trip on next clause.
                opp_infoset = env.infoset_key(opp)
                a_opp = opponent.sample(opp_infoset, rng)
                obs = env.step(a_opp)
                continue

            s: State = env.infoset_key(me)
            legal = tuple(env.legal_actions())

            a = pick_action(s, legal)
            obs = env.step(a)

            # If our action ended the game (e.g., CALL)
            if obs["terminal"]:
                reward = 1.0 if obs["winner"] == me else -1.0
                visits[(s, a)] += 1
                counts[(s, a)][_TERM] += 1
                rew_sum[(s, a)] += reward
                break

            # Opponent moves once (strict alternation)
            opp_infoset = env.infoset_key(opp)
            a_opp = opponent.sample(opp_infoset, rng)
            obs = env.step(a_opp)

            if obs["terminal"]:
                reward = 1.0 if obs["winner"] == me else -1.0
                visits[(s, a)] += 1
                counts[(s, a)][_TERM] += 1
                rew_sum[(s, a)] += reward
                break

            # Next OUR infoset
            s_prime: State = env.infoset_key(me)
            visits[(s, a)] += 1
            counts[(s, a)][s_prime] += 1

    # --- Phase 2: solve the empirical DAG via backward induction ---
    states: set[State] = set()
    for (s, a), ctr in counts.items():
        states.add(s)
        for sp in ctr.keys():
            if sp is not _TERM:
                states.add(sp)

    if not states:
        return TabularPolicy()  # fallback: unseen → uniform at use-time

    def depth_of(state: State) -> int:
        # Heuristic: history length is usually at index 3: (pid, last_idx, hand, history)
        try:
            hist = state[3]
            return len(hist) if isinstance(hist, tuple) else 0
        except Exception:
            return 0

    ordered_states = sorted(states, key=depth_of, reverse=True)

    V: Dict[State, float] = {}
    best_action: Dict[State, int] = {}

    for s in ordered_states:
        actions_here = sorted({a for (ss, a) in counts.keys() if ss == s})
        if not actions_here:
            V[s] = 0.0
            continue

        best_val = -math.inf
        best_a: Optional[int] = None

        for a in actions_here:
            n = visits[(s, a)]
            if n <= 0:
                continue

            # Expected immediate reward (only on terminal transitions)
            r_hat = rew_sum[(s, a)] / n

            # Expected next-state value (exclude terminal sentinel)
            total = r_hat
            for sp, c in counts[(s, a)].items():
                if sp is _TERM:
                    continue
                p = c / n
                total += p * V.get(sp, 0.0)

            if (total > best_val) or (total == best_val and (best_a is None or a < best_a)):
                best_val = total
                best_a = a

        V[s] = best_val if best_a is not None else 0.0
        if best_a is not None:
            best_action[s] = best_a



    br = TabularPolicy()
    br.bind_rules(rules)
    for s, a in best_action.items():
        br.set(s, {a: 1.0})
    return br, V


class RLLearner:
    """RL learner base stub (no training yet)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        _ = (args, kwargs)

    def fit_off_policy(self, opponent: Policy, env: "Env", **params: Any) -> None:
        """Train against a fixed opponent in the given environment."""
        raise NotImplementedError

    def export_policy(self) -> Policy:
        """Return the learned policy."""
        raise NotImplementedError
