from __future__ import annotations

import random
from typing import Dict, Tuple

from liars_poker.core import GameSpec, possible_starting_hands
from liars_poker.env import Env, rules_for_spec, resolve_call_winner
from liars_poker.infoset import InfoSet, CALL
from liars_poker.policies.base import Policy
from liars_poker.policies.tabular import TabularPolicy
from liars_poker.algo.br_exact import adjustment_factor


def play_match(env: Env, p1: Policy, p2: Policy, episodes: int = 10, seed: int = 0) -> Dict[str, int]:
    rng = random.Random(seed)
    wins = {"P1": 0, "P2": 0}

    p1.bind_rules(env.rules)
    p2.bind_rules(env.rules)

    for _ in range(episodes):
        obs = env.reset(seed=rng.randint(0, 2_147_483_647))
        p1.begin_episode(rng)
        p2.begin_episode(rng)

        while True:
            if obs["terminal"]:
                winner = obs["winner"]
                if winner in wins:
                    wins[winner] += 1
                break

            player = env.current_player()
            policy = p1 if player == "P1" else p2
            infoset = env.infoset_key(player)
            action = policy.sample(infoset, rng)
            obs = env.step(action)

    return wins


def eval_vs(spec: GameSpec, opponent: Policy, candidate: Policy, episodes: int = 2000, seed: int = 0) -> Dict[str, int]:
    env = Env(spec)
    opponent.bind_rules(env.rules)
    candidate.bind_rules(env.rules)
    return play_match(env, candidate, opponent, episodes=episodes, seed=seed)


def eval_both_seats(spec: GameSpec, p1: Policy, p2: Policy, episodes: int = 2000, seed: int = 0) -> Dict[str, int]:
    rng = random.Random(seed)
    env = Env(spec)
    p1.bind_rules(env.rules)
    p2.bind_rules(env.rules)

    first_half = episodes // 2
    second_half = episodes - first_half

    wins_ab = play_match(env, p1, p2, episodes=first_half, seed=rng.randint(0, 2_147_483_647))
    wins_ba = play_match(env, p2, p1, episodes=second_half, seed=rng.randint(0, 2_147_483_647))

    a_wins = wins_ab["P1"] + wins_ba["P2"]
    b_wins = wins_ab["P2"] + wins_ba["P1"]
    total = a_wins + b_wins

    if total == 0:
        return {"P1": 0.0, "P2": 0.0}

    return {"P1": a_wins / total, "P2": b_wins / total}


def exact_eval_tabular(spec: GameSpec, p1: TabularPolicy, p2: TabularPolicy) -> Dict[str, float]:
    """Return the exact win probabilities when two tabular policies play (P1 opens)."""

    rules = rules_for_spec(spec)
    p1.bind_rules(rules)
    p2.bind_rules(rules)

    hands = tuple(possible_starting_hands(spec))

    policy_cache: Dict[Tuple[int, Tuple[int, ...], Tuple[int, ...]], Dict[int, float]] = {}

    def dist_for(policy: TabularPolicy, pid: int, hand: Tuple[int, ...], history: Tuple[int, ...]) -> Dict[int, float]:
        key = (pid, hand, history)
        cached = policy_cache.get(key)
        if cached is not None:
            return cached
        infoset = InfoSet(pid=pid, hand=hand, history=history)
        dist = policy.action_probs(infoset)
        policy_cache[key] = dist
        return dist

    def value(history: Tuple[int, ...], p1_hand: Tuple[int, ...], p2_hand: Tuple[int, ...]) -> float:
        if history and history[-1] == CALL:
            winner = resolve_call_winner(spec, history, p1_hand, p2_hand)
            return 1.0 if winner == "P1" else 0.0

        to_play = len(history) % 2
        hand = p1_hand if to_play == 0 else p2_hand
        dist = dist_for(p1 if to_play == 0 else p2, to_play, hand, history)
        if not dist:
            return 0.0

        total = 0.0
        for action, prob in dist.items():
            total += prob * value(history + (action,), p1_hand, p2_hand)
        return total

    numerator = 0.0
    denominator = 0.0
    for p1_hand in hands:
        weight_p1 = adjustment_factor(spec, (), p1_hand)
        if weight_p1 <= 0:
            continue
        for p2_hand in hands:
            weight_pair = adjustment_factor(spec, p1_hand, p2_hand)
            if weight_pair <= 0:
                continue
            combos = weight_p1 * weight_pair
            numerator += combos * value(tuple(), p1_hand, p2_hand)
            denominator += combos

    if denominator == 0:
        return {"P1": 0.0, "P2": 0.0}

    p1_win = numerator / denominator
    return {"P1": p1_win, "P2": 1.0 - p1_win}


def exact_eval_tabular_both_seats(spec: GameSpec, p1: TabularPolicy, p2: TabularPolicy) -> Dict[str, float]:
    """Exact evaluation averaging over both seating assignments (P1 first, then swapped)."""

    first = exact_eval_tabular(spec, p1, p2)
    second = exact_eval_tabular(spec, p2, p1)

    avg_p1 = 0.5 * (first["P1"] + (1.0 - second["P1"]))
    return {"P1": avg_p1, "P2": 1.0 - avg_p1}
