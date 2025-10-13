from __future__ import annotations

import random
from typing import Dict

from liars_poker.core import GameSpec
from liars_poker.env import Env
from liars_poker.policies.base import Policy


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


def eval_both_seats(spec: GameSpec, A: Policy, B: Policy, episodes: int = 2000, seed: int = 0) -> Dict[str, int]:
    rng = random.Random(seed)
    env = Env(spec)
    A.bind_rules(env.rules)
    B.bind_rules(env.rules)

    first_half = episodes // 2
    second_half = episodes - first_half

    wins_ab = play_match(env, A, B, episodes=first_half, seed=rng.randint(0, 2_147_483_647))
    wins_ba = play_match(env, B, A, episodes=second_half, seed=rng.randint(0, 2_147_483_647))

    return {
        "A": wins_ab["P1"] + wins_ba["P2"],
        "B": wins_ab["P2"] + wins_ba["P1"],
        "total": (wins_ab["P1"] + wins_ba["P2"]) + (wins_ab["P2"] + wins_ba["P1"]),
    }

