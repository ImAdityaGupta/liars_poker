from __future__ import annotations

import random
from typing import Dict

from .env import Env


def play_match(env: Env, p1, p2, episodes: int = 10, seed: int = 0) -> Dict[str, int]:
    rng = random.Random(seed)
    wins = {"P1": 0, "P2": 0}
    p1.bind_rules(env.rules)
    p2.bind_rules(env.rules)
    for _ in range(episodes):
        obs = env.reset(seed=rng.randint(0, 1_000_000))
        p1.begin_episode(rng)
        p2.begin_episode(rng)
        # Optional: for commit-once mixers, allow manual hook
        while True:
            if obs["terminal"]:
                if obs["winner"]:
                    wins[obs["winner"]] += 1
                break
            player = env.current_player()
            pi = p1 if player == "P1" else p2
            infoset = env.infoset_key(player)
            action = pi.sample(infoset, rng)
            obs = env.step(action)
    return wins

def eval_vs(spec, opponent, candidate, episodes=2000, seed=0):
    env = Env(spec)
    opponent.bind_rules(env.rules)
    candidate.bind_rules(env.rules)
    return play_match(env, candidate, opponent, episodes=episodes, seed=seed)


def eval_both_seats(spec, A, B, episodes=2000, seed=0):
    rng = random.Random(seed)
    env = Env(spec)
    A.bind_rules(env.rules); B.bind_rules(env.rules)

    # A as P1 vs B as P2
    w_AB = play_match(env, A, B, episodes=episodes//2, seed=rng.randint(0, 10**9))
    # swap seats: A as P2 vs B as P1
    w_BA = play_match(env, B, A, episodes=episodes - episodes//2, seed=rng.randint(0, 10**9))

    # Re-map wins to policy identity
    a_wins = w_AB["P1"] + w_BA["P2"]
    b_wins = w_AB["P2"] + w_BA["P1"]
    return {"A": a_wins, "B": b_wins, "total": a_wins + b_wins}


def approx_exploitability(*args, **kwargs) -> float:
    """Placeholder stub."""
    raise NotImplementedError
