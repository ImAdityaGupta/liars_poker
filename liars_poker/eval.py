from __future__ import annotations

import random
from typing import Dict

from .env import Env


def play_match(env: Env, p1, p2, episodes: int = 10, seed: int = 0) -> Dict[str, int]:
    rng = random.Random(seed)
    wins = {"P1": 0, "P2": 0}
    for _ in range(episodes):
        obs = env.reset(seed=rng.randint(0, 1_000_000))
        p1.begin_episode(rng)
        p2.begin_episode(rng)
        # Optional: for commit-once mixers, allow manual hook
        for t in range(1000):
            if obs["terminal"]:
                if obs["winner"]:
                    wins[obs["winner"]] += 1
                break
            player = env.current_player()
            pi = p1 if player == "P1" else p2
            infoset = env.infoset_key(player)
            action = pi.sample(infoset, obs["legal_actions"], rng)
            obs = env.step(action)
    return wins


def approx_exploitability(*args, **kwargs) -> float:
    """Placeholder stub."""
    raise NotImplementedError
