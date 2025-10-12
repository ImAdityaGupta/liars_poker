#!/usr/bin/env python3
from __future__ import annotations

import random

from liars_poker.core import GameSpec
from liars_poker.env import Env
from liars_poker.policy import RandomPolicy


def main() -> None:
    spec = GameSpec(ranks=13, suits=1, hand_size=2, starter="random", claim_kinds=("RankHigh", "Pair"))
    env = Env(spec, seed=123)
    human = "P1"
    bot = RandomPolicy()
    rng = random.Random(0)

    while True:
        obs = env.reset()
        bot.begin_episode(rng)
        print("New game. You are P1. Your hand:", obs["hand"])  # ints for now
        while not obs["terminal"]:
            player = env.current_player()
            if player == human:
                print("To play:", player)
                last = "None" if obs["last_claim_idx"] is None else env.render_action(obs["last_claim_idx"])
                print("Last claim:", last)
                print("Legal:", [env.render_action(a) for a in obs["legal_actions"]])
                raw = input("Your action (e.g., CALL or RankHigh:10): ").strip()
                try:
                    a = env.parse_action(raw, obs["legal_actions"])
                except Exception as e:
                    print("Error:", e)
                    continue
            else:
                infoset = env.infoset_key(player)
                a = bot.sample(infoset, obs["legal_actions"], rng)
                print(f"Bot plays: {env.render_action(a)}")
            obs = env.step(a)
        print("Winner:", obs["winner"])
        again = input("Play again? [y/N]: ").strip().lower()
        if again != "y":
            break


if __name__ == "__main__":
    main()
