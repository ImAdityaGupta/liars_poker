from __future__ import annotations

import argparse

from liars_poker.core import GameSpec
from liars_poker.io.run_manager import RunManager
from liars_poker.policies.random import RandomPolicy
from liars_poker.training.configs import FSPConfig
from liars_poker.training.fsp_trainer import FSPTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train FSP run for Liar's Poker.")
    parser.add_argument("--ranks", type=int, default=13, help="Number of ranks in the deck.")
    parser.add_argument("--suits", type=int, default=1, help="Number of suits in the deck.")
    parser.add_argument("--hand-size", type=int, default=2, help="Cards per player.")
    parser.add_argument("--iters", type=int, default=1, help="Number of FSP iterations to execute.")
    parser.add_argument("--episodes0", type=int, default=10000, help="Best-response simulation episodes.")
    parser.add_argument("--epsilon", type=float, default=0.1, help="Exploration epsilon for BR sampling.")
    parser.add_argument("--min-visits", type=int, default=1, help="Minimum visits per action before epsilon-greedy.")
    parser.add_argument("--seed", type=int, default=0, help="Master seed for reproducibility.")
    parser.add_argument("--save-root", default="artifacts", help="Directory root for run artifacts.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    spec = GameSpec(ranks=args.ranks, suits=args.suits, hand_size=args.hand_size)
    run = RunManager(spec, save_root=args.save_root, seed=args.seed)

    init_policy = RandomPolicy()
    a0_id = run.log_policy(
        init_policy,
        role="average",
        parents=[],
        mixing=None,
        seed=args.seed,
        train={"algo": "init_random"},
        notes="Initialized uniform random policy.",
    )

    config = FSPConfig(
        episodes0=args.episodes0,
        epsilon=args.epsilon,
        min_visits=args.min_visits,
        max_iters=args.iters,
        seed=args.seed,
    )
    trainer = FSPTrainer(run, config)

    print("Run directory:", run.run_dir)
    print("Initial average policy:", a0_id)

    for iter_index in range(config.max_iters):
        opponent = run.current_policy()
        _, _, br_id, avg_id, metrics = trainer.step(opponent, iter_index)
        eta = metrics.get("eta")
        states = metrics.get("br_states")
        print(f"Iter {iter_index:02d} | BR: {br_id} | AVG: {avg_id} | eta={eta:.4f} | states={states}")


if __name__ == "__main__":
    main()

