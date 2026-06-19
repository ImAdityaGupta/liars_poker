from __future__ import annotations

import time
from typing import Dict, Tuple

from liars_poker.algo.br_neural import NeuralBRTrainer
from liars_poker.policies.neural_q import NeuralQPolicy


def _evaluate_both_roles(
    trainer: NeuralBRTrainer,
    episodes_per_role: int,
    rollout_batch_size: int,
) -> Dict[str, float]:
    first = trainer.evaluate_role(
        0,
        episodes_per_role,
        rollout_batch_size=rollout_batch_size,
    )
    second = trainer.evaluate_role(
        1,
        episodes_per_role,
        rollout_batch_size=rollout_batch_size,
    )
    p_first = float(first["win_rate"])
    p_second = float(second["win_rate"])
    return {
        "p_first": p_first,
        "p_second": p_second,
        "exploitability_lower_estimate": p_first + p_second - 1.0,
        "episodes_per_role": int(episodes_per_role),
    }


def neural_br_loop(
    trainer: NeuralBRTrainer,
    iterations: int,
    *,
    episodes_per_role: int = 4096,
    rollout_batch_size: int = 4096,
    eval_every: int = 0,
    eval_episodes_per_role: int = 100_000,
    debug: bool = False,
) -> Tuple[NeuralQPolicy, Dict[str, object], NeuralBRTrainer]:
    logs: Dict[str, object] = {"training_series": [], "evaluation_series": []}
    start = time.perf_counter()
    for _ in range(int(iterations)):
        record = trainer.run_iteration(
            episodes_per_role=episodes_per_role,
            rollout_batch_size=rollout_batch_size,
        )
        record["elapsed_s"] = time.perf_counter() - start
        logs["training_series"].append(record)
        if eval_every and trainer.iteration % eval_every == 0:
            evaluation = {
                "iter": trainer.iteration,
                "elapsed_s": time.perf_counter() - start,
                **_evaluate_both_roles(
                    trainer,
                    eval_episodes_per_role,
                    rollout_batch_size,
                ),
            }
            logs["evaluation_series"].append(evaluation)
            if debug:
                print(
                    f"[neural-br] iter={trainer.iteration} "
                    f"lower_estimate={evaluation['exploitability_lower_estimate']:.6f}"
                )
        elif debug:
            role0, role1 = record["roles"]
            print(
                f"[neural-br] iter={trainer.iteration} "
                f"collect={role0['collect_s'] + role1['collect_s']:.2f}s "
                f"fit={role0['fit_s'] + role1['fit_s']:.2f}s "
                f"replay={role0['replay_size'] + role1['replay_size']}"
            )
    return trainer.policy(), logs, trainer


def neural_br_timed_loop(
    trainer: NeuralBRTrainer,
    training_seconds: float,
    *,
    episodes_per_role: int = 4096,
    rollout_batch_size: int = 4096,
    eval_every_seconds: float = 0.0,
    eval_episodes_per_role: int = 100_000,
    final_eval: bool = True,
    debug: bool = False,
) -> Tuple[NeuralQPolicy, Dict[str, object], NeuralBRTrainer]:
    """Train for a measured budget excluding periodic held-out evaluations."""

    logs: Dict[str, object] = {"training_series": [], "evaluation_series": []}
    measured_training_s = 0.0
    next_eval = float(eval_every_seconds) if eval_every_seconds > 0 else float("inf")

    while measured_training_s < float(training_seconds):
        start = time.perf_counter()
        record = trainer.run_iteration(
            episodes_per_role=episodes_per_role,
            rollout_batch_size=rollout_batch_size,
        )
        measured_training_s += time.perf_counter() - start
        record["measured_training_s"] = measured_training_s
        logs["training_series"].append(record)

        if measured_training_s >= next_eval:
            evaluation = {
                "iter": trainer.iteration,
                "measured_training_s": measured_training_s,
                **_evaluate_both_roles(
                    trainer,
                    eval_episodes_per_role,
                    rollout_batch_size,
                ),
            }
            logs["evaluation_series"].append(evaluation)
            next_eval += float(eval_every_seconds)
            if debug:
                print(
                    f"[neural-br] train={measured_training_s / 60:.1f}m "
                    f"iter={trainer.iteration} "
                    f"lower_estimate={evaluation['exploitability_lower_estimate']:.6f}"
                )
        elif debug:
            print(
                f"[neural-br] train={measured_training_s / 60:.1f}m "
                f"iter={trainer.iteration}"
            )

    if final_eval and (
        not logs["evaluation_series"]
        or logs["evaluation_series"][-1]["iter"] != trainer.iteration
    ):
        logs["evaluation_series"].append(
            {
                "iter": trainer.iteration,
                "measured_training_s": measured_training_s,
                **_evaluate_both_roles(
                    trainer,
                    eval_episodes_per_role,
                    rollout_batch_size,
                ),
            }
        )
    logs["measured_training_s"] = measured_training_s
    return trainer.policy(), logs, trainer
