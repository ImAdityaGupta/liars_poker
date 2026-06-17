from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

from liars_poker.algo.br_exact_dense_to_dense import best_response_dense
from liars_poker.algo.deep_cfr import DeepCFRTrainer
from liars_poker.algo.deep_cfr_diagnostics import ExactDenseStrategyAverager
from liars_poker.core import GameSpec
from liars_poker.policies.neural import NeuralPolicy, compile_neural_to_dense
from liars_poker.serialization import save_policy


def _exact_evaluation(
    spec: GameSpec,
    trainer: DeepCFRTrainer,
    exact_averager: ExactDenseStrategyAverager | None = None,
) -> Dict[str, float]:
    average_dense = compile_neural_to_dense(trainer.average_policy())
    _, average_meta = best_response_dense(
        spec,
        average_dense,
        debug=False,
        store_state_values=False,
    )
    p_first, p_second = average_meta["computer"].exploitability()

    current_dense = trainer.current_policy_dense()
    _, current_meta = best_response_dense(
        spec,
        current_dense,
        debug=False,
        store_state_values=False,
    )
    current_p_first, current_p_second = current_meta["computer"].exploitability()

    result = {
        "p_first": p_first,
        "p_second": p_second,
        "predicted_avg": 0.5 * (p_first + p_second),
        "current_p_first": current_p_first,
        "current_p_second": current_p_second,
        "current_predicted_avg": 0.5 * (current_p_first + current_p_second),
    }
    if exact_averager is not None:
        exact_average = exact_averager.average_policy()
        _, exact_meta = best_response_dense(
            spec,
            exact_average,
            debug=False,
            store_state_values=False,
        )
        exact_p_first, exact_p_second = exact_meta["computer"].exploitability()
        result.update(
            {
                "exact_average_p_first": exact_p_first,
                "exact_average_p_second": exact_p_second,
                "exact_average_predicted_avg": 0.5 * (exact_p_first + exact_p_second),
            }
        )
    return result


def _observe_exact_average(
    trainer: DeepCFRTrainer,
    exact_averager: ExactDenseStrategyAverager | None,
    *,
    every: int = 1,
) -> None:
    if exact_averager is None:
        return
    if every <= 0 or trainer.iteration % every != 0:
        return
    weight = 1.0 if trainer.strategy_weighting == "uniform" else float(trainer.iteration + 1)
    exact_averager.observe(trainer.current_policy_dense(), weight=weight)


def deep_cfr_loop(
    spec: GameSpec,
    iterations: int,
    *,
    trainer: DeepCFRTrainer | None = None,
    trainer_kwargs: Dict[str, object] | None = None,
    traversals_per_player: int = 100,
    eval_every: int = 0,
    exact_averager: ExactDenseStrategyAverager | None = None,
    exact_average_every: int = 1,
    debug: bool = False,
) -> Tuple[NeuralPolicy, Dict[str, object], DeepCFRTrainer]:
    if trainer is None:
        trainer = DeepCFRTrainer(spec, **(trainer_kwargs or {}))

    logs: Dict[str, object] = {
        "training_series": [],
        "exploitability_series": [],
    }
    start = time.perf_counter()

    for _ in range(iterations):
        _observe_exact_average(
            trainer,
            exact_averager,
            every=exact_average_every,
        )
        record = trainer.run_iteration(traversals_per_player=traversals_per_player)
        if trainer.validation_fraction > 0.0:
            record["validation"] = trainer.validation_metrics()
        record["elapsed_s"] = time.perf_counter() - start
        logs["training_series"].append(record)

        if eval_every and trainer.iteration % eval_every == 0:
            logs["exploitability_series"].append(
                {
                    "iter": trainer.iteration,
                    **_exact_evaluation(spec, trainer, exact_averager),
                }
            )

        if debug:
            timing = record["timing"]
            retained = sum(record["advantage_buffer_sizes"]) + sum(record["strategy_buffer_sizes"])
            seen = sum(record["advantage_records_seen"]) + sum(record["strategy_records_seen"])
            print(
                f"[deep-cfr] iter={trainer.iteration} "
                f"elapsed={time.perf_counter() - start:.2f}s "
                f"traverse={timing['traversal_s']:.2f}s "
                f"fit={timing['advantage_training_s'] + timing['strategy_training_s']:.2f}s "
                f"buffer={retained}/{seen}"
            )

    return trainer.average_policy(), logs, trainer


def deep_cfr_timed_loop(
    spec: GameSpec,
    training_seconds: float,
    *,
    trainer: DeepCFRTrainer | None = None,
    trainer_kwargs: Dict[str, object] | None = None,
    traversals_per_player: int = 100,
    eval_every: int = 0,
    exact_averager: ExactDenseStrategyAverager | None = None,
    exact_average_every: int = 1,
    final_eval: bool = True,
    debug: bool = False,
) -> Tuple[NeuralPolicy, Dict[str, object], DeepCFRTrainer]:
    """Train for a fixed wall-clock budget, excluding exact-evaluation time.

    Setting exact_average_every above one samples iteration strategies rather
    than producing the full per-iteration exact generated average.
    """

    if trainer is None:
        trainer = DeepCFRTrainer(spec, **(trainer_kwargs or {}))

    logs: Dict[str, object] = {
        "training_series": [],
        "exploitability_series": [],
    }
    training_elapsed = 0.0

    while training_elapsed < training_seconds:
        _observe_exact_average(
            trainer,
            exact_averager,
            every=exact_average_every,
        )
        start = time.perf_counter()
        record = trainer.run_iteration(traversals_per_player=traversals_per_player)
        iteration_s = time.perf_counter() - start
        training_elapsed += iteration_s
        if trainer.validation_fraction > 0.0:
            record["validation"] = trainer.validation_metrics()
        record["elapsed_s"] = training_elapsed
        logs["training_series"].append(record)

        if eval_every and trainer.iteration % eval_every == 0:
            logs["exploitability_series"].append(
                {
                    "iter": trainer.iteration,
                    **_exact_evaluation(spec, trainer, exact_averager),
                }
            )

        if debug:
            timing = record["timing"]
            print(
                f"[deep-cfr] iter={trainer.iteration} "
                f"training_budget={training_elapsed:.2f}/{training_seconds:.2f}s "
                f"traverse={timing['traversal_s']:.2f}s "
                f"fit={timing['advantage_training_s'] + timing['strategy_training_s']:.2f}s"
            )

    if final_eval and (
        not logs["exploitability_series"]
        or logs["exploitability_series"][-1]["iter"] != trainer.iteration
    ):
        logs["exploitability_series"].append(
            {
                "iter": trainer.iteration,
                **_exact_evaluation(spec, trainer, exact_averager),
            }
        )

    return trainer.average_policy(), logs, trainer


def save_deep_cfr_run(
    run_dir: str | Path,
    *,
    policy: NeuralPolicy,
    trainer: DeepCFRTrainer,
    logs: Dict[str, object],
) -> None:
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)

    save_policy(policy, str(run_path / "policy"))
    trainer.save_checkpoint(run_path / "deep_cfr_checkpoint.pt")

    metrics = {
        "run_type": "deep_cfr",
        "spec": trainer.spec.to_json(),
        "iterations": trainer.iteration,
        "training_series": logs.get("training_series", []),
        "exploitability_series": logs.get("exploitability_series", []),
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    (run_path / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def load_deep_cfr_run(
    run_dir: str | Path,
    *,
    device: str = "cpu",
) -> Tuple[DeepCFRTrainer, Dict[str, object]]:
    run_path = Path(run_dir)
    trainer = DeepCFRTrainer.load_checkpoint(
        run_path / "deep_cfr_checkpoint.pt",
        device=device,
    )
    metrics_path = run_path / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    return trainer, metrics


def deep_cfr_resume(
    run_dir: str | Path,
    *,
    iterations: int,
    traversals_per_player: int = 100,
    eval_every: int = 0,
    device: str = "cpu",
    debug: bool = False,
) -> Tuple[NeuralPolicy, Dict[str, object], DeepCFRTrainer]:
    trainer, metrics = load_deep_cfr_run(run_dir, device=device)
    policy, new_logs, trainer = deep_cfr_loop(
        trainer.spec,
        iterations,
        trainer=trainer,
        traversals_per_player=traversals_per_player,
        eval_every=eval_every,
        debug=debug,
    )
    logs = {
        "training_series": list(metrics.get("training_series", []) or [])
        + list(new_logs.get("training_series", []) or []),
        "exploitability_series": list(metrics.get("exploitability_series", []) or [])
        + list(new_logs.get("exploitability_series", []) or []),
    }
    return policy, logs, trainer
