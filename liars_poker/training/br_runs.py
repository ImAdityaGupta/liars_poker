from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Dict, Sequence

from liars_poker.algo.br_fitted_return import FittedReturnBRTrainer
from liars_poker.algo.br_fitted_return_action_conditioned import (
    ActionConditionedFittedReturnBRTrainer,
)
from liars_poker.algo.br_neural import NeuralBRTrainer
from liars_poker.algo.br_ppo import PPOBRTrainer
from liars_poker.core import ARTIFACTS_ROOT
from liars_poker.eval.approx_br import evaluate_approximate_br
from liars_poker.eval.match_dense import evaluate_dense_response
from liars_poker.eval.neural_evaluators import compile_policy_to_dense
from liars_poker.serialization import load_policy, save_policy


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (tuple, set)):
        return list(value)
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _append_jsonl(path: Path, record: Dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=_json_default) + "\n")


def _load_target(target):
    if not isinstance(target, (str, Path)):
        return target
    path = Path(target)
    if (path / "policy" / "metadata.json").exists():
        path = path / "policy"
    policy, _ = load_policy(str(path))
    return policy


def _trainer_class(method: str):
    methods = {
        "dqn": NeuralBRTrainer,
        "fitted_return": FittedReturnBRTrainer,
        "action_conditioned_fitted_return": ActionConditionedFittedReturnBRTrainer,
        "ppo": PPOBRTrainer,
    }
    try:
        return methods[method]
    except KeyError as exc:
        raise ValueError(
            f"Unknown responder method {method!r}; choose from {tuple(methods)}."
        ) from exc


@dataclass
class BestResponseRunResult:
    run_dir: Path
    policy: object
    trainer: object
    training_records: list[Dict[str, object]]
    evaluation_records: list[Dict[str, object]]
    measured_training_s: float


def run_best_responder(
    target,
    *,
    method: str = "action_conditioned_fitted_return",
    minutes: float | None = None,
    iterations: int | None = None,
    trainer_kwargs: Dict[str, object] | None = None,
    episodes_per_role: int = 4096,
    rollout_batch_size: int = 4096,
    evaluate_every_minutes: float | None = None,
    evaluate_every_iterations: int | None = None,
    eval_episodes_per_role: int = 200_000,
    exact_evaluation: bool = False,
    exact_compile_batch_size: int = 65_536,
    run_dir: str | Path | None = None,
    debug: bool = False,
) -> BestResponseRunResult:
    """Train one approximate responder against a fixed target policy."""

    if (minutes is None) == (iterations is None):
        raise ValueError("Provide exactly one of minutes or iterations.")

    target_policy = _load_target(target)
    spec = target_policy._require_rules().spec
    trainer_cls = _trainer_class(method)
    trainer = trainer_cls(spec, target_policy, **(trainer_kwargs or {}))

    if run_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        run_path = (
            Path(ARTIFACTS_ROOT)
            / "best_response_runs"
            / method
            / f"{spec.to_short_str()}___{stamp}"
        )
    else:
        run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)

    manifest = {
        "run_type": "approximate_best_response",
        "method": method,
        "spec": spec.to_json(),
        "minutes": minutes,
        "iterations": iterations,
        "trainer_kwargs": trainer_kwargs or {},
        "episodes_per_role": episodes_per_role,
        "rollout_batch_size": rollout_batch_size,
        "exact_evaluation": exact_evaluation,
    }
    (run_path / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=_json_default),
        encoding="utf-8",
    )

    target_dense = (
        compile_policy_to_dense(
            target_policy,
            batch_size=exact_compile_batch_size,
        )
        if exact_evaluation
        else None
    )
    training_records: list[Dict[str, object]] = []
    evaluation_records: list[Dict[str, object]] = []
    measured_training_s = 0.0
    next_eval_s = (
        60.0 * float(evaluate_every_minutes)
        if evaluate_every_minutes
        else float("inf")
    )
    next_eval_iteration = (
        int(evaluate_every_iterations)
        if evaluate_every_iterations
        else 2**63 - 1
    )
    iteration_limit = int(iterations) if iterations is not None else 2**63 - 1
    seconds_limit = 60.0 * float(minutes) if minutes is not None else float("inf")

    def evaluate() -> Dict[str, object]:
        policy = trainer.policy()
        result: Dict[str, object] = {
            "iteration": int(trainer.iteration),
            "measured_training_s": measured_training_s,
            **evaluate_approximate_br(
                trainer,
                episodes_per_role=eval_episodes_per_role,
                rollout_batch_size=rollout_batch_size,
            ),
        }
        if target_dense is not None:
            responder_dense = compile_policy_to_dense(
                policy,
                batch_size=exact_compile_batch_size,
            )
            p_first, p_second = evaluate_dense_response(
                spec,
                opponent=target_dense,
                responder=responder_dense,
            )
            result.update(
                {
                    "exact_p_first": float(p_first),
                    "exact_p_second": float(p_second),
                    "exact_exploitability": float(p_first + p_second - 1.0),
                }
            )
        evaluation_records.append(result)
        _append_jsonl(run_path / "evaluations.jsonl", result)
        return result

    while (
        trainer.iteration < iteration_limit
        and measured_training_s < seconds_limit
    ):
        start = time.perf_counter()
        record = trainer.run_iteration(
            episodes_per_role=episodes_per_role,
            rollout_batch_size=rollout_batch_size,
        )
        measured_training_s += time.perf_counter() - start
        record["measured_training_s"] = measured_training_s
        training_records.append(record)
        _append_jsonl(run_path / "training.jsonl", record)

        due = (
            measured_training_s >= next_eval_s
            or trainer.iteration >= next_eval_iteration
        )
        if due:
            result = evaluate()
            if evaluate_every_minutes:
                period = 60.0 * float(evaluate_every_minutes)
                while next_eval_s <= measured_training_s:
                    next_eval_s += period
            if evaluate_every_iterations:
                period = int(evaluate_every_iterations)
                while next_eval_iteration <= trainer.iteration:
                    next_eval_iteration += period
            if debug:
                value = result.get(
                    "exact_exploitability",
                    result["exploitability_lower_bound"],
                )
                print(
                    f"[{method}] iter={trainer.iteration} "
                    f"train={measured_training_s / 60.0:.2f}m "
                    f"value={float(value):.6f}"
                )
        elif debug:
            print(
                f"[{method}] iter={trainer.iteration} "
                f"train={measured_training_s / 60.0:.2f}m"
            )

    if (
        not evaluation_records
        or evaluation_records[-1]["iteration"] != trainer.iteration
    ):
        evaluate()

    policy = trainer.policy()
    save_policy(policy, str(run_path / "policy"))
    summary = {
        **manifest,
        "iterations_completed": int(trainer.iteration),
        "measured_training_s": measured_training_s,
        "final_evaluation": evaluation_records[-1],
    }
    (run_path / "metrics.json").write_text(
        json.dumps(summary, indent=2, default=_json_default),
        encoding="utf-8",
    )
    return BestResponseRunResult(
        run_dir=run_path,
        policy=policy,
        trainer=trainer,
        training_records=training_records,
        evaluation_records=evaluation_records,
        measured_training_s=measured_training_s,
    )


@dataclass(frozen=True)
class BestResponseConfig:
    method: str
    seeds: Sequence[int] = (7,)
    minutes: float | None = None
    iterations: int | None = None
    trainer_kwargs: Dict[str, object] | None = None
    episodes_per_role: int = 4096
    rollout_batch_size: int = 4096
    eval_episodes_per_role: int = 200_000
    exact_evaluation: bool = False


class BestResponseSuiteEvaluator:
    """Blocking evaluator that runs responder configurations serially."""

    def __init__(self, configs: Sequence[BestResponseConfig]) -> None:
        self.configs = tuple(configs)

    def __call__(
        self,
        policy,
        context: Dict[str, object],
        output_dir: Path,
    ) -> Dict[str, object]:
        runs = []
        best_first = float("-inf")
        best_second = float("-inf")
        bound_first = float("-inf")
        bound_second = float("-inf")

        for config in self.configs:
            for seed in config.seeds:
                kwargs = dict(config.trainer_kwargs or {})
                kwargs["seed"] = int(seed)
                result = run_best_responder(
                    policy,
                    method=config.method,
                    minutes=config.minutes,
                    iterations=config.iterations,
                    trainer_kwargs=kwargs,
                    episodes_per_role=config.episodes_per_role,
                    rollout_batch_size=config.rollout_batch_size,
                    eval_episodes_per_role=config.eval_episodes_per_role,
                    exact_evaluation=config.exact_evaluation,
                    run_dir=output_dir / config.method / f"seed_{seed}",
                )
                final = dict(result.evaluation_records[-1])
                best_p_first = max(
                    float(record["p_first"])
                    for record in result.evaluation_records
                )
                best_p_second = max(
                    float(record["p_second"])
                    for record in result.evaluation_records
                )
                best_p_first_lcb = max(
                    float(record["p_first_lcb"])
                    for record in result.evaluation_records
                )
                best_p_second_lcb = max(
                    float(record["p_second_lcb"])
                    for record in result.evaluation_records
                )
                runs.append(
                    {
                        "method": config.method,
                        "seed": int(seed),
                        "run_dir": str(result.run_dir),
                        "best_p_first": best_p_first,
                        "best_p_second": best_p_second,
                        "best_p_first_lcb": best_p_first_lcb,
                        "best_p_second_lcb": best_p_second_lcb,
                        **final,
                    }
                )
                best_first = max(best_first, best_p_first)
                best_second = max(best_second, best_p_second)
                bound_first = max(bound_first, best_p_first_lcb)
                bound_second = max(bound_second, best_p_second_lcb)

        return {
            "runs": runs,
            "p_first_best": best_first,
            "p_second_best": best_second,
            "exploitability_estimate": best_first + best_second - 1.0,
            "p_first_lower_bound": bound_first,
            "p_second_lower_bound": bound_second,
            "exploitability_lower_bound": bound_first + bound_second - 1.0,
        }
