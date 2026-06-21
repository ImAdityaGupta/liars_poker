from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import time
from typing import Callable, Dict, Sequence

import numpy as np

from liars_poker.algo.deep_cfr import DeepCFRTrainer
from liars_poker.algo.deep_cfr_plus import DeepCFRPlusTrainer
from liars_poker.algo.neural_fsp import NeuralFSPTrainer
from liars_poker.core import ARTIFACTS_ROOT, GameSpec
from liars_poker.eval.neural_evaluators import ScheduledEvaluation
from liars_poker.serialization import save_policy


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (tuple, set)):
        return list(value)
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _append_jsonl(path: Path, record: Dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=_json_default) + "\n")


def _atomic_checkpoint(trainer, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".new")
    trainer.save_checkpoint(temporary)
    os.replace(temporary, path)


def _default_run_dir(method: str, spec: GameSpec) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return (
        Path(ARTIFACTS_ROOT)
        / "neural_runs"
        / method
        / f"{spec.to_short_str()}___{stamp}"
    )


@dataclass
class NeuralRunResult:
    run_dir: Path
    policy: object
    trainer: object
    training_records: list[Dict[str, object]]
    evaluation_records: list[Dict[str, object]]
    measured_training_s: float
    wall_s: float


def _run_neural(
    *,
    method: str,
    spec: GameSpec,
    trainer_cls,
    step: Callable[[object], Dict[str, object]],
    run_kwargs: Dict[str, object],
    minutes: float | None,
    iterations: int | None,
    trainer_kwargs: Dict[str, object] | None,
    evaluations: Sequence[ScheduledEvaluation],
    checkpoint_every_minutes: float | None,
    checkpoint_every_iterations: int | None,
    run_dir: str | Path | None,
    resume_from: str | Path | None,
    device: str | None,
    save_checkpoint: bool,
    wait_for_evaluations: bool,
    debug: bool,
) -> NeuralRunResult:
    if (minutes is None) == (iterations is None):
        raise ValueError("Provide exactly one of minutes or iterations.")

    if resume_from is not None:
        run_path = Path(resume_from)
        trainer = trainer_cls.load_checkpoint(
            run_path / "latest_checkpoint.pt",
            device=device or "cpu",
        )
        if trainer.spec != spec:
            raise ValueError("Resume checkpoint spec does not match the supplied spec.")
        manifest_path = run_path / "manifest.json"
        manifest = (
            json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest_path.exists()
            else {}
        )
        measured_training_s = float(manifest.get("measured_training_s", 0.0))
    else:
        run_path = Path(run_dir) if run_dir is not None else _default_run_dir(method, spec)
        trainer = trainer_cls(spec, **(trainer_kwargs or {}))
        manifest = {}
        measured_training_s = 0.0

    run_path.mkdir(parents=True, exist_ok=True)
    training_path = run_path / "training.jsonl"
    evaluations_path = run_path / "evaluations.jsonl"
    manifest_path = run_path / "manifest.json"

    session_start_training_s = measured_training_s
    session_target_training_s = (
        measured_training_s + 60.0 * float(minutes)
        if minutes is not None
        else math.inf
    )
    session_target_iteration = (
        int(trainer.iteration) + int(iterations)
        if iterations is not None
        else 2**63 - 1
    )
    next_checkpoint_s = (
        (
            math.floor(measured_training_s / (60.0 * checkpoint_every_minutes))
            + 1
        )
        * 60.0
        * checkpoint_every_minutes
        if checkpoint_every_minutes
        else math.inf
    )
    next_checkpoint_iteration = (
        (trainer.iteration // checkpoint_every_iterations + 1)
        * checkpoint_every_iterations
        if checkpoint_every_iterations
        else 2**63 - 1
    )

    for schedule in evaluations:
        schedule.start(
            run_path,
            measured_training_s=measured_training_s,
            iteration=trainer.iteration,
        )

    manifest.update(
        {
            "run_type": method,
            "spec": spec.to_json(),
            "trainer_kwargs": trainer_kwargs or manifest.get("trainer_kwargs", {}),
            "run_kwargs": run_kwargs,
            "created_or_resumed_at": datetime.now(timezone.utc).isoformat(),
        }
    )

    training_records: list[Dict[str, object]] = []
    evaluation_records: list[Dict[str, object]] = []
    wall_start = time.perf_counter()

    def write_manifest(status: str) -> None:
        manifest.update(
            {
                "status": status,
                "iterations": int(trainer.iteration),
                "measured_training_s": measured_training_s,
                "session_training_s": measured_training_s - session_start_training_s,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        manifest_path.write_text(
            json.dumps(manifest, indent=2, default=_json_default),
            encoding="utf-8",
        )

    def store_evaluations(records: Sequence[Dict[str, object]]) -> None:
        for record in records:
            evaluation_records.append(record)
            _append_jsonl(evaluations_path, record)

    write_manifest("running")
    try:
        while (
            measured_training_s < session_target_training_s
            and trainer.iteration < session_target_iteration
        ):
            start = time.perf_counter()
            record = step(trainer)
            iteration_s = time.perf_counter() - start
            measured_training_s += iteration_s
            record["iteration"] = int(trainer.iteration)
            record["iteration_s"] = iteration_s
            record["measured_training_s"] = measured_training_s
            record["session_training_s"] = (
                measured_training_s - session_start_training_s
            )
            training_records.append(record)
            _append_jsonl(training_path, record)

            context = {
                "method": method,
                "iteration": int(trainer.iteration),
                "measured_training_s": measured_training_s,
                "session_training_s": measured_training_s - session_start_training_s,
            }
            for schedule in evaluations:
                if schedule.due(
                    measured_training_s=measured_training_s,
                    iteration=trainer.iteration,
                ):
                    store_evaluations(
                        schedule.submit(trainer.average_policy(), context)
                    )
                store_evaluations(schedule.collect_ready())

            checkpoint_due = (
                measured_training_s >= next_checkpoint_s
                or trainer.iteration >= next_checkpoint_iteration
            )
            if save_checkpoint and checkpoint_due:
                _atomic_checkpoint(trainer, run_path / "latest_checkpoint.pt")
                if checkpoint_every_minutes:
                    period = 60.0 * float(checkpoint_every_minutes)
                    while next_checkpoint_s <= measured_training_s:
                        next_checkpoint_s += period
                if checkpoint_every_iterations:
                    period = int(checkpoint_every_iterations)
                    while next_checkpoint_iteration <= trainer.iteration:
                        next_checkpoint_iteration += period
                write_manifest("running")

            if debug:
                pending = sum(
                    int(getattr(schedule.evaluator, "pending_count", 0))
                    for schedule in evaluations
                )
                print(
                    f"[{method}] iter={trainer.iteration} "
                    f"train={measured_training_s / 60.0:.2f}m "
                    f"step={iteration_s:.2f}s pending_evals={pending}"
                )

        final_policy = trainer.average_policy()
        final_context = {
            "method": method,
            "iteration": int(trainer.iteration),
            "measured_training_s": measured_training_s,
            "session_training_s": measured_training_s - session_start_training_s,
            "final": True,
        }
        for schedule in evaluations:
            if schedule.should_run_final(trainer.iteration):
                store_evaluations(schedule.submit(final_policy, final_context))

        save_policy(final_policy, str(run_path / "policy"))
        if save_checkpoint:
            _atomic_checkpoint(trainer, run_path / "latest_checkpoint.pt")
        write_manifest("waiting_for_evaluations")

        for schedule in evaluations:
            store_evaluations(schedule.collect_ready())
            store_evaluations(schedule.close(wait=wait_for_evaluations))

        write_manifest("complete")
    except BaseException:
        write_manifest("failed")
        for schedule in evaluations:
            try:
                store_evaluations(schedule.close(wait=False))
            except BaseException:
                pass
        raise

    return NeuralRunResult(
        run_dir=run_path,
        policy=final_policy,
        trainer=trainer,
        training_records=training_records,
        evaluation_records=evaluation_records,
        measured_training_s=measured_training_s,
        wall_s=time.perf_counter() - wall_start,
    )


def run_deep_cfr(
    spec: GameSpec,
    *,
    minutes: float | None = None,
    iterations: int | None = None,
    traversals_per_player: int = 1024,
    trainer_kwargs: Dict[str, object] | None = None,
    evaluations: Sequence[ScheduledEvaluation] = (),
    checkpoint_every_minutes: float | None = None,
    checkpoint_every_iterations: int | None = None,
    run_dir: str | Path | None = None,
    resume_from: str | Path | None = None,
    device: str | None = None,
    save_checkpoint: bool = True,
    wait_for_evaluations: bool = True,
    debug: bool = False,
) -> NeuralRunResult:
    return _run_neural(
        method="deep_cfr",
        spec=spec,
        trainer_cls=DeepCFRTrainer,
        step=lambda trainer: trainer.run_iteration(
            traversals_per_player=traversals_per_player
        ),
        run_kwargs={"traversals_per_player": traversals_per_player},
        minutes=minutes,
        iterations=iterations,
        trainer_kwargs=trainer_kwargs,
        evaluations=evaluations,
        checkpoint_every_minutes=checkpoint_every_minutes,
        checkpoint_every_iterations=checkpoint_every_iterations,
        run_dir=run_dir,
        resume_from=resume_from,
        device=device,
        save_checkpoint=save_checkpoint,
        wait_for_evaluations=wait_for_evaluations,
        debug=debug,
    )


def run_neural_cfr_plus(
    spec: GameSpec,
    *,
    minutes: float | None = None,
    iterations: int | None = None,
    traversals_per_player: int = 1024,
    trainer_kwargs: Dict[str, object] | None = None,
    evaluations: Sequence[ScheduledEvaluation] = (),
    checkpoint_every_minutes: float | None = None,
    checkpoint_every_iterations: int | None = None,
    run_dir: str | Path | None = None,
    resume_from: str | Path | None = None,
    device: str | None = None,
    save_checkpoint: bool = True,
    wait_for_evaluations: bool = True,
    debug: bool = False,
) -> NeuralRunResult:
    return _run_neural(
        method="neural_cfr_plus",
        spec=spec,
        trainer_cls=DeepCFRPlusTrainer,
        step=lambda trainer: trainer.run_iteration(
            traversals_per_player=traversals_per_player
        ),
        run_kwargs={"traversals_per_player": traversals_per_player},
        minutes=minutes,
        iterations=iterations,
        trainer_kwargs=trainer_kwargs,
        evaluations=evaluations,
        checkpoint_every_minutes=checkpoint_every_minutes,
        checkpoint_every_iterations=checkpoint_every_iterations,
        run_dir=run_dir,
        resume_from=resume_from,
        device=device,
        save_checkpoint=save_checkpoint,
        wait_for_evaluations=wait_for_evaluations,
        debug=debug,
    )


def run_neural_fsp(
    spec: GameSpec,
    *,
    minutes: float | None = None,
    iterations: int | None = None,
    trainer_kwargs: Dict[str, object] | None = None,
    br_iterations: int = 100,
    br_episodes_per_role: int = 512,
    br_rollout_batch_size: int = 256,
    strategy_episodes_per_role: int = 512,
    strategy_collection_batch_size: int = 256,
    strategy_train_steps: int | None = None,
    evaluations: Sequence[ScheduledEvaluation] = (),
    checkpoint_every_minutes: float | None = None,
    checkpoint_every_iterations: int | None = None,
    run_dir: str | Path | None = None,
    resume_from: str | Path | None = None,
    device: str | None = None,
    save_checkpoint: bool = True,
    wait_for_evaluations: bool = True,
    debug: bool = False,
) -> NeuralRunResult:
    return _run_neural(
        method="neural_fsp",
        spec=spec,
        trainer_cls=NeuralFSPTrainer,
        step=lambda trainer: trainer.run_iteration(
            br_iterations=br_iterations,
            br_episodes_per_role=br_episodes_per_role,
            br_rollout_batch_size=br_rollout_batch_size,
            strategy_episodes_per_role=strategy_episodes_per_role,
            strategy_collection_batch_size=strategy_collection_batch_size,
            strategy_train_steps=strategy_train_steps,
        ),
        run_kwargs={
            "br_iterations": br_iterations,
            "br_episodes_per_role": br_episodes_per_role,
            "br_rollout_batch_size": br_rollout_batch_size,
            "strategy_episodes_per_role": strategy_episodes_per_role,
            "strategy_collection_batch_size": strategy_collection_batch_size,
            "strategy_train_steps": strategy_train_steps,
        },
        minutes=minutes,
        iterations=iterations,
        trainer_kwargs=trainer_kwargs,
        evaluations=evaluations,
        checkpoint_every_minutes=checkpoint_every_minutes,
        checkpoint_every_iterations=checkpoint_every_iterations,
        run_dir=run_dir,
        resume_from=resume_from,
        device=device,
        save_checkpoint=save_checkpoint,
        wait_for_evaluations=wait_for_evaluations,
        debug=debug,
    )
