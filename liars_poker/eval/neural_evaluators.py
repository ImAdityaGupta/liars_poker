from __future__ import annotations

from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass, field
import json
import math
import multiprocessing as mp
from pathlib import Path
import time
from typing import Callable, Dict, List

import numpy as np

from liars_poker.algo.br_exact_dense_to_dense import best_response_dense
from liars_poker.policies.action_conditioned import (
    ActionConditionedPolicy,
    ActionConditionedQPolicy,
    compile_action_conditioned_q_to_dense,
    compile_action_conditioned_to_dense,
)
from liars_poker.policies.neural import NeuralPolicy, compile_neural_to_dense
from liars_poker.policies.neural_q import NeuralQPolicy, compile_neural_q_to_dense
from liars_poker.policies.tabular_dense import DenseTabularPolicy
from liars_poker.serialization import load_policy, save_policy


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


def compile_policy_to_dense(policy, *, batch_size: int = 65_536) -> DenseTabularPolicy:
    """Compile any currently supported playable policy to the dense format."""

    if isinstance(policy, DenseTabularPolicy):
        return policy
    if isinstance(policy, ActionConditionedPolicy):
        return compile_action_conditioned_to_dense(policy, batch_size=batch_size)
    if isinstance(policy, ActionConditionedQPolicy):
        return compile_action_conditioned_q_to_dense(policy, batch_size=batch_size)
    if isinstance(policy, NeuralQPolicy):
        return compile_neural_q_to_dense(policy, batch_size=batch_size)
    if isinstance(policy, NeuralPolicy):
        return compile_neural_to_dense(policy, batch_size=batch_size)
    raise TypeError(f"Cannot compile policy type {type(policy).__name__} to dense.")


def evaluate_saved_policy_exact(
    policy_dir: str | Path,
    *,
    context: Dict[str, object],
    compile_batch_size: int = 65_536,
) -> Dict[str, object]:
    total_start = time.perf_counter()

    start = time.perf_counter()
    policy, spec = load_policy(str(policy_dir))
    load_policy_s = time.perf_counter() - start

    start = time.perf_counter()
    dense = compile_policy_to_dense(policy, batch_size=compile_batch_size)
    dense_compile_s = time.perf_counter() - start

    start = time.perf_counter()
    _, meta = best_response_dense(
        spec,
        dense,
        debug=False,
        store_state_values=False,
    )
    exact_br_s = time.perf_counter() - start
    p_first, p_second = meta["computer"].exploitability()
    predicted_avg = 0.5 * (p_first + p_second)

    result = dict(context)
    result.update(
        {
            "p_first": float(p_first),
            "p_second": float(p_second),
            "predicted_avg": float(predicted_avg),
            "exploitability": float(2.0 * predicted_avg - 1.0),
            "policy_dir": str(policy_dir),
            "load_policy_s": load_policy_s,
            "dense_compile_s": dense_compile_s,
            "exact_br_s": exact_br_s,
            "evaluation_s": time.perf_counter() - total_start,
        }
    )

    result_path = Path(policy_dir) / "result.json"
    tmp_path = result_path.with_suffix(".json.tmp")
    tmp_path.write_text(
        json.dumps(result, indent=2, default=_json_default),
        encoding="utf-8",
    )
    tmp_path.replace(result_path)
    return result


class AsyncExactExploitabilityEvaluator:
    """Queue exact exploitability evaluations in separate CPU processes."""

    def __init__(
        self,
        *,
        max_workers: int = 1,
        compile_batch_size: int = 65_536,
    ) -> None:
        self.max_workers = int(max_workers)
        self.compile_batch_size = int(compile_batch_size)
        self.root: Path | None = None
        self.executor: ProcessPoolExecutor | None = None
        self.pending: List[Future] = []
        self._submitted = 0

    @property
    def pending_count(self) -> int:
        return len(self.pending)

    def start(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.executor = ProcessPoolExecutor(
            max_workers=self.max_workers,
            mp_context=mp.get_context("spawn"),
        )

    def submit(self, policy, context: Dict[str, object]) -> List[Dict[str, object]]:
        if self.root is None or self.executor is None:
            raise RuntimeError("Evaluator has not been started.")
        snapshot_id = self._submitted
        self._submitted += 1
        iteration = int(context.get("iteration", 0))
        policy_dir = self.root / "snapshots" / (
            f"iter_{iteration:08d}_{snapshot_id:04d}"
        )
        save_policy(policy, str(policy_dir))
        self.pending.append(
            self.executor.submit(
                evaluate_saved_policy_exact,
                policy_dir,
                context=dict(context),
                compile_batch_size=self.compile_batch_size,
            )
        )
        return []

    def collect_ready(self) -> List[Dict[str, object]]:
        ready: List[Dict[str, object]] = []
        pending: List[Future] = []
        for future in self.pending:
            if future.done():
                ready.append(future.result())
            else:
                pending.append(future)
        self.pending = pending
        return ready

    def close(self, *, wait: bool = True) -> List[Dict[str, object]]:
        results: List[Dict[str, object]] = []
        if wait:
            results = [future.result() for future in self.pending]
            self.pending.clear()
        if self.executor is not None:
            self.executor.shutdown(wait=wait)
            self.executor = None
        return results


class BlockingFunctionEvaluator:
    """Run a user-supplied evaluation function before training resumes."""

    def __init__(
        self,
        function: Callable[[object, Dict[str, object], Path], Dict[str, object]],
    ) -> None:
        self.function = function
        self.root: Path | None = None
        self._submitted = 0

    @property
    def pending_count(self) -> int:
        return 0

    def start(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def submit(self, policy, context: Dict[str, object]) -> List[Dict[str, object]]:
        if self.root is None:
            raise RuntimeError("Evaluator has not been started.")
        iteration = int(context.get("iteration", 0))
        output_dir = self.root / f"iter_{iteration:08d}_{self._submitted:04d}"
        self._submitted += 1
        output_dir.mkdir(parents=True, exist_ok=True)
        save_policy(policy, str(output_dir / "target_policy"))
        start = time.perf_counter()
        result = dict(context)
        result.update(self.function(policy, dict(context), output_dir))
        result.setdefault("evaluation_s", time.perf_counter() - start)
        (output_dir / "result.json").write_text(
            json.dumps(result, indent=2, default=_json_default),
            encoding="utf-8",
        )
        return [result]

    def collect_ready(self) -> List[Dict[str, object]]:
        return []

    def close(self, *, wait: bool = True) -> List[Dict[str, object]]:
        return []


@dataclass
class ScheduledEvaluation:
    """Schedule an evaluator by measured training time and/or iteration."""

    name: str
    evaluator: object
    every_minutes: float | None = None
    every_iterations: int | None = None
    run_at_end: bool = True
    _next_training_s: float = field(init=False, default=math.inf)
    _next_iteration: int = field(init=False, default=2**63 - 1)
    _last_submitted_iteration: int | None = field(init=False, default=None)

    def start(
        self,
        run_dir: str | Path,
        *,
        measured_training_s: float,
        iteration: int,
    ) -> None:
        root = Path(run_dir) / "evaluations" / self.name
        self.evaluator.start(root)
        if self.every_minutes is not None and self.every_minutes > 0:
            period = 60.0 * float(self.every_minutes)
            self._next_training_s = (
                math.floor(measured_training_s / period) + 1
            ) * period
        if self.every_iterations is not None and self.every_iterations > 0:
            period = int(self.every_iterations)
            self._next_iteration = (iteration // period + 1) * period

    def due(self, *, measured_training_s: float, iteration: int) -> bool:
        return (
            measured_training_s >= self._next_training_s
            or iteration >= self._next_iteration
        )

    def mark_submitted(
        self,
        *,
        measured_training_s: float,
        iteration: int,
    ) -> None:
        self._last_submitted_iteration = int(iteration)
        if self.every_minutes is not None and self.every_minutes > 0:
            period = 60.0 * float(self.every_minutes)
            while self._next_training_s <= measured_training_s:
                self._next_training_s += period
        if self.every_iterations is not None and self.every_iterations > 0:
            period = int(self.every_iterations)
            while self._next_iteration <= iteration:
                self._next_iteration += period

    def should_run_final(self, iteration: int) -> bool:
        return self.run_at_end and self._last_submitted_iteration != int(iteration)

    def submit(
        self,
        policy,
        context: Dict[str, object],
    ) -> List[Dict[str, object]]:
        labelled_context = dict(context)
        labelled_context["evaluator"] = self.name
        results = self.evaluator.submit(policy, labelled_context)
        self.mark_submitted(
            measured_training_s=float(context["measured_training_s"]),
            iteration=int(context["iteration"]),
        )
        return results

    def collect_ready(self) -> List[Dict[str, object]]:
        return self.evaluator.collect_ready()

    def close(self, *, wait: bool = True) -> List[Dict[str, object]]:
        return self.evaluator.close(wait=wait)
