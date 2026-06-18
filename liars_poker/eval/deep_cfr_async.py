from __future__ import annotations

from concurrent.futures import Future, ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path
import time
from typing import Dict, List

from liars_poker.algo.br_exact_dense_to_dense import best_response_dense
from liars_poker.policies.neural import compile_neural_to_dense
from liars_poker.serialization import load_policy, save_policy


def evaluate_saved_neural_policy(
    policy_dir: str | Path,
    *,
    iteration: int,
    label: str = "learned_average",
    metadata: Dict[str, object] | None = None,
    compile_batch_size: int = 16_384,
) -> Dict[str, object]:
    total_start = time.perf_counter()
    start = time.perf_counter()
    policy, spec = load_policy(str(policy_dir))
    load_policy_s = time.perf_counter() - start

    start = time.perf_counter()
    dense = compile_neural_to_dense(policy, batch_size=compile_batch_size)
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
    result: Dict[str, object] = dict(metadata or {})
    result.update({
        "iter": int(iteration),
        "label": label,
        "p_first": float(p_first),
        "p_second": float(p_second),
        "predicted_avg": float(predicted_avg),
        "exploitability": float(2.0 * predicted_avg - 1.0),
        "policy_dir": str(policy_dir),
        "load_policy_s": load_policy_s,
        "dense_compile_s": dense_compile_s,
        "exact_br_s": exact_br_s,
        "evaluation_s": time.perf_counter() - total_start,
    })
    result_path = Path(policy_dir) / "result.json"
    tmp_path = result_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    tmp_path.replace(result_path)
    return result


class AsyncDeepCFREvaluator:
    """Run exact learned-policy evaluations without blocking GPU training."""

    def __init__(
        self,
        run_dir: str | Path,
        *,
        max_workers: int = 1,
        compile_batch_size: int = 16_384,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.snapshot_root = self.run_dir / "eval_snapshots"
        self.snapshot_root.mkdir(parents=True, exist_ok=True)
        self.compile_batch_size = int(compile_batch_size)
        self.executor = ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=mp.get_context("spawn"),
        )
        self.pending: List[Future] = []
        self._submitted = 0

    @property
    def pending_count(self) -> int:
        return len(self.pending)

    def submit(
        self,
        iteration: int,
        policy,
        *,
        label: str = "learned_average",
        metadata: Dict[str, object] | None = None,
    ) -> Path:
        snapshot_id = self._submitted
        self._submitted += 1
        policy_dir = self.snapshot_root / (
            f"iter_{int(iteration):08d}_{snapshot_id:04d}_{label}"
        )
        save_policy(policy, str(policy_dir))
        self.pending.append(
            self.executor.submit(
                evaluate_saved_neural_policy,
                policy_dir,
                iteration=int(iteration),
                label=label,
                metadata=metadata,
                compile_batch_size=self.compile_batch_size,
            )
        )
        return policy_dir

    def collect_ready(self) -> List[Dict[str, object]]:
        ready = []
        still_pending = []
        for future in self.pending:
            if future.done():
                ready.append(future.result())
            else:
                still_pending.append(future)
        self.pending = still_pending
        return ready

    def wait(self) -> List[Dict[str, object]]:
        results = [future.result() for future in self.pending]
        self.pending.clear()
        return results

    def close(self, *, wait: bool = True) -> None:
        self.executor.shutdown(wait=wait)

    def __enter__(self) -> "AsyncDeepCFREvaluator":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close(wait=True)
