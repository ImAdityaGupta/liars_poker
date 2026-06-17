from __future__ import annotations

from concurrent.futures import Future, ProcessPoolExecutor
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List

from liars_poker.algo.br_exact_dense_to_dense import best_response_dense
from liars_poker.policies.neural import compile_neural_to_dense
from liars_poker.serialization import load_policy, save_policy


def evaluate_saved_neural_policy(
    policy_dir: str | Path,
    *,
    iteration: int,
    label: str = "learned_average",
) -> Dict[str, float | int | str]:
    policy, spec = load_policy(str(policy_dir))
    dense = compile_neural_to_dense(policy)
    _, meta = best_response_dense(
        spec,
        dense,
        debug=False,
        store_state_values=False,
    )
    p_first, p_second = meta["computer"].exploitability()
    predicted_avg = 0.5 * (p_first + p_second)
    return {
        "iter": int(iteration),
        "label": label,
        "p_first": float(p_first),
        "p_second": float(p_second),
        "predicted_avg": float(predicted_avg),
        "exploitability": float(2.0 * predicted_avg - 1.0),
        "policy_dir": str(policy_dir),
    }


class AsyncDeepCFREvaluator:
    """Run exact learned-policy evaluations without blocking GPU training."""

    def __init__(self, run_dir: str | Path, *, max_workers: int = 1) -> None:
        self.run_dir = Path(run_dir)
        self.snapshot_root = self.run_dir / "eval_snapshots"
        self.snapshot_root.mkdir(parents=True, exist_ok=True)
        self.executor = ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=mp.get_context("spawn"),
        )
        self.pending: List[Future] = []

    @property
    def pending_count(self) -> int:
        return len(self.pending)

    def submit(self, iteration: int, policy, *, label: str = "learned_average") -> Path:
        policy_dir = self.snapshot_root / f"iter_{int(iteration):08d}_{label}"
        save_policy(policy, str(policy_dir))
        self.pending.append(
            self.executor.submit(
                evaluate_saved_neural_policy,
                policy_dir,
                iteration=int(iteration),
                label=label,
            )
        )
        return policy_dir

    def collect_ready(self) -> List[Dict[str, float | int | str]]:
        ready = []
        still_pending = []
        for future in self.pending:
            if future.done():
                ready.append(future.result())
            else:
                still_pending.append(future)
        self.pending = still_pending
        return ready

    def wait(self) -> List[Dict[str, float | int | str]]:
        results = [future.result() for future in self.pending]
        self.pending.clear()
        return results

    def close(self, *, wait: bool = True) -> None:
        self.executor.shutdown(wait=wait)

    def __enter__(self) -> "AsyncDeepCFREvaluator":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close(wait=True)
