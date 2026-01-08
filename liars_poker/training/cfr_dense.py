from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from liars_poker.core import GameSpec, ARTIFACTS_ROOT
from liars_poker.serialization import save_policy, load_policy
from liars_poker.policies.tabular_dense import DenseTabularPolicy
from liars_poker.algo.cfr_exact_dense import CFRExactDense
from liars_poker.algo.br_exact_dense_to_dense import best_response_dense


def _spec_from_json(spec_json: str) -> GameSpec:
    data = json.loads(spec_json)
    return GameSpec(
        ranks=int(data["ranks"]),
        suits=int(data["suits"]),
        hand_size=int(data["hand_size"]),
        claim_kinds=tuple(data["claim_kinds"]),
        suit_symmetry=bool(data["suit_symmetry"]),
    )


def _atomic_save_policy(policy: DenseTabularPolicy, policy_dir: Path) -> None:
    stamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S-%f")
    tmp_dir = policy_dir.parent / f"{policy_dir.name}_new_{stamp}"
    backup_dir = policy_dir.parent / f"{policy_dir.name}_old_{stamp}"
    save_policy(policy, tmp_dir)
    if policy_dir.exists():
        policy_dir.rename(backup_dir)
    tmp_dir.rename(policy_dir)


def save_cfr_run(
    run_id: str,
    *,
    policy: DenseTabularPolicy,
    cfr: CFRExactDense,
    logs: Dict[str, object],
    spec: GameSpec,
    iterations_done: int,
    eval_every: int,
    root: Path | None = None,
) -> None:
    root = root or Path(ARTIFACTS_ROOT)
    run_dir = root / "benchmark_runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    policy_dir = run_dir / "policy"
    _atomic_save_policy(policy, policy_dir)

    state_path = run_dir / "cfr_state.npz"
    np.savez_compressed(
        state_path,
        R0=cfr.R0,
        R1=cfr.R1,
        SS0=cfr.SS0,
        SS1=cfr.SS1,
        iterations=np.asarray([iterations_done], dtype=np.int64),
    )

    metrics = {
        "run_type": "cfr",
        "spec": spec.to_json(),
        "iterations": iterations_done,
        "eval_every": eval_every,
        "exploitability_series": logs.get("exploitability_series", []),
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def load_cfr_state(run_dir: str | Path) -> Tuple[CFRExactDense, GameSpec, Dict[str, object], int]:
    run_path = Path(run_dir)
    metrics_path = run_path / "metrics.json"
    metrics: Dict[str, object] = {}
    spec: GameSpec | None = None
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        spec_json = metrics.get("spec")
        if isinstance(spec_json, str):
            spec = _spec_from_json(spec_json)

    if spec is None:
        policy_dir = run_path / "policy"
        _, spec = load_policy(str(policy_dir))

    state_path = run_path / "cfr_state.npz"
    if not state_path.exists():
        raise FileNotFoundError(f"Missing CFR state file: {state_path}")

    data = np.load(state_path)
    cfr = CFRExactDense(spec)
    cfr.R0 = data["R0"]
    cfr.R1 = data["R1"]
    cfr.SS0 = data["SS0"]
    cfr.SS1 = data["SS1"]
    cfr._update_strategy()
    cfr._recompute_likelihoods()
    iterations_done = int(data["iterations"][0]) if "iterations" in data else int(metrics.get("iterations", 0))
    return cfr, spec, metrics, iterations_done


def cfr_dense_loop(
    spec: GameSpec,
    iterations: int,
    *,
    cfr: CFRExactDense | None = None,
    start_iter: int = 0,
    eval_every: int = 0,
    debug: bool = False,
) -> Tuple[DenseTabularPolicy, Dict[str, object], CFRExactDense]:
    if cfr is None:
        cfr = CFRExactDense(spec)

    logs: Dict[str, object] = {"exploitability_series": []}
    start = time.perf_counter()

    for i in range(iterations):
        cfr.iterate()
        iter_idx = start_iter + i + 1
        if eval_every and iter_idx % eval_every == 0:
            avg_policy = cfr.average_policy()
            _, meta = best_response_dense(spec, avg_policy, debug=False, store_state_values=False)
            p_first, p_second = meta["computer"].exploitability()
            logs["exploitability_series"].append(
                {
                    "iter": iter_idx,
                    "p_first": p_first,
                    "p_second": p_second,
                    "predicted_avg": 0.5 * (p_first + p_second),
                }
            )
            if debug:
                elapsed = time.perf_counter() - start
                print(
                    f"[cfr] iter={iter_idx} elapsed={elapsed:.2f}s "
                    f"exploitability_avg={(p_first + p_second) * 0.5:.6f}"
                )
        elapsed = time.perf_counter() - start
        print(
            f"[cfr] iter={iter_idx} elapsed={elapsed:.2f}s "
        )   
    return cfr.average_policy(), logs, cfr


def cfr_dense_resume(
    run_dir: str | Path,
    *,
    remaining_iterations: int,
    eval_every: int = 0,
    debug: bool = False,
) -> Tuple[DenseTabularPolicy, Dict[str, object], CFRExactDense, GameSpec, int]:
    cfr, spec, metrics, iter_done = load_cfr_state(run_dir)

    base_series = list(metrics.get("exploitability_series", []) or [])
    avg_policy, new_logs, cfr = cfr_dense_loop(
        spec=spec,
        iterations=remaining_iterations,
        cfr=cfr,
        start_iter=iter_done,
        eval_every=eval_every,
        debug=debug,
    )

    merged = {
        "exploitability_series": base_series + list(new_logs.get("exploitability_series", []) or []),
    }
    total_iters = iter_done + remaining_iterations
    return avg_policy, merged, cfr, spec, total_iters
