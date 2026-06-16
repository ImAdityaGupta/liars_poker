from __future__ import annotations

import time
from typing import Dict, Tuple

from liars_poker.algo.br_exact_dense_to_dense import best_response_dense
from liars_poker.algo.cfr_external_sampling_dense import CFRExternalSamplingDense
from liars_poker.core import GameSpec
from liars_poker.policies.tabular_dense import DenseTabularPolicy


def _exact_evaluation(spec: GameSpec, cfr: CFRExternalSamplingDense) -> Dict[str, float]:
    average = cfr.average_policy()
    _, average_meta = best_response_dense(spec, average, debug=False, store_state_values=False)
    p_first, p_second = average_meta["computer"].exploitability()

    current = cfr.current_policy()
    _, current_meta = best_response_dense(spec, current, debug=False, store_state_values=False)
    current_p_first, current_p_second = current_meta["computer"].exploitability()
    return {
        "p_first": p_first,
        "p_second": p_second,
        "predicted_avg": 0.5 * (p_first + p_second),
        "current_p_first": current_p_first,
        "current_p_second": current_p_second,
        "current_predicted_avg": 0.5 * (current_p_first + current_p_second),
    }


def external_sampling_cfr_loop(
    spec: GameSpec,
    iterations: int,
    *,
    cfr: CFRExternalSamplingDense | None = None,
    cfr_kwargs: Dict[str, object] | None = None,
    traversals_per_player: int = 100,
    eval_every: int = 0,
    debug: bool = False,
) -> Tuple[DenseTabularPolicy, Dict[str, object], CFRExternalSamplingDense]:
    if cfr is None:
        cfr = CFRExternalSamplingDense(spec, **(cfr_kwargs or {}))

    logs: Dict[str, object] = {"training_series": [], "exploitability_series": []}
    start = time.perf_counter()
    for _ in range(iterations):
        record = cfr.run_iteration(traversals_per_player=traversals_per_player)
        record["elapsed_s"] = time.perf_counter() - start
        logs["training_series"].append(record)
        if eval_every and cfr.iteration % eval_every == 0:
            logs["exploitability_series"].append({"iter": cfr.iteration, **_exact_evaluation(spec, cfr)})
        if debug:
            print(
                f"[external-cfr] iter={cfr.iteration} elapsed={record['elapsed_s']:.2f}s "
                f"traverse={record['traversal_s']:.2f}s"
            )
    return cfr.average_policy(), logs, cfr


def external_sampling_cfr_timed_loop(
    spec: GameSpec,
    training_seconds: float,
    *,
    cfr: CFRExternalSamplingDense | None = None,
    cfr_kwargs: Dict[str, object] | None = None,
    traversals_per_player: int = 100,
    eval_every: int = 0,
    debug: bool = False,
) -> Tuple[DenseTabularPolicy, Dict[str, object], CFRExternalSamplingDense]:
    if cfr is None:
        cfr = CFRExternalSamplingDense(spec, **(cfr_kwargs or {}))

    logs: Dict[str, object] = {"training_series": [], "exploitability_series": []}
    training_elapsed = 0.0
    while training_elapsed < training_seconds:
        record = cfr.run_iteration(traversals_per_player=traversals_per_player)
        training_elapsed += float(record["traversal_s"])
        record["elapsed_s"] = training_elapsed
        logs["training_series"].append(record)
        if eval_every and cfr.iteration % eval_every == 0:
            logs["exploitability_series"].append({"iter": cfr.iteration, **_exact_evaluation(spec, cfr)})
        if debug:
            print(
                f"[external-cfr] iter={cfr.iteration} "
                f"training_budget={training_elapsed:.2f}/{training_seconds:.2f}s"
            )

    if not logs["exploitability_series"] or logs["exploitability_series"][-1]["iter"] != cfr.iteration:
        logs["exploitability_series"].append({"iter": cfr.iteration, **_exact_evaluation(spec, cfr)})
    return cfr.average_policy(), logs, cfr
