from __future__ import annotations

import time
from typing import Dict, Tuple

from liars_poker.algo.neural_fsp import NeuralFSPTrainer
from liars_poker.core import GameSpec
from liars_poker.policies.action_conditioned import ActionConditionedPolicy


def neural_fsp_loop(
    spec: GameSpec,
    iterations: int,
    *,
    trainer: NeuralFSPTrainer | None = None,
    trainer_kwargs: Dict[str, object] | None = None,
    br_iterations: int = 100,
    br_episodes_per_role: int = 512,
    br_rollout_batch_size: int = 256,
    strategy_episodes_per_role: int = 512,
    strategy_collection_batch_size: int = 256,
    strategy_train_steps: int | None = None,
    debug: bool = False,
) -> Tuple[ActionConditionedPolicy, Dict[str, object], NeuralFSPTrainer]:
    if trainer is None:
        trainer = NeuralFSPTrainer(spec, **(trainer_kwargs or {}))

    records = []
    start = time.perf_counter()
    for _ in range(int(iterations)):
        record = trainer.run_iteration(
            br_iterations=br_iterations,
            br_episodes_per_role=br_episodes_per_role,
            br_rollout_batch_size=br_rollout_batch_size,
            strategy_episodes_per_role=strategy_episodes_per_role,
            strategy_collection_batch_size=strategy_collection_batch_size,
            strategy_train_steps=strategy_train_steps,
        )
        record["elapsed_s"] = time.perf_counter() - start
        records.append(record)
        if debug:
            print(
                f"[neural-fsp] iter={record['iter']} "
                f"elapsed={record['elapsed_s']:.2f}s "
                f"br={record['br_s']:.2f}s "
                f"average_collect={record['strategy_collect_s']:.2f}s "
                f"average_fit={record['strategy_fit_s']:.2f}s"
            )

    return trainer.average_policy(), {"iterations": records}, trainer
