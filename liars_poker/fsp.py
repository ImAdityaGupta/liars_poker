from __future__ import annotations

import json
import os
import time
from typing import Dict

from .core import ARTIFACTS_ROOT, GameSpec, env_hash
from .logging import StrategyManifest, save_json, write_strategy_manifest
from .policy import RandomPolicy


def train_fsp(
    spec: GameSpec,
    eta_schedule: str = "harmonic",
    mix: str = "commit_once",
    max_iters: int = 1,
    save_root: str = ARTIFACTS_ROOT,
    seed: int = 0,
) -> Dict[str, str]:
    """Minimal FSP scaffold: produce an average policy id and manifests.

    This does not implement real BR or RL; it creates placeholder policies and
    logs a StrategyManifest for lineage.
    """

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_id = f"run_{timestamp}_{seed}"
    runs_root = os.path.join(save_root, "runs")
    run_dir = os.path.join(runs_root, run_id)
    policies_dir = os.path.join(run_dir, "policies")
    manifests_dir = os.path.join(run_dir, "manifests")
    os.makedirs(policies_dir, exist_ok=True)
    os.makedirs(manifests_dir, exist_ok=True)

    env_h = env_hash(spec)
    spec_payload = {
        "spec": json.loads(spec.to_json()),
        "env_hash": env_h,
        "seed": seed,
    }
    save_json(os.path.join(run_dir, "env_spec.json"), spec_payload)

    def _save_policy_stub(policy_id: str) -> None:
        policy_path = os.path.join(policies_dir, f"{policy_id}.json")
        policy_stub = RandomPolicy().to_json()
        save_json(policy_path, policy_stub)

    def _manifest_path(policy_id: str) -> str:
        return os.path.join(manifests_dir, f"{policy_id}.json")

    a0_id = "A0"
    _save_policy_stub(a0_id)
    a0_manifest = StrategyManifest(
        id=a0_id,
        role="average",
        kind="policy/random",
        env_hash=env_h,
        parents=[],
        mixing=None,
        seeds={"init": seed},
        train={"algo": "FSP", "eta_schedule": eta_schedule, "mix": mix, "iters": max_iters},
        artifacts={"policy": f"policies/{a0_id}.json"},
        code_sha="unknown",
    )
    write_strategy_manifest(_manifest_path(a0_id), a0_manifest)

    avg_id = a0_id
    if max_iters >= 1:
        a1_id = "A1"
        _save_policy_stub(a1_id)
        eta_value = 1.0
        if eta_schedule == "harmonic":
            eta_value = 1.0 / (1 + 1)
        a1_manifest = StrategyManifest(
            id=a1_id,
            role="average",
            kind="policy/random",
            env_hash=env_h,
            parents=[{"id": a0_id, "role": "avg", "weight": 1.0}],
            mixing={"impl": mix, "schedule": eta_schedule, "eta_k": eta_value},
            seeds={"iter0": seed + 1},
            train={"algo": "FSP", "eta_schedule": eta_schedule, "mix": mix, "iters": max_iters},
            artifacts={"policy": f"policies/{a1_id}.json"},
            code_sha="unknown",
        )
        write_strategy_manifest(_manifest_path(a1_id), a1_manifest)
        avg_id = a1_id

    return {"run_dir": run_dir, "average_policy_id": avg_id}
