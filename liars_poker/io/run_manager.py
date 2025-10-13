from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional

from liars_poker.core import GameSpec, env_hash
from liars_poker.env import rules_for_spec
from liars_poker.io.jsonio import load_json, load_jsonz, save_json, save_jsonz
from liars_poker.io.manifest import StrategyManifest, write_strategy_manifest, read_strategy_manifest
from liars_poker.io.policy_io import policy_from_json, policy_to_json
from liars_poker.policies.base import Policy


class RunManager:
    """Persist policies, manifests, events, and indexes under a run directory."""

    def __init__(self, spec: GameSpec, save_root: str, seed: int, code_sha: str | None = None) -> None:
        self.spec = spec
        self.seed = seed
        self.code_sha = code_sha
        self.save_root = os.path.abspath(save_root)
        self.env_hash = env_hash(spec)

        runs_root = os.path.join(self.save_root, "runs")
        os.makedirs(runs_root, exist_ok=True)

        self.run_id = self._generate_run_id(seed)
        self.run_dir = os.path.join(runs_root, self.run_id)
        os.makedirs(self.run_dir, exist_ok=True)

        self.policies_dir = os.path.join(self.run_dir, "policies")
        self.manifests_dir = os.path.join(self.run_dir, "manifests")
        self.trainer_dir = os.path.join(self.run_dir, "trainer")
        self.index_dir = os.path.join(self.run_dir, "indexes")

        for path in (self.policies_dir, self.manifests_dir, self.trainer_dir, self.index_dir):
            os.makedirs(path, exist_ok=True)

        self._index_path = os.path.join(self.index_dir, "index.json")
        self._latest_avg_path = os.path.join(self.index_dir, "latest_avg.txt")
        self._timeline_path = os.path.join(self.trainer_dir, "timeline.jsonl")

        self._index: Dict[str, Dict[str, str]] = {}
        self._avg_counter = 0
        self._br_counter = 0
        self._current_avg: Optional[str] = None

        self._write_run_metadata()
        self._hydrate_from_disk()

    def log_policy(
        self,
        policy: Policy,
        *,
        role: str,
        parents: List[Dict],
        mixing: Dict | None = None,
        notes: str | None = None,
        seed: int | None = None,
        train: Dict | None = None,
    ) -> str:
        policy_id = self._next_policy_id(role)
        payload = policy_to_json(policy)
        policy_path = os.path.join(self.policies_dir, f"{policy_id}.json.gz")
        save_jsonz(policy_path, payload)

        manifest = StrategyManifest(
            id=policy_id,
            role=role,
            kind=self._policy_kind(policy),
            env_hash=self.env_hash,
            parents=list(parents),
            mixing=mixing,
            seeds={"seed": seed} if seed is not None else {},
            train=train or {},
            artifacts={"policy": f"policies/{policy_id}.json.gz"},
            code_sha=self.code_sha or "unknown",
            notes=notes,
        )
        manifest_path = os.path.join(self.manifests_dir, f"{policy_id}.json")
        write_strategy_manifest(manifest_path, manifest)

        created_at = datetime.now(timezone.utc).isoformat()
        self._index[policy_id] = {
            "role": role,
            "policy_path": manifest.artifacts["policy"],
            "manifest_path": f"manifests/{policy_id}.json",
            "created_at": created_at,
        }
        self._save_index()

        if role == "average":
            self._current_avg = policy_id
            self._write_latest_avg(policy_id)

        return policy_id

    def current_policy_id(self) -> str:
        if not self._current_avg:
            raise RuntimeError("No average policy logged yet.")
        return self._current_avg

    def current_policy(self) -> Policy:
        return self.load_policy(self.current_policy_id())

    def load_policy(self, policy_id: str) -> Policy:
        entry = self._index.get(policy_id)
        if entry is None:
            raise KeyError(f"Unknown policy id: {policy_id}")
        policy_rel = entry["policy_path"]
        policy_path = os.path.join(self.run_dir, policy_rel)
        payload = self._load_policy_payload(policy_path)
        policy = policy_from_json(payload)
        policy.bind_rules(rules_for_spec(self.spec))
        return policy

    def expand_lineage(self, avg_id: str) -> List[tuple[str, str, float]]:
        manifest_path = os.path.join(self.manifests_dir, f"{avg_id}.json")
        manifest = read_strategy_manifest(manifest_path)
        lineage = []
        for parent in manifest.parents:
            parent_id = parent.get("id")
            parent_role = parent.get("role", "")
            weight = float(parent.get("weight", 0.0))
            lineage.append((parent_id, parent_role, weight))
        return lineage

    def log_event(self, event: str, **payload) -> None:
        record = {"event": event, "timestamp": datetime.now(timezone.utc).isoformat(), **payload}
        os.makedirs(os.path.dirname(self._timeline_path), exist_ok=True)
        with open(self._timeline_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True))
            f.write("\n")

    def write_iteration_artifacts(self, iter_num: int, eval_results: Dict, summary: Dict) -> None:
        base = os.path.join(self.trainer_dir, "iterations", f"{iter_num:04d}")
        os.makedirs(base, exist_ok=True)
        save_json(os.path.join(base, "eval.json"), eval_results)
        save_json(os.path.join(base, "summary.json"), summary)

    def list_policies(self, role: str | None = None) -> List[str]:
        ids = sorted(self._index.keys())
        if role is None:
            return ids
        return [pid for pid in ids if self._index[pid]["role"] == role]

    # Internal helpers -------------------------------------------------

    def _generate_run_id(self, seed: int) -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_id = f"run_{timestamp}_{seed}"
        suffix = 1
        while os.path.exists(os.path.join(self.save_root, "runs", run_id)):
            run_id = f"run_{timestamp}_{seed}_{suffix}"
            suffix += 1
        return run_id

    def _write_run_metadata(self) -> None:
        payload = {
            "spec": json.loads(self.spec.to_json()),
            "env_hash": self.env_hash,
            "seed": self.seed,
            "code_sha": self.code_sha,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        save_json(os.path.join(self.run_dir, "run.json"), payload)

    def _hydrate_from_disk(self) -> None:
        if os.path.exists(self._index_path):
            self._index = load_json(self._index_path).get("policies", {})
        else:
            self._index = {}

        self._avg_counter = 0
        self._br_counter = 0

        for policy_id in self._index.keys():
            if policy_id.startswith("A"):
                try:
                    numeric = int(policy_id[1:])
                    self._avg_counter = max(self._avg_counter, numeric + 1)
                except ValueError:
                    continue
            elif policy_id.startswith("B"):
                try:
                    numeric = int(policy_id[1:])
                    self._br_counter = max(self._br_counter, numeric + 1)
                except ValueError:
                    continue

        if os.path.exists(self._latest_avg_path):
            with open(self._latest_avg_path, "r", encoding="utf-8") as f:
                latest = f.read().strip()
                self._current_avg = latest or None
        else:
            self._current_avg = None

    def _next_policy_id(self, role: str) -> str:
        if role == "average":
            policy_id = f"A{self._avg_counter}"
            self._avg_counter += 1
            return policy_id
        if role == "best_response":
            policy_id = f"B{self._br_counter}"
            self._br_counter += 1
            return policy_id
        raise ValueError(f"Unsupported role: {role}")

    def _policy_kind(self, policy: Policy) -> str:
        from liars_poker.policies.random import RandomPolicy
        from liars_poker.policies.tabular import TabularPolicy
        from liars_poker.policies.commit_once import CommitOnceMixture

        if isinstance(policy, RandomPolicy):
            return "policy/random"
        if isinstance(policy, TabularPolicy):
            return "policy/tabular"
        if isinstance(policy, CommitOnceMixture):
            return "policy/mixture"
        return f"policy/{policy.__class__.__name__.lower()}"

    def _save_index(self) -> None:
        payload = {"policies": self._index}
        save_json(self._index_path, payload)

    def _write_latest_avg(self, policy_id: str) -> None:
        with open(self._latest_avg_path, "w", encoding="utf-8") as f:
            f.write(policy_id)

    def _load_policy_payload(self, path: str) -> Dict:
        if path.endswith(".json"):
            return load_json(path)
        return load_jsonz(path)
