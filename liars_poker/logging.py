from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional


@dataclass(slots=True, frozen=True)
class StrategyManifest:
    id: str
    role: str  # "average" | "best_response"
    kind: str  # "policy/tabular" | "policy/neural" | "policy/random"
    env_hash: str
    parents: List[Dict[str, Any]]  # [{id, role, weight}]
    mixing: Optional[Dict[str, Any]]  # {impl, eta, ...}
    seeds: Dict[str, Any]
    train: Dict[str, Any]
    artifacts: Dict[str, str]
    code_sha: str
    notes: Optional[str] = None

def save_json(path: str, obj: Dict[str, Any]) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_strategy_manifest(path: str, manifest: StrategyManifest) -> None:
    save_json(path, asdict(manifest))


def read_strategy_manifest(path: str) -> StrategyManifest:
    data = load_json(path)
    return StrategyManifest(**data)

