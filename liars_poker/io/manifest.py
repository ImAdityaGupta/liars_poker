from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from .jsonio import load_json, save_json


@dataclass(slots=True, frozen=True)
class StrategyManifest:
    id: str
    role: str
    kind: str
    env_hash: str
    parents: List[Dict[str, Any]]
    mixing: Optional[Dict[str, Any]]
    seeds: Dict[str, Any]
    train: Dict[str, Any]
    artifacts: Dict[str, str]
    code_sha: str
    notes: Optional[str] = None


def write_strategy_manifest(path: str, manifest: StrategyManifest) -> None:
    save_json(path, asdict(manifest))


def read_strategy_manifest(path: str) -> StrategyManifest:
    data = load_json(path)
    return StrategyManifest(**data)

