from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Iterable, Tuple, Type

import numpy as np

from liars_poker.core import GameSpec
from liars_poker.env import rules_for_spec

if False:  # pragma: nocover - type checking only
    from liars_poker.policies.base import Policy


REGISTRY: Dict[str, Type["Policy"]] = {}
FORMAT_VERSION = 1
METADATA_FILENAME = "metadata.json"
BLOBS_FILENAME = "blobs.npz"


def register_policy(cls: Type["Policy"]) -> Type["Policy"]:
    """Register a policy class by its POLICY_KIND."""

    kind = getattr(cls, "POLICY_KIND", None) or cls.__name__
    REGISTRY[kind] = cls
    return cls


def _spec_to_dict(spec: GameSpec) -> Dict:
    return {
        "ranks": spec.ranks,
        "suits": spec.suits,
        "hand_size": spec.hand_size,
        "claim_kinds": list(spec.claim_kinds),
        "suit_symmetry": spec.suit_symmetry,
    }


def _spec_from_dict(d: Dict) -> GameSpec:
    return GameSpec(
        ranks=d["ranks"],
        suits=d["suits"],
        hand_size=d["hand_size"],
        claim_kinds=tuple(d["claim_kinds"]),
        suit_symmetry=bool(d["suit_symmetry"]),
    )


def _normalize_children(policy: "Policy") -> Iterable[Tuple[str, "Policy"]]:
    for idx, child_entry in enumerate(policy.iter_children()):
        if isinstance(child_entry, tuple) and len(child_entry) == 2:
            label, child = child_entry
        else:
            label, child = f"child_{idx}", child_entry  # type: ignore[assignment]
        yield str(label), child


def _collect_payload(policy: "Policy", prefix: str = "") -> Tuple[Dict, Dict[str, np.ndarray]]:
    data_payload, local_blobs = policy.to_payload()
    flat_blobs: Dict[str, np.ndarray] = {
        f"{prefix}{name}": arr for name, arr in local_blobs.items()
    }

    children_payloads = []
    for child_idx, (label, child) in enumerate(_normalize_children(policy)):
        child_prefix = f"{prefix}{label}/"
        payload, blobs = _collect_payload(child, child_prefix)
        children_payloads.append(payload)
        flat_blobs.update(blobs)

    payload = {
        "kind": getattr(policy, "POLICY_KIND", policy.__class__.__name__),
        "version": getattr(policy, "POLICY_VERSION", 1),
        "prefix": prefix,
        "data": data_payload,
        "children": children_payloads,
    }
    return payload, flat_blobs


def save_policy(policy: "Policy", directory: str) -> None:
    """Serialize a policy (and its spec) into `directory`.

    - metadata.json stores kind/version/spec plus the nested payload tree.
    - blobs.npz stores all numpy arrays, flattened with hierarchical keys.
    """

    rules = policy._require_rules()
    spec = rules.spec

    payload, blobs = _collect_payload(policy, prefix="")

    meta = {
        "format_version": FORMAT_VERSION,
        "spec": _spec_to_dict(spec),
        "root": payload,
    }

    out_dir = Path(directory)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_path = out_dir / METADATA_FILENAME
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, sort_keys=True)

    if blobs:
        np.savez_compressed(out_dir / BLOBS_FILENAME, **blobs)


def _local_blobs(prefix: str, blobs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    n = len(prefix)
    return {k[n:]: v for k, v in blobs.items() if k.startswith(prefix)}


def _instantiate(payload: Dict, blobs: Dict[str, np.ndarray]) -> "Policy":
    kind = payload["kind"]
    cls = REGISTRY.get(kind)
    if cls is None:
        raise ValueError(f"Unknown policy kind '{kind}' in payload.")

    prefix = payload.get("prefix", "")
    children_payloads = payload.get("children", []) or []
    children = [_instantiate(child_payload, blobs) for child_payload in children_payloads]

    scoped_blobs = _local_blobs(prefix, blobs)
    return cls.from_payload(
        payload.get("data", {}),
        blob_prefix=prefix,
        blobs=scoped_blobs,
        children=children,
    )


def load_policy(directory: str) -> Tuple["Policy", GameSpec]:
    """Load a policy (and spec) from `directory`."""

    meta_path = Path(directory) / METADATA_FILENAME
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata file at {meta_path}")

    with meta_path.open("r", encoding="utf-8") as handle:
        meta = json.load(handle)

    format_version = meta.get("format_version")
    if format_version != FORMAT_VERSION:
        raise ValueError(f"Unsupported format_version {format_version}, expected {FORMAT_VERSION}")

    spec = _spec_from_dict(meta["spec"])

    blobs_path = Path(directory) / BLOBS_FILENAME
    blobs: Dict[str, np.ndarray] = {}
    if blobs_path.exists():
        with np.load(blobs_path, allow_pickle=True) as npz:
            blobs = {k: npz[k] for k in npz.files}

    policy = _instantiate(meta["root"], blobs)
    policy.bind_rules(rules_for_spec(spec))
    return policy, spec


def _ensure_builtin_registration() -> None:
    # Lazy import to avoid cycles
    # We removed the try/except so Python will scream if a file is missing
    from liars_poker.policies.random_policy import RandomPolicy
    from liars_poker.policies.tabular import TabularPolicy
    from liars_poker.policies.tabular_dense import DenseTabularPolicy
    from liars_poker.policies.commit_once import CommitOnceMixture

    for cls in (RandomPolicy, TabularPolicy, DenseTabularPolicy, CommitOnceMixture):
        register_policy(cls)


_ensure_builtin_registration()
