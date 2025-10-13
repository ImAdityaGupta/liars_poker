from __future__ import annotations

from typing import Any, Dict, List

from liars_poker.infoset import InfoSet
from liars_poker.policies.commit_once import CommitOnceMixture
from liars_poker.policies.random import RandomPolicy
from liars_poker.policies.tabular import TabularPolicy
from liars_poker.policies.base import Policy


def policy_to_json(policy: Policy) -> Dict[str, Any]:
    if isinstance(policy, RandomPolicy):
        return {"class": "RandomPolicy"}
    if isinstance(policy, TabularPolicy):
        entries = []
        for infoset, dist in policy.probs.items():
            entries.append({"infoset": _pack_infoset(infoset), "dist": dict(dist)})
        return {"class": "TabularPolicy", "entries": entries}
    if isinstance(policy, CommitOnceMixture):
        return {
            "class": "CommitOnceMixture",
            "weights": list(policy.weights),
            "policies": [policy_to_json(child) for child in policy.policies],
        }
    raise ValueError(f"Unsupported policy type for serialization: {policy.__class__.__name__}")


def policy_from_json(payload: Dict[str, Any]) -> Policy:
    cls_name = payload.get("class")
    if cls_name == "RandomPolicy":
        return RandomPolicy()
    if cls_name == "TabularPolicy":
        policy = TabularPolicy()
        for entry in payload.get("entries", []):
            infoset = _unpack_infoset(entry["infoset"])
            dist = entry.get("dist", {})
            policy.set(infoset, dist)
        return policy
    if cls_name == "CommitOnceMixture":
        children_payload = payload.get("policies", [])
        weights = payload.get("weights", [])
        children = [policy_from_json(child) for child in children_payload]
        return CommitOnceMixture(children, weights)
    raise ValueError(f"Unknown policy class: {cls_name}")


def _pack_infoset(iset: InfoSet) -> List[Any]:
    return [
        iset.pid,
        iset.last_idx,
        list(iset.hand),
        list(iset.history),
    ]


def _unpack_infoset(payload: List[Any]) -> InfoSet:
    pid, last_idx, hand, history = payload
    return InfoSet(pid=int(pid), last_idx=int(last_idx), hand=tuple(int(c) for c in hand), history=tuple(int(a) for a in history))

