from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any, Dict, Sequence, Tuple

if TYPE_CHECKING:
    from .env import Rules


class Policy:
    """Policy protocol: per-episode init plus distributions and sampling.

    begin_episode(rng) -> None
    action_probs(infoset_key) -> Dict[action, prob]
    sample(..., rng) -> action
    """

    def __init__(self) -> None:
        self._rules: Rules | None = None

    def begin_episode(self, rng: random.Random | None = None) -> None:
        """Called exactly once at the start of each episode."""

        return None

    def bind_rules(self, rules: Rules) -> None:
        self._rules = rules

    def _require_rules(self) -> Rules:
        if self._rules is None:
            raise RuntimeError("Policy must be bound to Rules before use.")
        return self._rules

    def _legal(self, infoset_key: Tuple) -> Tuple[int, ...]:
        rules = self._require_rules()
        computed = rules.legal_actions_for(infoset_key)
        return computed

    def action_probs(self, infoset_key: Tuple) -> Dict[int, float]:
        raise NotImplementedError

    def sample(
        self,
        infoset_key: Tuple,
        rng: random.Random | None = None,
    ) -> int:
        if rng is None:
            raise ValueError("Policy.sample requires an RNG instance.")
        probs = self.action_probs(infoset_key)
        return _sample_from_probs(probs, rng)

    def to_json(self) -> Dict[str, Any]:
        raise NotImplementedError


def _sample_from_probs(probs: Dict[int, float], rng: random.Random) -> int:
    actions = sorted(probs.keys()) if probs else []
    p = [probs[a] for a in actions]
    # Normalize just in case
    s = sum(p)
    if s <= 0:
        raise ValueError("Empty or zero-prob distribution")
    pn = [x / s for x in p]
    r = rng.random()
    c = 0.0
    for a, pr in zip(actions, pn):
        c += pr
        if r <= c:
            return a
    return actions[-1]


class RandomPolicy(Policy):
    def begin_episode(self, rng: random.Random | None = None) -> None:
        return None

    def action_probs(self, infoset_key: Tuple) -> Dict[int, float]:
        legal = self._legal(infoset_key)
        n = len(legal)
        if n == 0:
            return {}
        p = 1.0 / n
        return {a: p for a in legal}

    def to_json(self) -> Dict[str, Any]:
        return {"class": "RandomPolicy"}

    @classmethod
    def from_json(cls, _: Dict[str, Any]) -> "RandomPolicy":
        return cls()


class TabularPolicy(Policy):
    """A simple dict-of-dicts policy.

    - probs[infoset_key] = {action: prob}
    If an infoset is unseen, defaults to uniform over legal actions.
    """

    def __init__(self):
        super().__init__()
        self.probs: Dict[Tuple, Dict[int, float]] = {}

    def begin_episode(self, rng: random.Random | None = None) -> None:
        return None

    def set(self, infoset_key: Tuple, dist: Dict[int, float]) -> None:
        self.probs[infoset_key] = dict(dist)

    def action_probs(self, infoset_key: Tuple) -> Dict[int, float]:
        legal = self._legal(infoset_key)
        if infoset_key not in self.probs:
            n = len(legal)
            if n == 0:
                return {}
            p = 1.0 / n
            return {a: p for a in legal}
        # Filter to legal support, renormalize
        dist = {a: self.probs[infoset_key].get(a, 0.0) for a in legal}
        s = sum(dist.values())
        if s <= 0:
            # fallback to uniform
            n = len(legal)
            return {a: 1.0 / n for a in legal}
        return {a: v / s for a, v in dist.items()}

    def to_json(self) -> Dict[str, Any]:
        entries = []
        for infoset, dist in self.probs.items():
            entries.append({
                "infoset": _pack_structure(infoset),
                "dist": dist,
            })
        return {"class": "TabularPolicy", "entries": entries}

    @classmethod
    def from_json(cls, payload: Dict[str, Any]) -> "TabularPolicy":
        policy = cls()
        for entry in payload.get("entries", []):
            infoset = _unpack_structure(entry["infoset"])
            dist = entry.get("dist", {})
            policy.set(infoset, dist)
        return policy


class PerDecisionMixture(Policy):
    """Per-decision convex mixture: mu(u) = (1-w) pi(u) + w beta(u)."""

    def __init__(self, pi: Policy, beta: Policy, w: float):
        super().__init__()
        assert 0.0 <= w <= 1.0
        self.pi = pi
        self.beta = beta
        self.w = w

    def begin_episode(self, rng: random.Random | None = None) -> None:
        self.pi.begin_episode(rng)
        self.beta.begin_episode(rng)

    def action_probs(self, infoset_key: Tuple) -> Dict[int, float]:
        legal = self._legal(infoset_key)
        p_pi = self.pi.action_probs(infoset_key)
        p_be = self.beta.action_probs(infoset_key)
        # Combine on the same support, then renormalize
        supp = list(legal)
        mixed = {a: (1.0 - self.w) * p_pi.get(a, 0.0) + self.w * p_be.get(a, 0.0) for a in supp}
        s = sum(mixed.values())
        if s <= 0:
            n = len(supp)
            return {a: 1.0 / n for a in supp}
        return {a: v / s for a, v in mixed.items()}

    def bind_rules(self, rules: Rules) -> None:
        super().bind_rules(rules)
        self.pi.bind_rules(rules)
        self.beta.bind_rules(rules)

    def to_json(self) -> Dict[str, Any]:
        return {
            "class": "PerDecisionMixture",
            "pi": self.pi.to_json(),
            "beta": self.beta.to_json(),
            "w": self.w,
        }

    @classmethod
    def from_json(cls, payload: Dict[str, Any]) -> "PerDecisionMixture":
        pi = policy_from_json(payload["pi"])
        beta = policy_from_json(payload["beta"])
        return cls(pi, beta, float(payload.get("w", 0.0)))


class CommitOnceMixture(Policy):
    """Normal-form mixture: flip once per episode and commit to one policy."""

    def __init__(self, pi: Policy, beta: Policy, w: float, rng: random.Random | None = None):
        super().__init__()
        assert 0.0 <= w <= 1.0
        self.pi = pi
        self.beta = beta
        self.w = w
        self._rng = rng or random.Random()
        self._choice: int | None = None  # 0 -> pi, 1 -> beta
 
    def begin_episode(self, rng: random.Random | None = None) -> None:
        if rng is not None:
            self._rng = rng
        self._choice = 1 if self._rng.random() < self.w else 0
        self.pi.begin_episode(rng)
        self.beta.begin_episode(rng)

    def action_probs(self, infoset_key: Tuple) -> Dict[int, float]:
        legal = self._legal(infoset_key)
        assert self._choice is not None, "CommitOnceMixture: call begin_episode() after env.reset()."
        if self._choice == 0:
            return self.pi.action_probs(infoset_key)
        else:
            return self.beta.action_probs(infoset_key)

    def sample(
        self,
        infoset_key: Tuple,
        rng: random.Random | None = None,
    ) -> int:
        if rng is None:
            raise ValueError("Policy.sample requires an RNG instance.")
        assert self._choice is not None, "CommitOnceMixture: call begin_episode() after env.reset()."
        if self._choice == 0:
            return self.pi.sample(infoset_key, rng)
        else:
            return self.beta.sample(infoset_key, rng)

    def to_json(self) -> Dict[str, Any]:
        return {
            "class": "CommitOnceMixture",
            "pi": self.pi.to_json(),
            "beta": self.beta.to_json(),
            "w": self.w,
        }

    @classmethod
    def from_json(cls, payload: Dict[str, Any]) -> "CommitOnceMixture":
        pi = policy_from_json(payload["pi"])
        beta = policy_from_json(payload["beta"])
        w = float(payload.get("w", 0.0))
        return cls(pi, beta, w)

    def bind_rules(self, rules: Rules) -> None:
        super().bind_rules(rules)
        self.pi.bind_rules(rules)
        self.beta.bind_rules(rules)


def _pack_structure(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_pack_structure(v) for v in value]
    return value


def _unpack_structure(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_unpack_structure(v) for v in value)
    return value


def policy_from_json(payload: Dict[str, Any]) -> Policy:
    cls_name = payload.get("class")
    if cls_name == "RandomPolicy":
        return RandomPolicy.from_json(payload)
    if cls_name == "TabularPolicy":
        return TabularPolicy.from_json(payload)
    if cls_name == "PerDecisionMixture":
        return PerDecisionMixture.from_json(payload)
    if cls_name == "CommitOnceMixture":
        return CommitOnceMixture.from_json(payload)
    raise ValueError(f"Unknown policy class: {cls_name}")
