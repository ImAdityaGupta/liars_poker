from __future__ import annotations

import os
import pickle
import random
from typing import Any, Dict, List, Sequence, Tuple

from liars_poker.core import GameSpec, possible_starting_hands
from liars_poker.env import rules_for_spec
from liars_poker.infoset import CALL, NO_CLAIM, InfoSet

from .base import Policy
from .tabular import TabularPolicy


class CommitOnceMixture(Policy):
    POLICY_KIND = "CommitOnceMixture"
    """Commit-once mixture that can expose posterior-aware distributions."""

    def __init__(
        self,
        policies: Sequence[Policy],
        weights: Sequence[float],
        rng: random.Random | None = None,
    ) -> None:
        super().__init__()
        if len(policies) != len(weights):
            raise ValueError("Policies and weights must have the same length.")

        filtered: List[Tuple[Policy, float]] = []
        total = 0.0
        for policy, weight in zip(policies, weights):
            if weight < 0:
                raise ValueError("CommitOnceMixture weights must be non-negative.")
            if weight > 0:
                filtered.append((policy, weight))
                total += weight

        if not filtered or total <= 0:
            raise ValueError("At least one positive weight is required.")

        self.policies: List[Policy] = [p for p, _ in filtered]
        self.weights: List[float] = [w / total for _, w in filtered]
        self._rng = rng or random.Random()
        self._choice: int | None = None

    def bind_rules(self, rules) -> None:  # type: ignore[override]
        super().bind_rules(rules)
        for policy in self.policies:
            policy.bind_rules(rules)

    def begin_episode(self, rng: random.Random | None = None) -> None:
        if rng is not None:
            self._rng = rng
        self._choice = self._weighted_choice()
        for policy in self.policies:
            policy.begin_episode(self._rng)

    def action_probs(self, infoset: InfoSet) -> Dict[int, float]:
        if self._choice is None:
            raise RuntimeError("CommitOnceMixture: begin_epsiode not yet called, call prob_dist_at_infoset for posterior-weighted mixture.")
        return self.policies[self._choice].action_probs(infoset)

    def sample(self, infoset: InfoSet, rng: random.Random) -> int:
        if self._choice is None:
            raise RuntimeError("CommitOnceMixture: call begin_episode() after env.reset().")
        return self.policies[self._choice].sample(infoset, rng)

    def prob_dist_at_infoset(self, infoset: InfoSet) -> Dict[int, float]:
        posteriors = self._posterior_weights(infoset)
        mixed: Dict[int, float] = {}
        for weight, policy in zip(posteriors, self.policies):
            dist = policy.prob_dist_at_infoset(infoset)
            for action, prob in dist.items():
                mixed[action] = mixed.get(action, 0.0) + weight * prob

        total = sum(mixed.values())
        if total <= 0:
            legal = self._legal_actions(infoset)
            if not legal:
                return {}
            prob = 1.0 / len(legal)
            return {action: prob for action in legal}
        return {action: value / total for action, value in mixed.items()}

    def _weighted_choice(self) -> int:
        pick = self._rng.random()
        cumulative = 0.0
        for idx, weight in enumerate(self.weights):
            cumulative += weight
            if pick < cumulative:
                return idx
        return len(self.weights) - 1

    def _posterior_weights(self, infoset: InfoSet) -> List[float]:
        likelihoods: List[float] = []
        for policy, prior in zip(self.policies, self.weights):
            likelihood = self._sequence_likelihood(policy, infoset)
            likelihoods.append(prior * likelihood)

        total = sum(likelihoods)
        if total <= 0:
            return list(self.weights)
        return [value / total for value in likelihoods]

    def _sequence_likelihood(self, policy: Policy, infoset: InfoSet) -> float:
        history = infoset.history
        pid = infoset.pid
        likelihood = 1.0

        for turn in range(len(history)):
            if turn % 2 != pid:
                continue
            action = history[turn]
            prefix = history[:turn]
            partial_iset = InfoSet(
                pid=pid,
                hand=infoset.hand,
                history=prefix,
            )
            dist = policy.prob_dist_at_infoset(partial_iset)
            likelihood *= dist.get(action, 0.0)
            if likelihood == 0.0:
                break
        return likelihood

    def store_efficiently(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        spec = self._require_rules().spec
        child_entries = []
        for idx, child in enumerate(self.policies):
            child_dir_name = f"child_{idx}"
            child_dir = os.path.join(directory, child_dir_name)
            child.store_efficiently(child_dir)
            child_entries.append({"dir": child_dir_name, "kind": child.__class__.__name__})

        payload = {
            "kind": self.POLICY_KIND,
            "spec": spec,
            "weights": list(self.weights),
            "children": child_entries,
        }
        path = os.path.join(directory, self.POLICY_BINARY_FILENAME)
        with open(path, "wb") as handle:
            pickle.dump(payload, handle)

    @classmethod
    def load_efficiently(cls, directory: str) -> Tuple["CommitOnceMixture", GameSpec]:
        path = os.path.join(directory, cls.POLICY_BINARY_FILENAME)
        with open(path, "rb") as handle:
            payload = pickle.load(handle)
        policy, spec = cls._from_serialized(payload, directory)
        policy.bind_rules(rules_for_spec(spec))
        return policy, spec

    @classmethod
    def _from_serialized(cls, payload, directory: str) -> Tuple["CommitOnceMixture", GameSpec]:
        spec: GameSpec = payload["spec"]
        children_meta: List[Dict[str, Any]] = payload.get("children", [])
        weights: List[float] = list(payload.get("weights", []))
        children: List[Policy] = []
        for entry in children_meta:
            child_dir = os.path.join(directory, entry["dir"])
            child_policy, child_spec = Policy.load_policy(child_dir)
            if child_spec != spec:
                child_policy.bind_rules(rules_for_spec(spec))
            children.append(child_policy)
        policy = cls(children, weights, rng=random.Random())
        return policy, spec

    def to_tabular(self) -> TabularPolicy:
        """Materialize this mixture as a TabularPolicy by enumerating all infosets."""

        rules = self._require_rules()
        spec = rules.spec
        tab = TabularPolicy()
        tab.bind_rules(rules)

        hands = possible_starting_hands(spec)

        def histories(history: Tuple[int, ...]) -> List[Tuple[int, ...]]:
            yield history
            if history and history[-1] == CALL:
                return
            last_claim = InfoSet.last_claim_idx(history)
            last_idx = None if last_claim == NO_CLAIM else last_claim
            for action in rules.legal_actions_from_last(last_idx):
                new_history = history + (action,)
                if action == CALL:
                    yield new_history
                else:
                    yield from histories(new_history)

        seen: set[InfoSet] = set()
        for history in histories(tuple()):
            if history and history[-1] == CALL:
                continue
            pid = len(history) % 2
            for hand in hands:
                infoset = InfoSet(pid=pid, hand=hand, history=history)
                if infoset in seen:
                    continue
                seen.add(infoset)
                dist = self.prob_dist_at_infoset(infoset)
                if dist:
                    tab.set(infoset, dist)

        return tab
