from __future__ import annotations

import random
from typing import Any, Dict, List, Sequence, Tuple, Iterable

from liars_poker.core import possible_starting_hands
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
            raise RuntimeError("CommitOnceMixture: begin_episode not yet called, call prob_dist_at_infoset for posterior-weighted mixture.")
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

    # --- Serialization ---

    def to_payload(self) -> Tuple[Dict, Dict[str, object]]:
        return {
            "kind": self.POLICY_KIND,
            "version": self.POLICY_VERSION,
            "weights": list(self.weights),
        }, {}

    @classmethod
    def from_payload(
        cls,
        payload: Dict,
        *,
        blob_prefix: str,
        blobs: Dict[str, object],
        children: Iterable[Policy],
    ) -> "CommitOnceMixture":
        _ = (blob_prefix, blobs)
        weights = list(payload.get("weights", []))
        children_list = list(children)
        if len(children_list) != len(weights):
            raise ValueError("CommitOnceMixture payload children/weights length mismatch.")
        return cls(children_list, weights, rng=random.Random())

    def iter_children(self):
        for idx, child in enumerate(self.policies):
            yield f"child_{idx}", child

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
