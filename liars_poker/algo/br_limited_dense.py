"""Belief-exact, depth-limited responders with lazy opponent queries.

Depth counts the responder's decisions. Depth one is local BR: claim, then
CALL at the next opportunity. Pruning affects planning only; reported values
always evaluate the resulting receding-horizon policy on full opponent support.
"""
from __future__ import annotations

from functools import lru_cache
import time

import numpy as np
import torch

from liars_poker.algo.br_exact_dense_to_dense import adjustment_factor
from liars_poker.core import card_rank, possible_starting_hands
from liars_poker.env import rules_for_spec
from liars_poker.infoset import CALL
from liars_poker.policies.base import Policy
from liars_poker.policies.neural import NeuralPolicy
from liars_poker.policies.neural_regret import NeuralRegretMatchingPolicy
from liars_poker.policies.tabular_dense import DenseTabularPolicy


class LimitedBestResponse:
    def __init__(self, opponent: Policy, depth: int = 1, epsilon: float = 0.0):
        if depth < 1 or epsilon < 0:
            raise ValueError("depth must be positive and epsilon nonnegative")
        if not isinstance(opponent, (DenseTabularPolicy, NeuralPolicy, NeuralRegretMatchingPolicy)):
            raise TypeError("Expected a dense, neural-average, or neural-regret policy")
        self.opponent = opponent
        self.spec = opponent.spec
        self.rules = rules_for_spec(self.spec)
        self.depth = depth
        self.epsilon = epsilon
        self.hands = tuple(possible_starting_hands(self.spec))
        self.n = len(self.hands)
        self.k = len(self.rules.claims)
        self.network_queries = 0
        self.network_rows = 0
        self.network_query_s = 0.0
        self.blockers = np.array([[adjustment_factor(opponent.spec, a, b)
                                   for b in self.hands] for a in self.hands], dtype=np.float64)
        self.hand_weights = np.array([adjustment_factor(opponent.spec, (), hand)
                                      for hand in self.hands], dtype=np.float64)
        ranks = np.array([[sum(card_rank(card, opponent.spec) == r for card in hand)
                           for r in range(opponent.spec.ranks + 1)] for hand in self.hands])
        self.truth = []
        for kind, value in self.rules.claims:
            if kind == "TwoPair":
                a, b = self.rules.two_pair_ranks[value]
                true = ((ranks[:, a, None] + ranks[None, :, a] >= 2)
                        & (ranks[:, b, None] + ranks[None, :, b] >= 2))
            elif kind == "FullHouse":
                a, b = self.rules.full_house_ranks[value]
                true = ((ranks[:, a, None] + ranks[None, :, a] >= 3)
                        & (ranks[:, b, None] + ranks[None, :, b] >= 2))
            else:
                need = {"RankHigh": 1, "Pair": 2, "Trips": 3, "Quads": 4}[kind]
                true = ranks[:, value, None] + ranks[None, :, value] >= need
            self.truth.append(true)
        self.plan_nodes = 0
        self.eval_nodes = 0
        self.skipped_search_branches = 0
        self.search_branches = 0
        # Per-run caches release their policy and search trees when this run ends.
        self._actions = lru_cache(maxsize=100_000)(self._actions_uncached)
        self._probs = lru_cache(maxsize=100_000)(self._probs_uncached)
        self._reach = lru_cache(maxsize=100_000)(self._reach_uncached)
        self._plan = lru_cache(maxsize=250_000)(self._plan_uncached)
        self._choice = lru_cache(maxsize=100_000)(self._choice_uncached)
        self._evaluate = lru_cache(maxsize=100_000)(self._evaluate_uncached)

    @staticmethod
    def _last_claim(hid: int) -> int:
        return hid.bit_length() - 1

    def _actions_uncached(self, hid: int) -> tuple[int, ...]:
        return self.rules.legal_actions_from_last(None if hid == 0 else self._last_claim(hid))

    def _probs_uncached(self, hid: int) -> np.ndarray:
        """One policy query for every opponent hand at this public history."""
        if isinstance(self.opponent, DenseTabularPolicy):
            return self.opponent.S[hid]
        start = time.perf_counter()
        policy = self.opponent
        history = tuple(i for i in range(self.k) if (hid >> i) & 1)
        x = torch.from_numpy(policy.encoder.encode_hands(self.hands, history)).to(policy.device)
        legal = self._actions(hid)
        cols = [0 if action == CALL else action + 1 for action in legal]
        with torch.inference_mode():
            logits = policy._model(hid.bit_count() & 1)(x)
            selected = logits[:, cols]
            if isinstance(policy, NeuralRegretMatchingPolicy):
                positive = selected.clamp_min(0)
                totals = positive.sum(dim=1, keepdim=True)
                probs = torch.where(totals > 0, positive / totals.clamp_min(1e-30),
                                    torch.full_like(positive, 1.0 / len(cols)))
            else:
                probs = torch.softmax(selected, dim=1)
            row = np.zeros((self.n, self.k + 1), dtype=np.float32)
            row[:, cols] = probs.cpu().numpy()
        self.network_queries += 1
        self.network_rows += self.n
        self.network_query_s += time.perf_counter() - start
        return row

    def _reach_uncached(self, hid: int, seat: int) -> np.ndarray:
        if isinstance(self.opponent, DenseTabularPolicy):
            return self.opponent.L_pid1[hid] if seat == 0 else self.opponent.L_pid0[hid]
        if hid == 0:
            return np.ones(self.n, dtype=np.float64)
        action = self._last_claim(hid)
        prev = hid ^ (1 << action)
        reach = self._reach(prev, seat)
        if (prev.bit_count() & 1) != seat:
            return reach * self._probs(prev)[:, action + 1]
        return reach

    def _win(self, hid: int, hand: int, seat: int, caller: int,
             reach: np.ndarray) -> float:
        mass = self.blockers[hand] * reach
        truth = self.truth[self._last_claim(hid)][hand]
        return float(np.dot(mass, truth if caller != seat else ~truth))

    def _next_opponent(self, hid: int, hand: int, seat: int,
                       depth: int, *, planning: bool) -> float:
        """Value after our claim; opponent acts at hid."""
        reach = self._reach(hid, seat)
        S = self._probs(hid)
        baseline_mass = float(np.dot(self.blockers[hand], reach))
        total = 0.0
        for action in self._actions(hid):
            col = 0 if action == CALL else action + 1
            weighted = reach * S[:, col]
            branch_mass = float(np.dot(self.blockers[hand], weighted))
            if branch_mass <= 0:
                continue
            if planning:
                self.search_branches += 1
                if action != CALL and baseline_mass > 0 and branch_mass < self.epsilon * baseline_mass:
                    self.skipped_search_branches += 1
                    continue
            if action == CALL:
                total += self._win(hid, hand, seat, 1 - seat, weighted)
            else:
                next_hid = hid | (1 << action)
                if planning:
                    total += (self._plan(next_hid, hand, seat, depth - 1)
                              if depth > 1 else self._win(next_hid, hand, seat, seat,
                                                           self._reach(next_hid, seat)))
                else:
                    total += self._evaluate(next_hid, hand, seat)
        return total

    def _plan_uncached(self, hid: int, hand: int, seat: int, depth: int) -> float:
        self.plan_nodes += 1
        reach = self._reach(hid, seat)
        best = -1.0
        for action in self._actions(hid):
            value = (self._win(hid, hand, seat, seat, reach) if action == CALL else
                     self._next_opponent(hid | (1 << action), hand, seat, depth,
                                         planning=True))
            if value > best:
                best = value
        return best

    def _choice_uncached(self, hid: int, hand: int, seat: int) -> int:
        reach = self._reach(hid, seat)
        best_action = self._actions(hid)[0]
        best = -1.0
        for action in self._actions(hid):
            value = (self._win(hid, hand, seat, seat, reach) if action == CALL else
                     self._next_opponent(hid | (1 << action), hand, seat, self.depth,
                                         planning=True))
            if value > best:
                best, best_action = value, action
        return best_action

    def _evaluate_uncached(self, hid: int, hand: int, seat: int) -> float:
        self.eval_nodes += 1
        if (hid.bit_count() & 1) != seat:
            reach = self._reach(hid, seat)
            S = self._probs(hid)
            value = 0.0
            for action in self._actions(hid):
                col = 0 if action == CALL else action + 1
                if action == CALL:
                    value += self._win(hid, hand, seat, 1 - seat, reach * S[:, col])
                elif np.any(S[:, col] * reach):
                    value += self._evaluate(hid | (1 << action), hand, seat)
            return value
        action = self._choice(hid, hand, seat)
        if action == CALL:
            return self._win(hid, hand, seat, seat, self._reach(hid, seat))
        return self._next_opponent(hid | (1 << action), hand, seat, self.depth,
                                   planning=False)

    def evaluate_seat(self, seat: int) -> dict:
        if seat not in (0, 1):
            raise ValueError("seat must be 0 or 1")
        start = time.perf_counter()
        values = np.array([self._evaluate(0, i, seat) for i in range(self.n)])
        initial_mass = self.blockers @ np.ones(self.n)
        p = float(np.dot(self.hand_weights, values / initial_mass) / self.hand_weights.sum())
        return {"seat": seat, "p": p, "elapsed_s": time.perf_counter() - start,
                "plan_nodes": self.plan_nodes, "eval_nodes": self.eval_nodes,
                "search_branches": self.search_branches,
                "skipped_search_branches": self.skipped_search_branches,
                "network_queries": self.network_queries,
                "network_rows": self.network_rows,
                "network_query_s": self.network_query_s,
                "policy_cache": self._probs.cache_info()._asdict(),
                "reach_cache": self._reach.cache_info()._asdict()}
