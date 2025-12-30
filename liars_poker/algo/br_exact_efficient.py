from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from liars_poker.core import GameSpec, generate_deck, possible_starting_hands, card_rank
from liars_poker.env import Rules, rules_for_spec
from liars_poker.infoset import CALL, NO_CLAIM, InfoSet
from liars_poker.policies.random import RandomPolicy
from liars_poker.policies.tabular import Policy, TabularPolicy


def adjustment_factor(spec: GameSpec, my_hand: Tuple[int, ...], opp_hand: Tuple[int, ...]) -> float:
    """Combinatorial weight of opp_hand given my_hand (card removal + multiset multiplicity)."""

    deck_counts = Counter(generate_deck(spec))

    for card in my_hand:
        if deck_counts[card] <= 0:
            return 0.0
        deck_counts[card] -= 1

    weight = 1
    opp_counts = Counter(opp_hand)
    for card, need in opp_counts.items():
        available = deck_counts.get(card, 0)
        if available < need:
            return 0.0
        weight *= math.comb(available, need)
        deck_counts[card] -= need

    return float(weight)


@dataclass(slots=True)
class _ClaimReq:
    rank: int
    need: int


class BestResponseComputerEfficient:
    """Vectorized exact best response with Bayesian weighting via a card-removal matrix."""

    def __init__(self, spec: GameSpec, opponent: Policy):
        self.spec = spec
        self.rules: Rules = rules_for_spec(spec)
        self.opponent = opponent

        self.hands: Tuple[Tuple[int, ...], ...] = tuple(possible_starting_hands(spec))
        self.n_hands = len(self.hands)
        self._hand_to_index = {hand: idx for idx, hand in enumerate(self.hands)}

        self.card_removal_mat = self._build_card_removal_matrix()
        self.hand_weights = np.asarray([adjustment_factor(spec, tuple(), h) for h in self.hands], dtype=float)
        self.hand_rank_counts = self._build_hand_rank_counts()

        # Baseline-compatible outputs
        self.state_card_values: Dict[Tuple[int, ...], Dict[Tuple[int, ...], float]] = {}
        self.probs: Dict[InfoSet, Dict[int, float]] = {}

        # Root values for exploitability reporting
        self._root_values: Dict[int, np.ndarray] = {}

        # Optional index for a tabular opponent
        self._opp_tabular_index: Optional[Dict[Tuple[int, Tuple[int, ...]], List[Tuple[int, Dict[int, float]]]]] = None
        if isinstance(opponent, TabularPolicy):
            index: Dict[Tuple[int, Tuple[int, ...]], List[Tuple[int, Dict[int, float]]]] = {}
            for iset, dist in opponent.probs.items():
                idx = self._hand_to_index.get(iset.hand)
                if idx is None:
                    continue
                key = (iset.pid, iset.history)
                index.setdefault(key, []).append((idx, dist))
            self._opp_tabular_index = index

        # Cache for opponent strategy matrices keyed by (pid, history, actions)
        self._strategy_cache: Dict[Tuple[int, Tuple[int, ...], Tuple[int, ...]], np.ndarray] = {}

        self._claim_reqs = self._build_claim_requirements()

    def _build_claim_requirements(self) -> Dict[int, _ClaimReq]:
        reqs: Dict[int, _ClaimReq] = {}
        for idx, (kind, rank_value) in enumerate(self.rules.claims):
            if kind == "RankHigh":
                need = 1
            elif kind == "Pair":
                need = 2
            else:
                raise ValueError(f"Unsupported claim kind: {kind}")
            reqs[idx] = _ClaimReq(rank=rank_value, need=need)
        return reqs

    def _build_hand_rank_counts(self) -> np.ndarray:
        # 1-based rank indexing for direct use with claim ranks.
        counts = np.zeros((self.n_hands, self.spec.ranks + 1), dtype=np.int16)
        for i, hand in enumerate(self.hands):
            for card in hand:
                r = card_rank(card, self.spec)
                counts[i, r] += 1
        return counts

    def _build_card_removal_matrix(self) -> np.ndarray:
        A = np.zeros((self.n_hands, self.n_hands), dtype=float)
        for i, my_hand in enumerate(self.hands):
            for j, opp_hand in enumerate(self.hands):
                A[i, j] = adjustment_factor(self.spec, my_hand, opp_hand)
        return A

    def _legal_actions_from_history(self, history: Tuple[int, ...]) -> Tuple[int, ...]:
        last_claim = InfoSet.last_claim_idx(history)
        last_idx = None if last_claim == NO_CLAIM else last_claim
        return self.rules.legal_actions_from_last(last_idx)

    def _get_strategy_matrix(self, policy: Policy, pid: int, history: Tuple[int, ...], actions: Tuple[int, ...]) -> np.ndarray:
        """Return an (N, |actions|) matrix of opponent action probabilities, per opponent hand."""

        K = len(actions)
        if K == 0:
            return np.zeros((self.n_hands, 0), dtype=float)

        cache_key = (pid, history, actions)
        cached = self._strategy_cache.get(cache_key)
        if cached is not None:
            return cached

        if isinstance(policy, RandomPolicy):
            return np.full((self.n_hands, K), 1.0 / K, dtype=float)

        # Initialize strategy matrix; fill rows either from tabular index or by querying the policy interface.
        S = np.full((self.n_hands, K), np.nan, dtype=float)

        # Optional fast fill from stored tabular entries (if present).
        if isinstance(policy, TabularPolicy) and self._opp_tabular_index is not None:
            for row_idx, dist in self._opp_tabular_index.get((pid, history), []):
                row = np.asarray([float(dist.get(a, 0.0)) for a in actions], dtype=float)
                s = float(row.sum())
                if s <= 0.0:
                    row[:] = 1.0 / K
                else:
                    row /= s
                S[row_idx, :] = row

        # Fill remaining rows via the public policy interface to support subclasses that don't populate .probs.
        for j, hand in enumerate(self.hands):
            if not np.isnan(S[j, 0]):
                continue
            iset = InfoSet(pid=pid, hand=hand, history=history)
            dist = policy.prob_dist_at_infoset(iset)
            row = np.asarray([float(dist.get(a, 0.0)) for a in actions], dtype=float)
            s = float(row.sum())
            if s <= 0.0:
                row[:] = 1.0 / K
            else:
                row /= s
            S[j, :] = row

        self._strategy_cache[cache_key] = S
        return S

    def _solve_terminal(
        self,
        history: Tuple[int, ...],
        opp_reach: np.ndarray,
        *,
        to_play: int,
        my_id: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Terminal history ends with CALL. Returns (win_mass, mass) per our hand."""

        if not history or history[-1] != CALL:
            raise ValueError("Terminal solver called on non-terminal history.")

        last_claim_idx = InfoSet.last_claim_idx(history[:-1])
        if last_claim_idx == NO_CLAIM:
            raise ValueError("CALL cannot resolve without a preceding claim.")

        req = self._claim_reqs[last_claim_idx]
        r = req.rank
        need = req.need

        # Determine who called (last action), using to_play (player who would act next).
        caller = 1 - to_play

        # Truth matrix T_ij = (count_i + count_j >= need)
        c = self.hand_rank_counts[:, r]
        T = (c[:, None] + c[None, :]) >= need

        # Mass for each of our hands i:
        #   M_i = sum_j A_ij * opp_reach_j
        M = self.card_removal_mat @ opp_reach

        # Satisfied mass:
        sat_mass = (self.card_removal_mat * T) @ opp_reach

        # If opponent called, we win if claim true; if we called, we win if claim false.
        if caller != my_id:
            WM = sat_mass
        else:
            WM = M - sat_mass

        # Store baseline-compatible terminal state values only when this is an opponent call,
        # i.e., when the next-to-play equals my_id (matches baseline storage behavior).
        if to_play == my_id:
            values = np.where(M > 0.0, WM / M, 0.0)
            self.state_card_values[history] = {self.hands[i]: float(values[i]) for i in range(self.n_hands)}

        return WM, M

    def _solve_node(
        self,
        history: Tuple[int, ...],
        opp_reach: np.ndarray,
        *,
        to_play: int,
        my_id: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if history and history[-1] == CALL:
            return self._solve_terminal(history, opp_reach, to_play=to_play, my_id=my_id)

        actions = self._legal_actions_from_history(history)
        next_to_play = 1 - to_play

        # Opponent node: integrate over opponent actions per opponent hand.
        if to_play != my_id:
            S = self._get_strategy_matrix(self.opponent, to_play, history, actions)
            total_wm = np.zeros(self.n_hands, dtype=float)
            total_m = np.zeros(self.n_hands, dtype=float)
            for col, a in enumerate(actions):
                new_hist = history + (a,)
                new_reach = opp_reach * S[:, col]
                wm, m = self._solve_node(new_hist, new_reach, to_play=next_to_play, my_id=my_id)
                total_wm += wm
                total_m += m
            return total_wm, total_m

        # Our node: evaluate each action, pick best per hand with zero-mass safety.
        wm_list: List[np.ndarray] = []
        m_list: List[np.ndarray] = []
        for a in actions:
            wm, m = self._solve_node(history + (a,), opp_reach, to_play=next_to_play, my_id=my_id)
            wm_list.append(wm)
            m_list.append(m)

        wm_mat = np.stack(wm_list, axis=0)
        m_mat = np.stack(m_list, axis=0)

        with np.errstate(divide="ignore", invalid="ignore"):
            V = np.where(m_mat > 0.0, wm_mat / m_mat, -np.inf)
        best_idx = np.argmax(V, axis=0)

        arange = np.arange(self.n_hands)
        best_wm = wm_mat[best_idx, arange]
        best_m = m_mat[best_idx, arange]

        # Store state values at this decision node (baseline-compatible)
        values = np.where(best_m > 0.0, best_wm / best_m, 0.0)
        self.state_card_values[history] = {self.hands[i]: float(values[i]) for i in range(self.n_hands)}

        # Store deterministic best action for each infoset (one loop over hands)
        for i, a_idx in enumerate(best_idx):
            best_action = actions[int(a_idx)]
            iset = InfoSet(pid=my_id, hand=self.hands[i], history=history)
            self.probs[iset] = {best_action: 1.0}

        return best_wm, best_m

    def solve(self) -> None:
        """Compute best response for both seats (pid=0 and pid=1)."""

        self.opponent.bind_rules(self.rules)
        root_reach = np.ones(self.n_hands, dtype=float)

        # As P1 (pid=0): our turn at root.
        wm0, m0 = self._solve_node(tuple(), root_reach, to_play=0, my_id=0)
        self._root_values[0] = np.where(m0 > 0.0, wm0 / m0, 0.0)

        # As P2 (pid=1): opponent acts first at root.
        wm1, m1 = self._solve_node(tuple(), root_reach, to_play=0, my_id=1)
        self._root_values[1] = np.where(m1 > 0.0, wm1 / m1, 0.0)

    def exploitability(self) -> Tuple[float, float]:
        """Return (p_first, p_second) win probabilities for our BR as P1 and as P2."""

        if 0 not in self._root_values or 1 not in self._root_values:
            raise RuntimeError("Must call solve() before exploitability().")

        w = self.hand_weights
        den = float(w.sum())
        if den <= 0.0:
            return 0.0, 0.0

        p_first = float(np.dot(w, self._root_values[0]) / den)
        p_second = float(np.dot(w, self._root_values[1]) / den)
        return p_first, p_second


def best_response_exact(
    spec: GameSpec,
    policy: Policy,
    debug: bool = False,
) -> Tuple[TabularPolicy, BestResponseComputerEfficient]:
    """Vectorized exact best response (candidate replacement for br_exact.best_response_exact)."""

    br = BestResponseComputerEfficient(spec, policy)
    if debug:
        print("Solving (efficient exact BR)...")
    br.solve()
    if debug:
        p1, p2 = br.exploitability()
        print(f"Done. exploitability(first={p1:.6f}, second={p2:.6f})")

    out = TabularPolicy()
    out.probs = br.probs
    return out, br


# Convenience alias for callers that prefer a distinct name.
best_response_efficient = best_response_exact
