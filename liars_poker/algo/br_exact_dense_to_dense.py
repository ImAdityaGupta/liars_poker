from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from liars_poker.core import GameSpec, card_rank, generate_deck
from liars_poker.env import Rules, rules_for_spec
from liars_poker.infoset import CALL
from liars_poker.policies.tabular_dense import DenseTabularPolicy


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
    kind: str
    rank1: int
    rank2: int
    need: int


class BestResponseComputerDense:
    """Exact best response against a DenseTabularPolicy using dense tensors only."""

    def __init__(self, spec: GameSpec, opponent: DenseTabularPolicy, *, store_state_values: bool = True):
        if not isinstance(opponent, DenseTabularPolicy):
            raise TypeError("BestResponseComputerDense requires a DenseTabularPolicy opponent.")
        if opponent.spec != spec:
            raise ValueError("DenseTabularPolicy spec mismatch.")

        self.spec = spec
        self.rules: Rules = rules_for_spec(spec)
        self.opponent = opponent
        self.store_state_values = store_state_values

        self.k = opponent.k
        self.hands = opponent.hands
        self.n_hands = len(self.hands)
        self._two_pair_ranks = self.rules.two_pair_ranks
        self._full_house_ranks = self.rules.full_house_ranks

        self.card_removal_mat = self._build_card_removal_matrix()
        self.hand_weights = np.asarray([adjustment_factor(spec, tuple(), h) for h in self.hands], dtype=float)
        self.hand_rank_counts = self._build_hand_rank_counts()

        self.state_card_values: Dict[Tuple[int, ...], Dict[Tuple[int, ...], float]] = {}
        self._root_values: Dict[int, np.ndarray] = {}

        self.br_policy = DenseTabularPolicy(spec)
        if self.store_state_values:
            self._history_by_hid = self._build_histories()
        self._claim_reqs = self._build_claim_requirements()

    def _build_claim_requirements(self) -> List[_ClaimReq]:
        reqs: List[_ClaimReq] = []
        for idx, (kind, rank_value) in enumerate(self.rules.claims):
            if kind == "RankHigh":
                reqs.append(_ClaimReq(kind=kind, rank1=rank_value, rank2=0, need=1))
            elif kind == "Pair":
                reqs.append(_ClaimReq(kind=kind, rank1=rank_value, rank2=0, need=2))
            elif kind == "Trips":
                reqs.append(_ClaimReq(kind=kind, rank1=rank_value, rank2=0, need=3))
            elif kind == "TwoPair":
                low, high = self._two_pair_ranks[rank_value]
                reqs.append(_ClaimReq(kind=kind, rank1=low, rank2=high, need=2))
            elif kind == "FullHouse":
                trip, pair = self._full_house_ranks[rank_value]
                reqs.append(_ClaimReq(kind=kind, rank1=trip, rank2=pair, need=0))
            elif kind == "Quads":
                reqs.append(_ClaimReq(kind=kind, rank1=rank_value, rank2=0, need=4))
            else:
                raise ValueError(f"Unsupported claim kind: {kind}")
        return reqs

    def _build_histories(self) -> List[Tuple[int, ...]]:
        H = 1 << self.k
        return [tuple(i for i in range(self.k) if (hid >> i) & 1) for hid in range(H)]

    def _build_hand_rank_counts(self) -> np.ndarray:
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

    def _solve_terminal(
        self,
        hid: int,
        opp_reach: np.ndarray,
        *,
        caller: int,
        my_id: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if hid == 0:
            raise ValueError("CALL cannot occur without a preceding claim.")

        last_claim_idx = int(self.opponent.last_claim[hid])
        req = self._claim_reqs[last_claim_idx]
        if req.kind == "TwoPair":
            c1 = self.hand_rank_counts[:, req.rank1]
            c2 = self.hand_rank_counts[:, req.rank2]
            T = (c1[:, None] + c1[None, :] >= req.need) & (c2[:, None] + c2[None, :] >= req.need)
        elif req.kind == "FullHouse":
            c3 = self.hand_rank_counts[:, req.rank1]
            c2 = self.hand_rank_counts[:, req.rank2]
            T = (c3[:, None] + c3[None, :] >= 3) & (c2[:, None] + c2[None, :] >= 2)
        else:
            c = self.hand_rank_counts[:, req.rank1]
            T = (c[:, None] + c[None, :]) >= req.need

        M = self.card_removal_mat @ opp_reach
        sat_mass = (self.card_removal_mat * T) @ opp_reach

        if caller != my_id:
            WM = sat_mass
        else:
            WM = M - sat_mass

        if caller != my_id and self.store_state_values:
            values = np.where(M > 0.0, WM / M, 0.0)
            terminal_hist = self._history_by_hid[hid] + (CALL,)
            self.state_card_values[terminal_hist] = {self.hands[i]: float(values[i]) for i in range(self.n_hands)}

        return WM, M

    def _solve_node(
        self,
        hid: int,
        opp_reach: np.ndarray,
        *,
        to_play: int,
        my_id: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        actions = self.opponent.legal_actions[hid]
        if not actions:
            zeros = np.zeros(self.n_hands, dtype=float)
            return zeros, zeros

        next_to_play = 1 - to_play

        if to_play != my_id:
            S = self.opponent.S[hid]
            total_wm = np.zeros(self.n_hands, dtype=float)
            total_m = np.zeros(self.n_hands, dtype=float)
            for action in actions:
                if action == CALL:
                    new_reach = opp_reach * S[:, 0]
                    wm, m = self._solve_terminal(hid, new_reach, caller=to_play, my_id=my_id)
                else:
                    col = action + 1
                    new_reach = opp_reach * S[:, col]
                    new_hid = hid | (1 << action)
                    wm, m = self._solve_node(new_hid, new_reach, to_play=next_to_play, my_id=my_id)
                total_wm += wm
                total_m += m
            return total_wm, total_m

        wm_list: List[np.ndarray] = []
        m_list: List[np.ndarray] = []
        for action in actions:
            if action == CALL:
                wm, m = self._solve_terminal(hid, opp_reach, caller=to_play, my_id=my_id)
            else:
                new_hid = hid | (1 << action)
                wm, m = self._solve_node(new_hid, opp_reach, to_play=next_to_play, my_id=my_id)
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

        if self.store_state_values:
            values = np.where(best_m > 0.0, best_wm / best_m, 0.0)
            hist = self._history_by_hid[hid]
            self.state_card_values[hist] = {self.hands[i]: float(values[i]) for i in range(self.n_hands)}

        for i, a_idx in enumerate(best_idx):
            action = actions[int(a_idx)]
            col = 0 if action == CALL else action + 1
            self.br_policy.S[hid, i, :] = 0.0
            self.br_policy.S[hid, i, col] = 1.0

        return best_wm, best_m

    def solve(self) -> None:
        root_reach = np.ones(self.n_hands, dtype=float)

        wm0, m0 = self._solve_node(0, root_reach, to_play=0, my_id=0)
        self._root_values[0] = np.where(m0 > 0.0, wm0 / m0, 0.0)

        wm1, m1 = self._solve_node(0, root_reach, to_play=0, my_id=1)
        self._root_values[1] = np.where(m1 > 0.0, wm1 / m1, 0.0)

        self.br_policy.recompute_likelihoods()

    def exploitability(self) -> Tuple[float, float]:
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
    policy: DenseTabularPolicy,
    debug: bool = False,
    *,
    store_state_values: bool = True,
) -> Tuple[DenseTabularPolicy, BestResponseComputerDense]:
    """Dense exact best response against a DenseTabularPolicy opponent."""

    br = BestResponseComputerDense(spec, policy, store_state_values=store_state_values)
    if debug:
        print("Solving (dense-to-dense exact BR)...")
    br.solve()
    if debug:
        p1, p2 = br.exploitability()
        print(f"Done. exploitability(first={p1:.6f}, second={p2:.6f}, avg={(p1+p2)/2:.6f})")

    dict_log = {"computes_exploitability": True, "computer": br}

    return br.br_policy, dict_log


best_response_dense = best_response_exact
