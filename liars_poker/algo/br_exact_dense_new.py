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


class BestResponseComputerDenseIter:
    """Exact best response against a DenseTabularPolicy using iterative DP over hid."""

    def __init__(self, spec: GameSpec, opponent: DenseTabularPolicy, *, store_state_values: bool = True):
        if not isinstance(opponent, DenseTabularPolicy):
            raise TypeError("BestResponseComputerDenseIter requires a DenseTabularPolicy opponent.")
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

        self.card_removal_mat = self._build_card_removal_matrix()
        self.hand_weights = np.asarray([adjustment_factor(spec, tuple(), h) for h in self.hands], dtype=float)
        self.hand_rank_counts = self._build_hand_rank_counts()

        self.state_card_values: Dict[Tuple[int, ...], Dict[Tuple[int, ...], float]] = {}
        self._root_values: Dict[int, np.ndarray] = {}

        self.br_policy = DenseTabularPolicy(spec)
        self._history_by_hid = self._build_histories() if store_state_values else []
        self._claim_reqs = self._build_claim_requirements()
        self._hids_by_popcount = self._group_hids_by_popcount()

    def _build_claim_requirements(self) -> List[_ClaimReq]:
        reqs: List[_ClaimReq] = []
        for kind, rank_value in self.rules.claims:
            if kind == "RankHigh":
                reqs.append(_ClaimReq(kind=kind, rank1=rank_value, rank2=0, need=1))
            elif kind == "Pair":
                reqs.append(_ClaimReq(kind=kind, rank1=rank_value, rank2=0, need=2))
            elif kind == "Trips":
                reqs.append(_ClaimReq(kind=kind, rank1=rank_value, rank2=0, need=3))
            elif kind == "TwoPair":
                low, high = self._two_pair_ranks[rank_value]
                reqs.append(_ClaimReq(kind=kind, rank1=low, rank2=high, need=2))
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

    def _group_hids_by_popcount(self) -> List[List[int]]:
        H = 1 << self.k
        groups: List[List[int]] = [[] for _ in range(self.k + 1)]
        pop = self.opponent.popcount
        for hid in range(H):
            groups[int(pop[hid])].append(hid)
        return groups

    def _opp_reach_table(self, my_id: int) -> np.ndarray:
        return self.opponent.L_pid1 if my_id == 0 else self.opponent.L_pid0

    def _build_truth_matrix(self, req: _ClaimReq) -> np.ndarray:
        if req.kind == "TwoPair":
            c1 = self.hand_rank_counts[:, req.rank1]
            c2 = self.hand_rank_counts[:, req.rank2]
            return (c1[:, None] + c1[None, :] >= req.need) & (c2[:, None] + c2[None, :] >= req.need)
        c = self.hand_rank_counts[:, req.rank1]
        return (c[:, None] + c[None, :] >= req.need)

    def _compute_call_layer(
        self,
        curr_hids: List[int],
        *,
        to_play: int,
        my_id: int,
        opp_reach_table: np.ndarray,
        S: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        n = len(curr_hids)
        call_wm = np.zeros((n, self.n_hands), dtype=float)
        call_m = np.zeros((n, self.n_hands), dtype=float)
        if to_play == 0 and 0 in curr_hids:
            pass
        if to_play == 0 and len(curr_hids) == 1 and curr_hids[0] == 0:
            return call_wm, call_m

        caller = to_play
        store_terminal = self.store_state_values and caller != my_id

        by_claim: Dict[int, List[int]] = {}
        last_claim = self.opponent.last_claim
        for pos, hid in enumerate(curr_hids):
            if hid == 0:
                continue
            claim = int(last_claim[hid])
            by_claim.setdefault(claim, []).append(pos)

        for claim_idx, positions in by_claim.items():
            req = self._claim_reqs[claim_idx]
            T = self._build_truth_matrix(req)

            hids = [curr_hids[p] for p in positions]
            R = opp_reach_table[hids]
            if caller != my_id:
                R = R * S[hids, :, 0]
            R_T = R.T

            M = self.card_removal_mat @ R_T
            sat = np.einsum("ij,ij,jb->ib", self.card_removal_mat, T, R_T, optimize=True)

            if caller != my_id:
                WM = sat
            else:
                WM = M - sat

            call_m[positions] = M.T
            call_wm[positions] = WM.T

            if store_terminal:
                values = np.where(M > 0.0, WM / M, 0.0).T
                for pos, row in zip(positions, values):
                    hist = self._history_by_hid[curr_hids[pos]] + (CALL,)
                    self.state_card_values[hist] = {self.hands[i]: float(row[i]) for i in range(self.n_hands)}

        return call_wm, call_m

    def _solve_for_player(self, my_id: int) -> Tuple[np.ndarray, np.ndarray]:
        wm_next: np.ndarray | None = None
        m_next: np.ndarray | None = None
        next_hids: List[int] = []
        idx_next: Dict[int, int] = {}

        opp_reach_table = self._opp_reach_table(my_id)
        S = self.opponent.S
        legal_actions = self.opponent.legal_actions

        root_wm = None
        root_m = None

        for pop in range(self.k, -1, -1):
            curr_hids = self._hids_by_popcount[pop]
            curr_len = len(curr_hids)
            if curr_len == 0:
                continue

            wm_curr = np.zeros((curr_len, self.n_hands), dtype=float)
            m_curr = np.zeros((curr_len, self.n_hands), dtype=float)
            idx_curr = {hid: i for i, hid in enumerate(curr_hids)}

            if pop < self.k:
                idx_next = {hid: i for i, hid in enumerate(next_hids)}

            to_play = pop & 1
            call_wm, call_m = self._compute_call_layer(
                curr_hids,
                to_play=to_play,
                my_id=my_id,
                opp_reach_table=opp_reach_table,
                S=S,
            )

            for pos, hid in enumerate(curr_hids):
                actions = legal_actions[hid]
                if not actions:
                    continue

                if to_play != my_id:
                    total_wm = np.zeros(self.n_hands, dtype=float)
                    total_m = np.zeros(self.n_hands, dtype=float)
                    for action in actions:
                        if action == CALL:
                            wm_a = call_wm[pos]
                            m_a = call_m[pos]
                        else:
                            child = hid | (1 << action)
                            child_idx = idx_next[child]
                            wm_a = wm_next[child_idx]
                            m_a = m_next[child_idx]
                        total_wm += wm_a
                        total_m += m_a
                    wm_curr[pos] = total_wm
                    m_curr[pos] = total_m
                    continue

                wm_list: List[np.ndarray] = []
                m_list: List[np.ndarray] = []
                for action in actions:
                    if action == CALL:
                        wm_a = call_wm[pos]
                        m_a = call_m[pos]
                    else:
                        child = hid | (1 << action)
                        child_idx = idx_next[child]
                        wm_a = wm_next[child_idx]
                        m_a = m_next[child_idx]
                    wm_list.append(wm_a)
                    m_list.append(m_a)

                wm_mat = np.stack(wm_list, axis=0)
                m_mat = np.stack(m_list, axis=0)
                with np.errstate(divide="ignore", invalid="ignore"):
                    V = np.where(m_mat > 0.0, wm_mat / m_mat, -np.inf)
                best_idx = np.argmax(V, axis=0)

                arange = np.arange(self.n_hands)
                best_wm = wm_mat[best_idx, arange]
                best_m = m_mat[best_idx, arange]

                wm_curr[pos] = best_wm
                m_curr[pos] = best_m

                if self.store_state_values:
                    values = np.where(best_m > 0.0, best_wm / best_m, 0.0)
                    hist = self._history_by_hid[hid]
                    self.state_card_values[hist] = {self.hands[i]: float(values[i]) for i in range(self.n_hands)}

                for i, a_idx in enumerate(best_idx):
                    action = actions[int(a_idx)]
                    col = 0 if action == CALL else action + 1
                    self.br_policy.S[hid, i, :] = 0.0
                    self.br_policy.S[hid, i, col] = 1.0

            if pop == 0:
                idx0 = idx_curr.get(0)
                if idx0 is not None:
                    root_wm = wm_curr[idx0]
                    root_m = m_curr[idx0]

            next_hids = curr_hids
            wm_next = wm_curr
            m_next = m_curr

        if root_wm is None or root_m is None:
            raise RuntimeError("Root values not computed.")
        return root_wm, root_m

    def solve(self) -> None:
        wm0, m0 = self._solve_for_player(0)
        self._root_values[0] = np.where(m0 > 0.0, wm0 / m0, 0.0)

        wm1, m1 = self._solve_for_player(1)
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
) -> Tuple[DenseTabularPolicy, Dict]:
    """Dense exact best response using iterative DP and L_pid lookup."""

    br = BestResponseComputerDenseIter(spec, policy, store_state_values=store_state_values)
    if debug:
        print("Solving (dense-to-dense exact BR, iterative)...")
    br.solve()
    if debug:
        p1, p2 = br.exploitability()
        print(f"Done. exploitability(first={p1:.6f}, second={p2:.6f}, avg={(p1+p2)/2:.6f})")

    dict_log = {"computes_exploitability": True, "computer": br}

    return br.br_policy, dict_log


best_response_dense = best_response_exact
