from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from liars_poker.core import GameSpec, card_rank, possible_starting_hands
from liars_poker.env import Rules, rules_for_spec
from liars_poker.infoset import CALL
from liars_poker.policies.tabular_dense import DenseTabularPolicy
from liars_poker.algo.br_exact_dense_to_dense import adjustment_factor


def _highest_set_bit(hid: int) -> int:
    if hid <= 0:
        return -1
    return hid.bit_length() - 1


@dataclass(slots=True)
class _ClaimReq:
    kind: str
    rank1: int
    rank2: int
    need: int


class CFRExactDense:
    """Exact CFR on dense public states with blocker-aware utilities."""

    def __init__(self, spec: GameSpec, *, dtype: np.dtype = np.float64) -> None:
        self.spec = spec
        self.rules: Rules = rules_for_spec(spec)
        self.dtype = dtype

        self.k = len(self.rules.claims)
        self.hands = tuple(possible_starting_hands(spec))
        self.hand_to_idx = {hand: idx for idx, hand in enumerate(self.hands)}
        self.n_hands = len(self.hands)

        self.H = 1 << self.k
        self.A = self.k + 1

        self.popcount = np.zeros(self.H, dtype=np.int16)
        self.last_claim = np.full(self.H, -1, dtype=np.int16)
        self.legal_mask = np.zeros((self.H, self.A), dtype=bool)
        self.legal_counts = np.zeros(self.H, dtype=np.int16)
        self.legal_actions: List[Tuple[int, ...]] = []
        self.legal_cols: List[Tuple[int, ...]] = []
        self.uniform_rows = np.zeros((self.H, self.A), dtype=self.dtype)

        for hid in range(self.H):
            self.popcount[hid] = int(hid.bit_count())
            if hid:
                self.last_claim[hid] = _highest_set_bit(hid)
                last_idx = int(self.last_claim[hid])
            else:
                last_idx = None

            legal = self.rules.legal_actions_from_last(last_idx)
            self.legal_actions.append(legal)
            cols = tuple(0 if action == CALL else action + 1 for action in legal)
            self.legal_cols.append(cols)
            if cols:
                self.legal_mask[hid, list(cols)] = True
                self.legal_counts[hid] = len(cols)
                self.uniform_rows[hid, list(cols)] = 1.0 / float(len(cols))

        self.hids_by_popcount: List[List[int]] = [[] for _ in range(self.k + 1)]
        for hid in range(self.H):
            self.hids_by_popcount[int(self.popcount[hid])].append(hid)

        self.hand_rank_counts = self._build_hand_rank_counts()
        self._claim_reqs = self._build_claim_requirements()
        self._truth_mats = self._build_truth_matrices()

        self.A0, self.A1 = self._build_blocker_matrices()

        self.S = np.zeros((self.H, self.n_hands, self.A), dtype=self.dtype)
        for hid in range(self.H):
            if self.legal_counts[hid] > 0:
                self.S[hid, :, :] = self.uniform_rows[hid]

        self.R0 = np.zeros((self.H, self.n_hands, self.A), dtype=self.dtype)
        self.R1 = np.zeros((self.H, self.n_hands, self.A), dtype=self.dtype)
        self.SS0 = np.zeros((self.H, self.n_hands, self.A), dtype=self.dtype)
        self.SS1 = np.zeros((self.H, self.n_hands, self.A), dtype=self.dtype)

        self.L0 = np.ones((self.H, self.n_hands), dtype=self.dtype)
        self.L1 = np.ones((self.H, self.n_hands), dtype=self.dtype)
        self._recompute_likelihoods()

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
                low, high = self.rules.two_pair_ranks[rank_value]
                reqs.append(_ClaimReq(kind=kind, rank1=low, rank2=high, need=2))
            elif kind == "FullHouse":
                trip, pair = self.rules.full_house_ranks[rank_value]
                reqs.append(_ClaimReq(kind=kind, rank1=trip, rank2=pair, need=0))
            elif kind == "Quads":
                reqs.append(_ClaimReq(kind=kind, rank1=rank_value, rank2=0, need=4))
            else:
                raise ValueError(f"Unsupported claim kind: {kind}")
        return reqs

    def _build_hand_rank_counts(self) -> np.ndarray:
        counts = np.zeros((self.n_hands, self.spec.ranks + 1), dtype=np.int16)
        for i, hand in enumerate(self.hands):
            for card in hand:
                r = card_rank(card, self.spec)
                counts[i, r] += 1
        return counts

    def _build_truth_matrices(self) -> List[np.ndarray]:
        mats: List[np.ndarray] = []
        for req in self._claim_reqs:
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
                T = (c[:, None] + c[None, :] >= req.need)
            mats.append(T)
        return mats

    def _build_blocker_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        A0 = np.zeros((self.n_hands, self.n_hands), dtype=self.dtype)
        A1 = np.zeros((self.n_hands, self.n_hands), dtype=self.dtype)
        for i, my_hand in enumerate(self.hands):
            for j, opp_hand in enumerate(self.hands):
                A0[i, j] = adjustment_factor(self.spec, my_hand, opp_hand)
                A1[j, i] = adjustment_factor(self.spec, opp_hand, my_hand)
        return A0, A1

    def _update_strategy(self) -> None:
        for hid in range(self.H):
            if self.legal_counts[hid] <= 0:
                continue
            to_play = int(self.popcount[hid] & 1)
            if to_play == 0:
                regrets = self.R0[hid]
            else:
                regrets = self.R1[hid]

            pos = np.maximum(regrets, 0.0)
            pos *= self.legal_mask[hid]
            sums = pos.sum(axis=1)

            self.S[hid, :, :] = self.uniform_rows[hid]
            use_pos = sums > 0.0
            if np.any(use_pos):
                self.S[hid, use_pos, :] = pos[use_pos] / sums[use_pos][:, None]

    def _recompute_likelihoods(self) -> None:
        self.L0.fill(1.0)
        self.L1.fill(1.0)
        for hid in range(1, self.H):
            c = int(self.last_claim[hid])
            if c < 0:
                continue
            prev = hid ^ (1 << c)
            pid_maker = int(self.popcount[prev] & 1)
            col = c + 1
            if pid_maker == 0:
                self.L0[hid] = self.L0[prev] * self.S[prev, :, col]
                self.L1[hid] = self.L1[prev]
            else:
                self.L1[hid] = self.L1[prev] * self.S[prev, :, col]
                self.L0[hid] = self.L0[prev]

    def _terminal_utility(
        self,
        hid: int,
        opp_reach: np.ndarray,
        *,
        caller: int,
        player: int,
    ) -> np.ndarray:
        if hid == 0:
            raise ValueError("CALL cannot occur without a preceding claim.")

        claim_idx = int(self.last_claim[hid])
        T = self._truth_mats[claim_idx]
        if player == 0:
            A = self.A0
        else:
            A = self.A1

        M = A @ opp_reach
        sat_mass = (A * T) @ opp_reach
        if caller != player:
            win_mass = sat_mass
        else:
            win_mass = M - sat_mass
        return 2.0 * win_mass - M

    def _update_player(self, player: int) -> np.ndarray:
        V = np.zeros((self.H, self.n_hands), dtype=self.dtype)
        Lp = self.L0 if player == 0 else self.L1
        Lopp = self.L1 if player == 0 else self.L0
        R = self.R0 if player == 0 else self.R1
        SS = self.SS0 if player == 0 else self.SS1

        for pc in range(self.k, -1, -1):
            for hid in self.hids_by_popcount[pc]:
                actions = self.legal_actions[hid]
                if not actions:
                    continue
                to_play = int(self.popcount[hid] & 1)

                if to_play == player:
                    cols = self.legal_cols[hid]
                    action_vals = np.zeros((len(actions), self.n_hands), dtype=self.dtype)
                    for idx, action in enumerate(actions):
                        if action == CALL:
                            action_vals[idx] = self._terminal_utility(
                                hid,
                                Lopp[hid],
                                caller=to_play,
                                player=player,
                            )
                        else:
                            child = hid | (1 << action)
                            action_vals[idx] = V[child]

                    sigma = self.S[hid, :, cols]
                    V_state = np.sum(action_vals * sigma, axis=0)

                    R[hid, :, cols] += (action_vals - V_state[None, :])
                    SS[hid, :, cols] += Lp[hid][None, :] * sigma
                    V[hid] = V_state
                else:
                    total = np.zeros(self.n_hands, dtype=self.dtype)
                    for action in actions:
                        if action == CALL:
                            call_prob = self.S[hid, :, 0]
                            opp_reach = Lopp[hid] * call_prob
                            total += self._terminal_utility(
                                hid,
                                opp_reach,
                                caller=to_play,
                                player=player,
                            )
                        else:
                            child = hid | (1 << action)
                            total += V[child]
                    V[hid] = total

        return V

    def iterate(self) -> None:
        self._update_strategy()
        self._recompute_likelihoods()
        self._update_player(0)
        self._update_player(1)

    def average_policy(self) -> DenseTabularPolicy:
        policy = DenseTabularPolicy(self.spec)

        for hid in range(self.H):
            if self.legal_counts[hid] <= 0:
                continue
            to_play = int(self.popcount[hid] & 1)
            source = self.SS0 if to_play == 0 else self.SS1
            rows = source[hid] * self.legal_mask[hid]
            sums = rows.sum(axis=1)

            policy.S[hid, :, :] = self.uniform_rows[hid]
            use_rows = sums > 0.0
            if np.any(use_rows):
                policy.S[hid, use_rows, :] = rows[use_rows] / sums[use_rows][:, None]

        policy.recompute_likelihoods()
        return policy
