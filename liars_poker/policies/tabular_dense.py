from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Sequence

import numpy as np

from liars_poker.core import GameSpec, possible_starting_hands
from liars_poker.env import Rules, rules_for_spec
from liars_poker.infoset import CALL, InfoSet

from .base import Policy



def _highest_set_bit(hid: int) -> int:
    if hid <= 0:
        return -1
    return hid.bit_length() - 1


def _history_to_hid(history: Tuple[int, ...], k: int) -> int:
    hid = 0
    last = -1
    for action in history:
        if action == CALL:
            raise ValueError("InfoSet history cannot include CALL.")
        if action < 0 or action >= k:
            raise ValueError(f"Invalid claim id {action} for k={k}.")
        if action <= last:
            raise ValueError("Claim history must be strictly increasing.")
        hid |= 1 << action
        last = action
    return hid


@dataclass(slots=True)
class DenseTabularPolicy(Policy):
    """Dense policy representation over (hid, hand, action)."""

    POLICY_KIND = "DenseTabularPolicy"
    POLICY_VERSION = 1

    spec: GameSpec
    rules: Rules
    k: int
    hands: Tuple[Tuple[int, ...], ...]
    hand_to_idx: Dict[Tuple[int, ...], int]
    S: np.ndarray
    popcount: np.ndarray
    last_claim: np.ndarray
    legal_mask: np.ndarray
    legal_counts: np.ndarray
    legal_actions: List[Tuple[int, ...]]
    L_pid0: np.ndarray
    L_pid1: np.ndarray

    def __init__(self, spec: GameSpec) -> None:
        super().__init__()
        self.spec = spec
        self.rules = rules_for_spec(spec)
        self.bind_rules(self.rules)

        self.k = len(self.rules.claims)
        self.hands = tuple(possible_starting_hands(spec))
        self.hand_to_idx = {hand: idx for idx, hand in enumerate(self.hands)}

        H = 1 << self.k
        N = len(self.hands)
        A = self.k + 1

        self.popcount = np.zeros(H, dtype=np.int16)
        self.last_claim = np.full(H, -1, dtype=np.int16)
        self.legal_mask = np.zeros((H, A), dtype=bool)
        self.legal_counts = np.zeros(H, dtype=np.int16)
        self.legal_actions = []

        for hid in range(H):
            self.popcount[hid] = int(hid.bit_count())
            if hid:
                self.last_claim[hid] = _highest_set_bit(hid)
                last_idx = int(self.last_claim[hid])
            else:
                last_idx = None

            legal = self.rules.legal_actions_from_last(last_idx)
            self.legal_actions.append(legal)
            cols = [0 if action == CALL else action + 1 for action in legal]
            if cols:
                self.legal_mask[hid, cols] = True
                self.legal_counts[hid] = len(cols)

        self.S = np.zeros((H, N, A), dtype=np.float32)
        for hid in range(H):
            count = int(self.legal_counts[hid])
            if count <= 0:
                continue
            cols = np.flatnonzero(self.legal_mask[hid])
            prob = 1.0 / count
            self.S[hid, :, cols] = prob

        self.L_pid0 = np.ones((H, N), dtype=np.float32)
        self.L_pid1 = np.ones((H, N), dtype=np.float32)
        self.recompute_likelihoods()

    def pid_to_act(self, hid: int) -> int:
        return int(self.popcount[hid] & 1)

    def recompute_likelihoods(self) -> None:
        H = self.S.shape[0]
        self.L_pid0.fill(1.0)
        self.L_pid1.fill(1.0)
        for hid in range(1, H):
            c = int(self.last_claim[hid])
            if c < 0:
                continue
            prev = hid ^ (1 << c)
            pid_maker = int(self.popcount[prev] & 1)
            col = c + 1
            if pid_maker == 0:
                self.L_pid0[hid] = self.L_pid0[prev] * self.S[prev, :, col]
                self.L_pid1[hid] = self.L_pid1[prev]
            else:
                self.L_pid1[hid] = self.L_pid1[prev] * self.S[prev, :, col]
                self.L_pid0[hid] = self.L_pid0[prev]

    def action_probs(self, infoset: InfoSet) -> Dict[int, float]:
        hid = _history_to_hid(infoset.history, self.k)
        expected_pid = self.pid_to_act(hid)
        if infoset.pid != expected_pid:
            raise ValueError(
                f"InfoSet pid={infoset.pid} does not match expected pid={expected_pid} for hid={hid}."
            )

        hand_idx = self.hand_to_idx.get(infoset.hand)
        if hand_idx is None:
            raise ValueError(f"Unknown hand in infoset: {infoset.hand}")

        legal = self.legal_actions[hid]
        if not legal:
            return {}

        row = self.S[hid, hand_idx]
        return {action: float(row[0 if action == CALL else action + 1]) for action in legal}

    def sample_action_fast(
        self,
        *,
        pid: int,
        hand: Tuple[int, ...],
        history: Tuple[int, ...],
        legal: Tuple[int, ...],
        rng: random.Random,
    ) -> int:
        _ = pid
        if not legal:
            raise ValueError("Cannot sample from empty policy distribution.")

        hid = _history_to_hid(history, self.k)
        hand_idx = self.hand_to_idx.get(hand)
        if hand_idx is None:
            raise ValueError(f"Unknown hand in infoset: {hand}")

        row = self.S[hid, hand_idx]
        total = 0.0
        for action in legal:
            col = 0 if action == CALL else action + 1
            total += float(row[col])
        if total <= 0.0:
            return legal[rng.randrange(len(legal))]

        pick = rng.random() * total
        cumulative = 0.0
        last_action = legal[-1]
        for action in legal:
            col = 0 if action == CALL else action + 1
            cumulative += float(row[col])
            if pick <= cumulative:
                return action
        return last_action

    def __repr__(self) -> str:
        return f"DenseTabularPolicy(spec={self.spec}, k={self.k}, hands={len(self.hands)})"

    # --- Serialization ---

    @staticmethod
    def _encode_spec(spec: GameSpec) -> Dict[str, object]:
        return {
            "ranks": spec.ranks,
            "suits": spec.suits,
            "hand_size": spec.hand_size,
            "claim_kinds": list(spec.claim_kinds),
            "suit_symmetry": spec.suit_symmetry,
        }

    @staticmethod
    def _decode_spec(d: Dict[str, object]) -> GameSpec:
        return GameSpec(
            ranks=int(d["ranks"]),
            suits=int(d["suits"]),
            hand_size=int(d["hand_size"]),
            claim_kinds=tuple(d["claim_kinds"]),
            suit_symmetry=bool(d["suit_symmetry"]),
        )

    def to_payload(self) -> Tuple[Dict, Dict[str, object]]:
        payload = {
            "kind": self.POLICY_KIND,
            "version": self.POLICY_VERSION,
            "counts": {
                "k": self.k,
                "H": int(self.S.shape[0]),
                "N": int(self.S.shape[1]),
                "A": int(self.S.shape[2]),
            },
            "dtype": str(self.S.dtype),
            "spec": self._encode_spec(self.spec),
        }
        blobs: Dict[str, object] = {
            "S": np.asarray(self.S, dtype=np.float32),
        }
        return payload, blobs

    @classmethod
    def from_payload(
        cls,
        payload: Dict,
        *,
        blob_prefix: str,
        blobs: Dict[str, object],
        children,
    ) -> "DenseTabularPolicy":
        _ = (blob_prefix, children)

        kind = payload.get("kind", cls.POLICY_KIND)
        if kind != cls.POLICY_KIND:
            raise ValueError(f"Unexpected policy kind '{kind}' for DenseTabularPolicy.")

        version = payload.get("version", 1)
        if version != cls.POLICY_VERSION:
            raise ValueError(f"Unsupported DenseTabularPolicy version {version}.")

        spec_dict = payload.get("spec")
        if spec_dict is None:
            raise ValueError("Missing spec in DenseTabularPolicy payload.")
        spec = cls._decode_spec(spec_dict)

        counts = payload.get("counts", {})
        k = int(counts.get("k", len(rules_for_spec(spec).claims)))
        H = int(counts.get("H", 1 << k))
        A = int(counts.get("A", k + 1))

        S_blob = blobs.get("S")
        if S_blob is None:
            raise ValueError("Missing S blob for DenseTabularPolicy.")

        S_arr = np.asarray(S_blob, dtype=np.float32)
        if S_arr.shape[0] != H or S_arr.shape[2] != A:
            raise ValueError(f"S blob shape {S_arr.shape} does not match counts (H={H}, A={A}).")

        policy = cls(spec)
        if S_arr.shape != policy.S.shape:
            raise ValueError(f"S blob shape {S_arr.shape} does not match expected {policy.S.shape}.")

        policy.S = S_arr
        policy.recompute_likelihoods()
        return policy

    def iter_children(self):
        return ()


def mix_dense(a: DenseTabularPolicy, b: DenseTabularPolicy, w_a: float) -> DenseTabularPolicy:
    if not 0.0 <= w_a <= 1.0:
        raise ValueError("w_a must be in [0, 1].")
    if a.spec != b.spec:
        raise ValueError("DenseTabularPolicy spec mismatch.")
    if a.k != b.k or a.hands != b.hands:
        raise ValueError("DenseTabularPolicy action/hand space mismatch.")
    if a.S.shape != b.S.shape:
        raise ValueError("DenseTabularPolicy tensor shape mismatch.")

    mix = DenseTabularPolicy(a.spec)

    pid_act = (a.popcount & 1).astype(np.int8)
    pick_a = pid_act[:, None] == 0
    L_a = np.where(pick_a, a.L_pid0, a.L_pid1)
    L_b = np.where(pick_a, b.L_pid0, b.L_pid1)

    alpha_a = w_a * L_a
    alpha_b = (1.0 - w_a) * L_b
    denom = alpha_a + alpha_b

    with np.errstate(divide="ignore", invalid="ignore"):
        alpha_a = np.divide(alpha_a, denom, out=np.zeros_like(alpha_a), where=denom > 0)
        alpha_b = np.divide(alpha_b, denom, out=np.zeros_like(alpha_b), where=denom > 0)

    mix_S = alpha_a[:, :, None] * a.S + alpha_b[:, :, None] * b.S
    use_mix = denom > 0
    mix.S = np.where(use_mix[:, :, None], mix_S, mix.S)
    mix.recompute_likelihoods()
    return mix


def mix_dense_multiple(
    policies: Sequence[DenseTabularPolicy],
    weights: Sequence[float],
) -> DenseTabularPolicy:
    if len(policies) != len(weights):
        raise ValueError("Policies and weights must have the same length.")
    if not policies:
        raise ValueError("mix_dense_multiple requires at least one policy.")

    w = np.asarray(weights, dtype=float)
    if np.any(w < 0):
        raise ValueError("mix_dense_multiple weights must be non-negative.")
    total = float(w.sum())
    if not np.isclose(total, 1.0):
        raise ValueError("mix_dense_multiple weights must sum to 1.")

    spec = policies[0].spec
    for pol in policies:
        if pol.spec != spec:
            raise ValueError("DenseTabularPolicy spec mismatch.")
        if pol.k != policies[0].k or pol.hands != policies[0].hands:
            raise ValueError("DenseTabularPolicy action/hand space mismatch.")
        if pol.S.shape != policies[0].S.shape:
            raise ValueError("DenseTabularPolicy tensor shape mismatch.")

    if len(policies) == 1:
        base = DenseTabularPolicy(spec)
        base.S = policies[0].S.copy()
        base.recompute_likelihoods()
        return base

    mix = DenseTabularPolicy(spec)
    pid_act = (mix.popcount & 1).astype(np.int8)

    denom = np.zeros((mix.S.shape[0], mix.S.shape[1]), dtype=float)
    L_tables: List[np.ndarray] = []
    for pol, w_i in zip(policies, w):
        pick_a = pid_act[:, None] == 0
        L_i = np.where(pick_a, pol.L_pid0, pol.L_pid1)
        L_tables.append(L_i)
        denom += float(w_i) * L_i

    mix_S = np.zeros_like(mix.S, dtype=float)
    for pol, w_i, L_i in zip(policies, w, L_tables):
        weight = float(w_i) * L_i
        with np.errstate(divide="ignore", invalid="ignore"):
            alpha = np.divide(weight, denom, out=np.zeros_like(weight), where=denom > 0)
        mix_S += alpha[:, :, None] * pol.S

    use_mix = denom > 0
    mix.S = np.where(use_mix[:, :, None], mix_S, mix.S)
    mix.recompute_likelihoods()
    return mix
