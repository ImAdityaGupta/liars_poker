"""Depth-limited responder with batched opponent queries.

``LimitedBestResponse`` asks the opponent network for one public history at a
time (one row per opponent hand), so a depth-2 or depth-3 search makes many
thousands of tiny network calls. This subclass, before each responder
decision, walks the planning tree level by level, collects every opponent
history the search will need, and evaluates them in a few large batches. The
recursive search then runs unchanged against a warm cache.

Prefetching only changes where probabilities are computed, not which are used:
any history the prefetch misses is still computed on demand, so results match
the parent class up to float rounding from different batch sizes.
"""
from __future__ import annotations

from collections import OrderedDict
from functools import lru_cache
import time

import numpy as np
import torch

from liars_poker.algo.br_limited_dense import LimitedBestResponse
from liars_poker.infoset import CALL
from liars_poker.policies.neural_regret import NeuralRegretMatchingPolicy
from liars_poker.policies.tabular_dense import DenseTabularPolicy


class _ProbCache:
    """Bounded LRU cache of opponent probabilities keyed by public history."""

    def __init__(self, compute, maxsize: int) -> None:
        self.compute = compute
        self.maxsize = maxsize
        self.rows: OrderedDict[int, np.ndarray] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def __call__(self, hid: int) -> np.ndarray:
        row = self.rows.get(hid)
        if row is not None:
            self.rows.move_to_end(hid)
            self.hits += 1
            return row
        self.misses += 1
        row = self.compute(hid)
        self.put(hid, row)
        return row

    def __contains__(self, hid: int) -> bool:
        return hid in self.rows

    def put(self, hid: int, row: np.ndarray) -> None:
        self.rows[hid] = row
        self.rows.move_to_end(hid)
        while len(self.rows) > self.maxsize:
            self.rows.popitem(last=False)

    def cache_info(self):
        from functools import _CacheInfo
        return _CacheInfo(self.hits, self.misses, self.maxsize, len(self.rows))


class BatchedLimitedBestResponse(LimitedBestResponse):
    def __init__(self, opponent, depth: int = 1, epsilon: float = 0.0, *,
                 cache_size: int = 300_000, max_batch_histories: int = 4096):
        super().__init__(opponent, depth=depth, epsilon=epsilon)
        self.batch_calls = 0
        self.max_batch_histories = max_batch_histories
        # Column a of not_true[hand] is 1 where claim a is false given (hand, opponent hand):
        # the responder's payoff when it calls claim a. Used for depth-1 leaves.
        self._not_true = np.stack([np.stack([~t[h] for t in self.truth], axis=1).astype(np.float64)
                                   for h in range(self.n)])
        self._true = np.stack([np.stack([t[h] for t in self.truth], axis=1).astype(np.float64)
                               for h in range(self.n)])
        self._reach = lru_cache(maxsize=1_000_000)(self._reach_uncached)
        if isinstance(opponent, DenseTabularPolicy):
            return
        self._probs = _ProbCache(self._probs_uncached, cache_size)
        self._hand_x = opponent.encoder.encode_hands(self.hands, ())
        self._bits = np.arange(self.k, dtype=np.int64)
        self._ranks = self.spec.ranks

    # ------------------------------------------------------------------
    def _fetch(self, hids: list[int]) -> None:
        """Evaluate the opponent network for many public histories at once."""
        todo = [h for h in dict.fromkeys(hids) if h not in self._probs]
        if not todo:
            return
        start = time.perf_counter()
        policy = self.opponent
        regret = isinstance(policy, NeuralRegretMatchingPolicy)
        for parity in (0, 1):
            group = [h for h in todo if (h.bit_count() & 1) == parity]
            for at in range(0, len(group), self.max_batch_histories):
                chunk = group[at:at + self.max_batch_histories]
                m = len(chunk)
                H = np.asarray(chunk, dtype=np.int64)
                x = np.broadcast_to(self._hand_x, (m, self.n, self._hand_x.shape[1])).copy()
                x[:, :, self._ranks:] = ((H[:, None] >> self._bits) & 1)[:, None, :]
                mask = np.zeros((m, self.k + 1), dtype=bool)
                for i, h in enumerate(chunk):
                    for action in self._actions(h):
                        mask[i, 0 if action == CALL else action + 1] = True
                with torch.inference_mode():
                    logits = policy._model(parity)(
                        torch.from_numpy(x.reshape(m * self.n, -1)).to(policy.device)
                    ).reshape(m, self.n, self.k + 1)
                    legal = torch.from_numpy(mask).to(logits.device)[:, None, :]
                    if regret:
                        positive = logits.clamp_min(0) * legal
                        totals = positive.sum(dim=2, keepdim=True)
                        uniform = legal.float() / legal.sum(dim=2, keepdim=True)
                        probs = torch.where(totals > 0, positive / totals.clamp_min(1e-30),
                                            uniform.expand_as(positive))
                    else:
                        probs = torch.softmax(logits.masked_fill(~legal, float("-inf")), dim=2)
                        probs = probs.masked_fill(~legal, 0.0)
                    probs = probs.float().cpu().numpy()
                for i, h in enumerate(chunk):
                    self._probs.put(h, probs[i])
                self.batch_calls += 1
                self.network_queries += 1
                self.network_rows += m * self.n
        self.network_query_s += time.perf_counter() - start

    def _prefetch(self, hid: int, hand: int, seat: int) -> None:
        """Warm the cache with every opponent history this decision's search can reach."""
        responder = [(hid, self.depth)]
        blockers = self.blockers[hand]
        while responder:
            opponent = []
            for h, d in responder:
                for action in self._actions(h):
                    if action != CALL:
                        opponent.append((h | (1 << action), d))
            self._fetch([h for h, _ in opponent])
            nxt = {}
            for h1, d in opponent:
                if d <= 1:
                    continue
                reach = self._reach(h1, seat)
                S = self._probs(h1)
                baseline = float(np.dot(blockers, reach))
                for action in self._actions(h1):
                    if action == CALL:
                        continue
                    mass = float(np.dot(blockers, reach * S[:, action + 1]))
                    if mass <= 0 or (baseline > 0 and mass < self.epsilon * baseline):
                        continue
                    nxt[h1 | (1 << action)] = d - 1
            responder = list(nxt.items())

    def _next_opponent(self, hid: int, hand: int, seat: int,
                       depth: int, *, planning: bool) -> float:
        """Vectorised planning step: all opponent replies' masses in one product."""
        if not planning:
            return super()._next_opponent(hid, hand, seat, depth, planning=planning)
        reach = self._reach(hid, seat)
        S = self._probs(hid)
        actions = self._actions(hid)
        cols = np.fromiter((0 if a == CALL else a + 1 for a in actions), dtype=np.int64,
                           count=len(actions))
        bw = self.blockers[hand] * reach
        baseline = float(bw.sum())
        masses = bw @ S[:, cols]
        alive = masses > 0
        is_call = cols == 0
        pruned = alive & ~is_call & (baseline > 0) & (masses < self.epsilon * baseline)
        self.search_branches += int(alive.sum())
        self.skipped_search_branches += int(pruned.sum())
        keep = alive & ~pruned
        total = 0.0
        if is_call.any() and keep[is_call].any():
            # The opponent calls our last claim: we win when it is true.
            last = self._last_claim(hid)
            total += float((bw * S[:, 0]) @ self._true[hand][:, last])
        claim_idx = np.flatnonzero(keep & ~is_call)
        if not len(claim_idx):
            return total
        if depth > 1:
            for i in claim_idx:
                total += self._plan(hid | (1 << actions[i]), hand, seat, depth - 1)
            return total
        # Depth-1 leaves: we call the opponent's raise immediately.
        claims = cols[claim_idx] - 1
        leaf = (bw[:, None] * S[:, cols[claim_idx]]) * self._not_true[hand][:, claims]
        return total + float(leaf.sum())

    def _plan_uncached(self, hid: int, hand: int, seat: int, depth: int) -> float:
        """Depth-1 responder nodes in one batched computation over all our actions."""
        if depth != 1:
            return super()._plan_uncached(hid, hand, seat, depth)
        self.plan_nodes += 1
        reach = self._reach(hid, seat)
        bw = self.blockers[hand] * reach
        actions = self._actions(hid)
        best = -1.0
        claims = [a for a in actions if a != CALL]
        if CALL in actions:
            # We call the opponent's last claim: we win when it is false.
            best = float(bw @ self._not_true[hand][:, self._last_claim(hid)])
        if not claims:
            return best
        A = np.asarray(claims, dtype=np.int64)
        S = np.stack([self._probs(hid | (1 << a)) for a in claims])          # (m, n, k+1)
        legal = np.zeros((len(claims), self.k + 1), dtype=bool)
        legal[:, 0] = True                                                    # the opponent may call
        legal[:, 1:] = np.arange(self.k)[None, :] > A[:, None]               # or raise above our claim
        masses = np.einsum("n,mnc->mc", bw, S)
        baseline = float(bw.sum())
        alive = (masses > 0) & legal
        pruned = alive & (baseline > 0) & (masses < self.epsilon * baseline)
        pruned[:, 0] = False
        self.search_branches += int(alive.sum())
        self.skipped_search_branches += int(pruned.sum())
        keep = alive & ~pruned
        call_value = np.einsum("n,mn,nm->m", bw, S[:, :, 0], self._true[hand][:, A])
        leaf = np.einsum("n,mnc,nc->mc", bw, S[:, :, 1:], self._not_true[hand])
        values = np.where(keep[:, 0], call_value, 0.0) + (leaf * keep[:, 1:]).sum(axis=1)
        return max(best, float(values.max()))

    def _choice_uncached(self, hid: int, hand: int, seat: int) -> int:
        if not isinstance(self.opponent, DenseTabularPolicy):
            self._prefetch(hid, hand, seat)
        return super()._choice_uncached(hid, hand, seat)
