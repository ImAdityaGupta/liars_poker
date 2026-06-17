from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import torch

from liars_poker.core import card_rank, generate_deck
from liars_poker.infoset import CALL


class BatchedDeepCFRTraverser:
    """Levelized external-sampling traversal for Deep CFR.

    This keeps the algorithmic semantics of the recursive Deep CFR traversal:
    opponent nodes sample one action, traverser nodes expand all legal actions.
    The difference is that neural inference is done per frontier rather than
    one infoset at a time.
    """

    KIND_RANK_HIGH = 0
    KIND_PAIR = 1
    KIND_TWO_PAIR = 2
    KIND_TRIPS = 3
    KIND_FULL_HOUSE = 4
    KIND_QUADS = 5

    def __init__(self, trainer) -> None:
        self.trainer = trainer
        self.spec = trainer.spec
        self.rules = trainer.rules
        self.encoder = trainer.encoder
        self.device = trainer.device
        self.k = self.encoder.k
        self.action_dim = self.encoder.action_dim
        self.ranks = self.spec.ranks

        deck_ranks = [card_rank(card, self.spec) - 1 for card in generate_deck(self.spec)]
        self.deck_ranks = torch.as_tensor(deck_ranks, dtype=torch.long, device=self.device)

        self._legal_cols_by_last: Dict[int, List[int]] = {}
        self._legal_mask_by_last: Dict[int, np.ndarray] = {}
        self._history_bits_cache: Dict[int, np.ndarray] = {}

        kind_codes: List[int] = []
        rank_a: List[int] = []
        rank_b: List[int] = []
        for kind, value in self.rules.claims:
            if kind == "RankHigh":
                kind_codes.append(self.KIND_RANK_HIGH)
                rank_a.append(value - 1)
                rank_b.append(-1)
            elif kind == "Pair":
                kind_codes.append(self.KIND_PAIR)
                rank_a.append(value - 1)
                rank_b.append(-1)
            elif kind == "TwoPair":
                low, high = self.rules.two_pair_ranks[value]
                kind_codes.append(self.KIND_TWO_PAIR)
                rank_a.append(low - 1)
                rank_b.append(high - 1)
            elif kind == "Trips":
                kind_codes.append(self.KIND_TRIPS)
                rank_a.append(value - 1)
                rank_b.append(-1)
            elif kind == "FullHouse":
                trip, pair = self.rules.full_house_ranks[value]
                kind_codes.append(self.KIND_FULL_HOUSE)
                rank_a.append(trip - 1)
                rank_b.append(pair - 1)
            elif kind == "Quads":
                kind_codes.append(self.KIND_QUADS)
                rank_a.append(value - 1)
                rank_b.append(-1)
            else:
                raise ValueError(f"Unsupported claim kind: {kind}")

        self.claim_kind = torch.as_tensor(kind_codes, dtype=torch.long, device=self.device)
        self.claim_rank_a = torch.as_tensor(rank_a, dtype=torch.long, device=self.device)
        self.claim_rank_b = torch.as_tensor(rank_b, dtype=torch.long, device=self.device)

    @staticmethod
    def _action_col(action: int) -> int:
        return 0 if action == CALL else action + 1

    @staticmethod
    def _last_claim_from_hid(hid: int) -> int:
        return -1 if hid == 0 else hid.bit_length() - 1

    def _legal_cols_for_last(self, last_claim: int) -> List[int]:
        cached = self._legal_cols_by_last.get(last_claim)
        if cached is not None:
            return cached
        ref = None if last_claim < 0 else last_claim
        cols = [self._action_col(action) for action in self.rules.legal_actions_from_last(ref)]
        self._legal_cols_by_last[last_claim] = cols
        return cols

    def _legal_mask_for_last(self, last_claim: int) -> np.ndarray:
        cached = self._legal_mask_by_last.get(last_claim)
        if cached is not None:
            return cached
        mask = np.zeros(self.action_dim, dtype=bool)
        mask[self._legal_cols_for_last(last_claim)] = True
        self._legal_mask_by_last[last_claim] = mask
        return mask

    def _history_bits(self, hid: int) -> np.ndarray:
        cached = self._history_bits_cache.get(hid)
        if cached is not None:
            return cached
        bits = np.zeros(self.k, dtype=np.float32)
        value = hid
        while value:
            low_bit = value & -value
            idx = low_bit.bit_length() - 1
            bits[idx] = 1.0
            value ^= low_bit
        self._history_bits_cache[hid] = bits
        return bits

    def _sample_deals(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        deck_size = int(self.deck_ranks.numel())
        draw_count = 2 * self.spec.hand_size
        order = torch.rand(batch_size, deck_size, device=self.device).argsort(dim=1)[:, :draw_count]
        ranks = self.deck_ranks[order]
        p1_ranks = ranks[:, : self.spec.hand_size]
        p2_ranks = ranks[:, self.spec.hand_size :]

        p1_counts = torch.zeros(batch_size, self.ranks, dtype=torch.float32, device=self.device)
        p2_counts = torch.zeros_like(p1_counts)
        p1_counts.scatter_add_(1, p1_ranks, torch.ones_like(p1_ranks, dtype=torch.float32))
        p2_counts.scatter_add_(1, p2_ranks, torch.ones_like(p2_ranks, dtype=torch.float32))
        return p1_counts, p2_counts, p1_counts + p2_counts

    def _features(
        self,
        *,
        actor: int,
        node_indices: Sequence[int],
        deal_indices: Sequence[int],
        hids: Sequence[int],
        p1_counts: torch.Tensor,
        p2_counts: torch.Tensor,
    ) -> torch.Tensor:
        deal_idx = torch.as_tensor(deal_indices, dtype=torch.long, device=self.device)
        hand_counts = (p1_counts if actor == 0 else p2_counts).index_select(0, deal_idx)
        history = np.stack([self._history_bits(hids[idx]) for idx in node_indices]).astype(np.float32)
        history_t = torch.as_tensor(history, dtype=torch.float32, device=self.device)
        return torch.cat((hand_counts, history_t), dim=1)

    def _legal_mask_batch(self, node_indices: Sequence[int], hids: Sequence[int]) -> np.ndarray:
        return np.stack(
            [
                self._legal_mask_for_last(self._last_claim_from_hid(hids[idx]))
                for idx in node_indices
            ]
        )

    def _strategy_batch(
        self,
        actor: int,
        features: torch.Tensor,
        legal_mask: torch.Tensor,
    ) -> torch.Tensor:
        model = self.trainer.advantage_nets[actor]
        with torch.no_grad():
            with self.trainer._autocast():
                values = self.trainer._forward(model, features)
            values = values.float()
            mask_float = legal_mask.float()
            positive = torch.relu(values) * mask_float
            totals = positive.sum(dim=1, keepdim=True)
            matched = positive / totals.clamp_min(1e-8)
            if self.trainer.highest_regret_fallback:
                masked_values = values.masked_fill(~legal_mask, -torch.inf)
                fallback = torch.zeros_like(values).scatter_(
                    1,
                    masked_values.argmax(dim=1, keepdim=True),
                    1.0,
                )
            else:
                fallback = mask_float / mask_float.sum(dim=1, keepdim=True).clamp_min(1.0)
            return torch.where(totals > 0.0, matched, fallback)

    def _terminal_values(
        self,
        *,
        hids: Sequence[int],
        deal_indices: Sequence[int],
        caller: int,
        traverser: int,
        total_counts: torch.Tensor,
    ) -> np.ndarray:
        last_claims = torch.as_tensor(
            [self._last_claim_from_hid(hid) for hid in hids],
            dtype=torch.long,
            device=self.device,
        )
        deal_idx = torch.as_tensor(deal_indices, dtype=torch.long, device=self.device)
        counts = total_counts.index_select(0, deal_idx)
        row = torch.arange(len(hids), dtype=torch.long, device=self.device)

        kind = self.claim_kind.index_select(0, last_claims)
        rank_a = self.claim_rank_a.index_select(0, last_claims)
        rank_b = self.claim_rank_b.index_select(0, last_claims)
        count_a = counts[row, rank_a]

        satisfied = torch.zeros(len(hids), dtype=torch.bool, device=self.device)
        satisfied |= (kind == self.KIND_RANK_HIGH) & (count_a >= 1)
        satisfied |= (kind == self.KIND_PAIR) & (count_a >= 2)
        satisfied |= (kind == self.KIND_TRIPS) & (count_a >= 3)
        satisfied |= (kind == self.KIND_QUADS) & (count_a >= 4)

        needs_b = (kind == self.KIND_TWO_PAIR) | (kind == self.KIND_FULL_HOUSE)
        if bool(needs_b.any().item()):
            safe_rank_b = rank_b.clamp_min(0)
            count_b = counts[row, safe_rank_b]
            satisfied |= (kind == self.KIND_TWO_PAIR) & (count_a >= 2) & (count_b >= 2)
            satisfied |= (kind == self.KIND_FULL_HOUSE) & (count_a >= 3) & (count_b >= 2)

        winner_if_true = 1 - caller
        winner_if_false = caller
        winner = torch.where(
            satisfied,
            torch.full_like(last_claims, winner_if_true),
            torch.full_like(last_claims, winner_if_false),
        )
        values = torch.where(
            winner == traverser,
            torch.ones_like(count_a),
            -torch.ones_like(count_a),
        )
        return values.detach().cpu().numpy().astype(np.float32, copy=False)

    def run_traversals(self, traverser: int, batch_size: int) -> None:
        p1_counts, p2_counts, total_counts = self._sample_deals(batch_size)

        node_deal_idx: List[int] = list(range(batch_size))
        node_hids: List[int] = [0] * batch_size
        node_depths: List[int] = [0] * batch_size
        levels: List[List[int]] = [[] for _ in range(self.k + 1)]
        levels[0] = list(range(batch_size))

        edge_cols: List[List[int]] = [[] for _ in range(batch_size)]
        edge_children: List[List[int]] = [[] for _ in range(batch_size)]
        edge_terminal_values: List[List[float]] = [[] for _ in range(batch_size)]
        sampled_child: List[int] = [-1] * batch_size
        sampled_terminal: List[float] = [0.0] * batch_size
        node_strategy: List[np.ndarray | None] = [None] * batch_size
        node_features: List[np.ndarray | None] = [None] * batch_size
        node_legal_mask: List[np.ndarray | None] = [None] * batch_size

        strategy_features: List[np.ndarray] = []
        strategy_targets: List[np.ndarray] = []
        strategy_masks: List[np.ndarray] = []

        def append_node(deal_idx: int, hid: int, depth: int) -> int:
            idx = len(node_hids)
            node_deal_idx.append(deal_idx)
            node_hids.append(hid)
            node_depths.append(depth)
            edge_cols.append([])
            edge_children.append([])
            edge_terminal_values.append([])
            sampled_child.append(-1)
            sampled_terminal.append(0.0)
            node_strategy.append(None)
            node_features.append(None)
            node_legal_mask.append(None)
            levels[depth].append(idx)
            return idx

        for depth in range(self.k + 1):
            nodes = levels[depth]
            if not nodes:
                continue

            actor = depth & 1
            features_t = self._features(
                actor=actor,
                node_indices=nodes,
                deal_indices=[node_deal_idx[idx] for idx in nodes],
                hids=node_hids,
                p1_counts=p1_counts,
                p2_counts=p2_counts,
            )
            legal_mask_np = self._legal_mask_batch(nodes, node_hids)
            legal_mask_t = torch.as_tensor(legal_mask_np, dtype=torch.bool, device=self.device)
            strategy_t = self._strategy_batch(actor, features_t, legal_mask_t)
            features_np = features_t.detach().cpu().numpy().astype(np.float32, copy=False)
            strategy_np = strategy_t.detach().cpu().numpy().astype(np.float32, copy=False)

            if actor == traverser:
                call_nodes: List[int] = []
                call_hids: List[int] = []
                call_deals: List[int] = []
                for row, idx in enumerate(nodes):
                    node_strategy[idx] = strategy_np[row].copy()
                    node_features[idx] = features_np[row].copy()
                    node_legal_mask[idx] = legal_mask_np[row].copy()
                    if legal_mask_np[row, 0]:
                        call_nodes.append(idx)
                        call_hids.append(node_hids[idx])
                        call_deals.append(node_deal_idx[idx])

                call_values: Dict[int, float] = {}
                if call_nodes:
                    values = self._terminal_values(
                        hids=call_hids,
                        deal_indices=call_deals,
                        caller=actor,
                        traverser=traverser,
                        total_counts=total_counts,
                    )
                    call_values = {idx: float(value) for idx, value in zip(call_nodes, values)}

                for row, idx in enumerate(nodes):
                    last_claim = self._last_claim_from_hid(node_hids[idx])
                    for col in self._legal_cols_for_last(last_claim):
                        edge_cols[idx].append(col)
                        if col == 0:
                            edge_children[idx].append(-1)
                            edge_terminal_values[idx].append(call_values[idx])
                        else:
                            action = col - 1
                            child_idx = append_node(
                                node_deal_idx[idx],
                                node_hids[idx] | (1 << action),
                                depth + 1,
                            )
                            edge_children[idx].append(child_idx)
                            edge_terminal_values[idx].append(0.0)
                continue

            strategy_features.append(features_np)
            strategy_targets.append(strategy_np)
            strategy_masks.append(legal_mask_np)

            sampled_cols = torch.multinomial(strategy_t, 1).squeeze(1).detach().cpu().numpy()
            call_rows = np.flatnonzero(sampled_cols == 0)
            call_values: Dict[int, float] = {}
            if len(call_rows):
                call_node_indices = [nodes[int(row)] for row in call_rows]
                values = self._terminal_values(
                    hids=[node_hids[idx] for idx in call_node_indices],
                    deal_indices=[node_deal_idx[idx] for idx in call_node_indices],
                    caller=actor,
                    traverser=traverser,
                    total_counts=total_counts,
                )
                call_values = {
                    idx: float(value) for idx, value in zip(call_node_indices, values)
                }

            for row, idx in enumerate(nodes):
                col = int(sampled_cols[row])
                if col == 0:
                    sampled_terminal[idx] = call_values[idx]
                    continue
                action = col - 1
                sampled_child[idx] = append_node(
                    node_deal_idx[idx],
                    node_hids[idx] | (1 << action),
                    depth + 1,
                )

        node_values = [0.0] * len(node_hids)
        advantage_features: List[np.ndarray] = []
        advantage_targets: List[np.ndarray] = []
        advantage_masks: List[np.ndarray] = []

        for idx in range(len(node_hids) - 1, -1, -1):
            actor = node_depths[idx] & 1
            if actor != traverser:
                child = sampled_child[idx]
                node_values[idx] = sampled_terminal[idx] if child < 0 else node_values[child]
                continue

            action_values = np.zeros(self.action_dim, dtype=np.float32)
            for col, child, terminal_value in zip(
                edge_cols[idx],
                edge_children[idx],
                edge_terminal_values[idx],
            ):
                action_values[col] = terminal_value if child < 0 else node_values[child]

            strategy = node_strategy[idx]
            legal_mask = node_legal_mask[idx]
            features = node_features[idx]
            if strategy is None or legal_mask is None or features is None:
                raise RuntimeError("Missing traverser node data in batched traversal.")

            node_value = float(np.dot(strategy, action_values))
            node_values[idx] = node_value
            target = np.zeros(self.action_dim, dtype=np.float32)
            target[legal_mask] = action_values[legal_mask] - node_value
            advantage_features.append(features)
            advantage_targets.append(target)
            advantage_masks.append(legal_mask)

        if advantage_features:
            self.trainer._add_records(
                self.trainer.advantage_buffers[traverser],
                self.trainer.advantage_validation_buffers[traverser],
                np.stack(advantage_features),
                np.stack(advantage_targets),
                np.stack(advantage_masks),
                float(self.trainer.iteration),
            )

        if strategy_features:
            weight = (
                1.0
                if self.trainer.strategy_weighting == "uniform"
                else float(self.trainer.iteration)
            )
            actor_pid = 1 - traverser
            self.trainer._add_records(
                self.trainer.strategy_buffers[actor_pid],
                self.trainer.strategy_validation_buffers[actor_pid],
                np.concatenate(strategy_features, axis=0),
                np.concatenate(strategy_targets, axis=0),
                np.concatenate(strategy_masks, axis=0),
                weight,
            )
