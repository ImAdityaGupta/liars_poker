from __future__ import annotations

import time
from typing import Dict, List

import torch

from liars_poker.algo.packed_history import PackedHistory
from liars_poker.core import card_rank, generate_deck
from liars_poker.infoset import CALL


class _DeviceRecordAccumulator:
    def __init__(
        self,
        trainer,
        buffer,
        validation_buffer,
        *,
        flush_size: int,
        commit_records: bool,
    ) -> None:
        self.trainer = trainer
        self.buffer = buffer
        self.validation_buffer = validation_buffer
        self.flush_size = int(flush_size)
        self.commit_records = bool(commit_records)
        self.features: List[torch.Tensor] = []
        self.targets: List[torch.Tensor] = []
        self.masks: List[torch.Tensor] = []
        self.weights: List[torch.Tensor] = []
        self.pending = 0
        self.count = 0

    def append(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
        masks: torch.Tensor,
        weights: torch.Tensor,
    ) -> None:
        rows = int(features.shape[0])
        if rows == 0:
            return
        self.count += rows
        if not self.commit_records:
            return
        self.features.append(features)
        self.targets.append(targets)
        self.masks.append(masks)
        self.weights.append(weights)
        self.pending += rows
        if self.pending >= self.flush_size:
            self.flush()

    def flush(self) -> None:
        if not self.commit_records or not self.features:
            self.features.clear()
            self.targets.clear()
            self.masks.clear()
            self.weights.clear()
            self.pending = 0
            return
        self.trainer._add_device_records(
            self.buffer,
            self.validation_buffer,
            torch.cat(self.features, dim=0),
            torch.cat(self.targets, dim=0),
            torch.cat(self.masks, dim=0),
            torch.cat(self.weights, dim=0),
        )
        self.features.clear()
        self.targets.clear()
        self.masks.clear()
        self.weights.clear()
        self.pending = 0


class GPUDeepCFRPlusTraverser:
    """Tensorized external-sampling traversal for neural CFR+."""

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
        self.action_sample_count = trainer.traverser_action_sample_count
        self.action_sample_fraction = trainer.traverser_action_sample_fraction
        self.action_full_first = trainer.traverser_action_full_first
        self.action_sample_schedule = trainer.traverser_action_sample_schedule
        self.action_priority_count = trainer.traverser_action_priority_count
        self.action_baseline = trainer.traverser_action_baseline
        self.action_sample_mode = trainer.traverser_action_sample_mode
        self.history = PackedHistory(self.k, self.device)

        deck_ranks = [
            card_rank(card, self.spec) - 1 for card in generate_deck(self.spec)
        ]
        self.deck_ranks = torch.tensor(
            deck_ranks,
            dtype=torch.long,
            device=self.device,
        )
        legal_masks = torch.zeros(
            (self.k + 1, self.action_dim),
            dtype=torch.bool,
            device=self.device,
        )
        for last_claim in range(-1, self.k):
            ref = None if last_claim < 0 else last_claim
            legal = self.rules.legal_actions_from_last(ref)
            cols = [0 if action == CALL else action + 1 for action in legal]
            legal_masks[last_claim + 1, cols] = True
        self.legal_masks = legal_masks

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

        self.claim_kind = torch.tensor(
            kind_codes,
            dtype=torch.long,
            device=self.device,
        )
        self.claim_rank_a = torch.tensor(
            rank_a,
            dtype=torch.long,
            device=self.device,
        )
        self.claim_rank_b = torch.tensor(
            rank_b,
            dtype=torch.long,
            device=self.device,
        )

    def _scheduled_action_sample_count(self, traverser_decision: int) -> int | None:
        if self.action_sample_schedule is None:
            return self.action_sample_count
        idx = min(traverser_decision, len(self.action_sample_schedule) - 1)
        return int(self.action_sample_schedule[idx])

    def _sample_deals(
        self,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        deck_size = int(self.deck_ranks.numel())
        draw_count = 2 * self.spec.hand_size
        order = torch.rand(
            (batch_size, deck_size),
            device=self.device,
        ).topk(draw_count, dim=1).indices
        ranks = self.deck_ranks.index_select(0, order.reshape(-1)).reshape(
            batch_size,
            draw_count,
        )
        p1_ranks = ranks[:, : self.spec.hand_size]
        p2_ranks = ranks[:, self.spec.hand_size :]

        p1_counts = torch.zeros(
            (batch_size, self.ranks),
            dtype=torch.float32,
            device=self.device,
        )
        p2_counts = torch.zeros_like(p1_counts)
        ones = torch.ones_like(p1_ranks, dtype=torch.float32)
        p1_counts.scatter_add_(1, p1_ranks, ones)
        p2_counts.scatter_add_(1, p2_ranks, ones)
        return p1_counts, p2_counts, p1_counts + p2_counts

    def _features(
        self,
        actor: int,
        deal_idx: torch.Tensor,
        histories: torch.Tensor,
        p1_counts: torch.Tensor,
        p2_counts: torch.Tensor,
    ) -> torch.Tensor:
        hand_counts = (p1_counts if actor == 0 else p2_counts).index_select(
            0,
            deal_idx,
        )
        history = self.history.features(histories)
        return torch.cat((hand_counts, history), dim=1)

    def _regrets_and_strategy(
        self,
        actor: int,
        features: torch.Tensor,
        legal_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        model = self.trainer.regret_nets[actor]
        with torch.inference_mode():
            with self.trainer._autocast():
                values = self.trainer._forward(model, features)
            values = values.float()
            positive = torch.relu(values) * legal_mask
            totals = positive.sum(dim=1, keepdim=True)
            matched = positive / totals.clamp_min(1e-8)
            fallback = legal_mask / legal_mask.sum(
                dim=1,
                keepdim=True,
            ).clamp_min(1)
            strategy = torch.where(totals > 0.0, matched, fallback)
        return values, strategy

    def _claim_edges(
        self,
        legal_mask: torch.Tensor,
        regret_values: torch.Tensor,
        histories: torch.Tensor,
        traverser_decision: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        int,
    ]:
        claim_legal = legal_mask[:, 1:]
        full_edge_count = claim_legal.sum()
        scheduled_sample_count = self._scheduled_action_sample_count(
            traverser_decision,
        )
        if (
            (
                scheduled_sample_count is None
                and self.action_sample_fraction is None
            )
            or (self.action_full_first and traverser_decision == 0)
        ):
            edges = claim_legal.nonzero(as_tuple=False)
            return (
                edges[:, 0],
                edges[:, 1],
                torch.ones(
                    len(edges),
                    dtype=torch.float32,
                    device=self.device,
                ),
                full_edge_count,
                full_edge_count,
            )

        legal_counts = claim_legal.sum(dim=1)
        if self.action_sample_fraction is not None:
            sample_counts = torch.ceil(
                legal_counts.float() * self.action_sample_fraction
            ).long()
        else:
            sample_counts = torch.full_like(
                legal_counts,
                scheduled_sample_count,
            )
        sample_counts = torch.minimum(
            sample_counts.clamp_min(1),
            legal_counts,
        )

        if self.action_priority_count:
            priority_counts = torch.minimum(
                torch.full_like(
                    legal_counts,
                    self.action_priority_count,
                ),
                sample_counts,
            )
            priority_scores = regret_values[:, 1:].masked_fill(
                ~claim_legal,
                -torch.inf,
            )
            priority_order = priority_scores.argsort(
                dim=1,
                descending=True,
            )
            priority_rank = priority_order.argsort(dim=1)
            priority_mask = claim_legal & (
                priority_rank < priority_counts[:, None]
            )

            random_counts = sample_counts - priority_counts
            random_eligible = claim_legal & ~priority_mask
            random_scores = self._sample_scores(
                random_eligible,
                histories,
                traverser_decision,
            )
            random_order = random_scores.argsort(dim=1, descending=True)
            random_rank = random_order.argsort(dim=1)
            random_mask = random_eligible & (
                random_rank < random_counts[:, None]
            )
            sampled_mask = priority_mask | random_mask

            remaining_counts = legal_counts - priority_counts
            random_inclusion = (
                random_counts.float()
                / remaining_counts.clamp_min(1).float()
            )
            inclusion_matrix = torch.where(
                priority_mask,
                torch.ones_like(priority_scores),
                random_inclusion[:, None].expand_as(priority_scores),
            )
        else:
            scores = self._sample_scores(
                claim_legal,
                histories,
                traverser_decision,
            )
            order = scores.argsort(dim=1, descending=True)
            rank = order.argsort(dim=1)
            sampled_mask = claim_legal & (rank < sample_counts[:, None])
            inclusion_by_row = (
                sample_counts.float() / legal_counts.clamp_min(1).float()
            )
            inclusion_matrix = inclusion_by_row[:, None].expand_as(scores)

        edges = sampled_mask.nonzero(as_tuple=False)
        parent_rows = edges[:, 0]
        claim_ids = edges[:, 1]
        inclusion = inclusion_matrix[parent_rows, claim_ids]
        return (
            parent_rows,
            claim_ids,
            inclusion,
            full_edge_count,
            int(len(edges)),
        )

    def _sample_scores(
        self,
        eligible: torch.Tensor,
        histories: torch.Tensor,
        traverser_decision: int,
    ) -> torch.Tensor:
        if self.action_sample_mode == "random":
            return torch.rand(
                eligible.shape,
                dtype=torch.float32,
                device=self.device,
            ).masked_fill(~eligible, -1.0)

        modulus = 2_147_483_647
        claim_ids = torch.arange(
            eligible.shape[1],
            dtype=torch.long,
            device=self.device,
        )
        history_words = histories.long()
        if history_words.ndim == 1:
            history_codes = torch.remainder(history_words, modulus)
        else:
            word_ids = torch.arange(
                history_words.shape[1],
                dtype=torch.long,
                device=self.device,
            )
            word_mix = torch.remainder(
                (word_ids + 1) * 1_000_003,
                modulus,
            )
            history_codes = torch.remainder(
                (torch.remainder(history_words, modulus) * word_mix).sum(dim=1),
                modulus,
            )
        history_codes = history_codes.unsqueeze(1)
        mixed = torch.remainder(
            history_codes * 1_103_515_245
            + claim_ids.unsqueeze(0) * 97_531
            + int(traverser_decision) * 17_389
            + int(self.trainer.iteration) * 7_919,
            modulus,
        )
        scores = mixed.float() / float(modulus)
        return scores.masked_fill(~eligible, -1.0)

    def _terminal_values(
        self,
        last_claim: torch.Tensor,
        deal_idx: torch.Tensor,
        *,
        caller: int,
        traverser: int,
        total_counts: torch.Tensor,
    ) -> torch.Tensor:
        safe_claim = last_claim.clamp_min(0)
        counts = total_counts.index_select(0, deal_idx)
        row = torch.arange(
            len(last_claim),
            dtype=torch.long,
            device=self.device,
        )

        kind = self.claim_kind.index_select(0, safe_claim)
        rank_a = self.claim_rank_a.index_select(0, safe_claim)
        rank_b = self.claim_rank_b.index_select(0, safe_claim)
        count_a = counts[row, rank_a]
        count_b = counts[row, rank_b.clamp_min(0)]

        satisfied = (
            ((kind == self.KIND_RANK_HIGH) & (count_a >= 1))
            | ((kind == self.KIND_PAIR) & (count_a >= 2))
            | ((kind == self.KIND_TWO_PAIR) & (count_a >= 2) & (count_b >= 2))
            | ((kind == self.KIND_TRIPS) & (count_a >= 3))
            | ((kind == self.KIND_FULL_HOUSE) & (count_a >= 3) & (count_b >= 2))
            | ((kind == self.KIND_QUADS) & (count_a >= 4))
        )
        winner = torch.where(
            satisfied,
            torch.full_like(last_claim, 1 - caller),
            torch.full_like(last_claim, caller),
        )
        return torch.where(
            winner == traverser,
            torch.ones_like(count_a),
            -torch.ones_like(count_a),
        )

    def _stream_profile_row(
        self,
        profile_by_depth: Dict[int, Dict[str, object]],
        depth: int,
        actor: int,
        traverser: int,
    ) -> Dict[str, object]:
        row = profile_by_depth.get(depth)
        if row is None:
            row = {
                "depth": depth,
                "actor": actor,
                "traverser": traverser,
                "calls": 0,
                "active_rows_in": 0,
                "active_rows_out": 0,
                "peak_rows": 0,
                "traverser_claim_edges_full": 0,
                "traverser_claim_edges_expanded": 0,
                "edge_chunks": 0,
                "row_splits": 0,
                "opponent_continuations": 0,
                "regret_records": 0,
                "strategy_records": 0,
            }
            profile_by_depth[depth] = row
        return row

    def _run_traversals_streaming(
        self,
        traverser: int,
        batch_size: int,
        *,
        profile: bool,
        commit_records: bool,
    ) -> Dict[str, object]:
        if profile and self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

        p1_counts, p2_counts, total_counts = self._sample_deals(batch_size)
        current_deal = torch.arange(
            batch_size,
            dtype=torch.long,
            device=self.device,
        )
        current_history = self.history.zeros(batch_size)
        current_last = torch.full(
            (batch_size,),
            -1,
            dtype=torch.long,
            device=self.device,
        )
        current_path_probability = torch.ones(
            batch_size,
            dtype=torch.float32,
            device=self.device,
        )

        regret_accumulator = _DeviceRecordAccumulator(
            self.trainer,
            self.trainer.regret_buffers[traverser],
            self.trainer.regret_validation_buffers[traverser],
            flush_size=self.trainer.traversal_record_flush_size,
            commit_records=commit_records,
        )
        strategy_actor = 1 - traverser
        strategy_accumulator = _DeviceRecordAccumulator(
            self.trainer,
            self.trainer.strategy_buffers[strategy_actor],
            self.trainer.strategy_validation_buffers[strategy_actor],
            flush_size=self.trainer.traversal_record_flush_size,
            commit_records=commit_records,
        )

        profile_by_depth: Dict[int, Dict[str, object]] = {}
        stats: Dict[str, float | int] = {
            "full_claim_edges": 0,
            "sampled_claim_edges": 0,
            "regret_weight_sum": 0.0,
            "regret_weight_square_sum": 0.0,
            "regret_weight_count": 0,
            "max_regret_weight": 0.0,
            "peak_rows": batch_size,
            "edge_chunks": 0,
            "row_splits": 0,
        }
        iteration = max(float(self.trainer.iteration), 1.0)
        previous_scale = (iteration - 1.0) / iteration
        instant_scale = 1.0 / iteration
        live_budget = self.trainer.traversal_live_row_budget

        def update_peak(rows: int) -> None:
            stats["peak_rows"] = max(int(stats["peak_rows"]), int(rows))

        def evaluate_frontier(
            depth: int,
            deal_idx: torch.Tensor,
            histories: torch.Tensor,
            last_claim: torch.Tensor,
            path_probability: torch.Tensor,
        ) -> torch.Tensor:
            rows = self.history.rows(histories)
            if rows == 0:
                return torch.empty(0, dtype=torch.float32, device=self.device)
            update_peak(rows)

            if live_budget is not None and rows > live_budget:
                values: List[torch.Tensor] = []
                for start in range(0, rows, live_budget):
                    end = min(start + live_budget, rows)
                    idx = torch.arange(
                        start,
                        end,
                        dtype=torch.long,
                        device=self.device,
                    )
                    values.append(
                        evaluate_frontier(
                            depth,
                            deal_idx.index_select(0, idx),
                            self.history.select(histories, idx),
                            last_claim.index_select(0, idx),
                            path_probability.index_select(0, idx),
                        )
                    )
                stats["row_splits"] = int(stats["row_splits"]) + 1
                if profile:
                    actor = depth & 1
                    row = self._stream_profile_row(
                        profile_by_depth,
                        depth,
                        actor,
                        traverser,
                    )
                    row["row_splits"] = int(row["row_splits"]) + 1
                return torch.cat(values, dim=0)

            actor = depth & 1
            if profile:
                row = self._stream_profile_row(
                    profile_by_depth,
                    depth,
                    actor,
                    traverser,
                )
                row["calls"] = int(row["calls"]) + 1
                row["active_rows_in"] = int(row["active_rows_in"]) + rows
                row["peak_rows"] = max(int(row["peak_rows"]), rows)

            features = self._features(
                actor,
                deal_idx,
                histories,
                p1_counts,
                p2_counts,
            )
            legal_mask = self.legal_masks.index_select(0, last_claim + 1)
            regret_values, strategy = self._regrets_and_strategy(
                actor,
                features,
                legal_mask,
            )
            terminal_values = self._terminal_values(
                last_claim,
                deal_idx,
                caller=actor,
                traverser=traverser,
                total_counts=total_counts,
            )

            if actor != traverser:
                sampled_cols = torch.multinomial(strategy, 1).squeeze(1)
                continues = sampled_cols > 0
                parent_rows = continues.nonzero(as_tuple=False).squeeze(1)
                values = terminal_values.clone()
                if parent_rows.numel():
                    claim_ids = sampled_cols.index_select(0, parent_rows) - 1
                    child_values = evaluate_frontier(
                        depth + 1,
                        deal_idx.index_select(0, parent_rows),
                        self.history.append(
                            self.history.select(histories, parent_rows),
                            claim_ids,
                        ),
                        claim_ids,
                        path_probability.index_select(0, parent_rows),
                    )
                    values[continues] = child_values
                iteration_weight = (
                    1.0
                    if self.trainer.strategy_weighting == "uniform"
                    else float(self.trainer.iteration)
                )
                strategy_accumulator.append(
                    features,
                    strategy,
                    legal_mask,
                    path_probability.clamp_min(1e-12).reciprocal()
                    * iteration_weight,
                )
                if profile:
                    row["opponent_continuations"] = (
                        int(row["opponent_continuations"]) + int(parent_rows.numel())
                    )
                    row["strategy_records"] = (
                        int(row["strategy_records"]) + int(features.shape[0])
                    )
                    row["active_rows_out"] = (
                        int(row["active_rows_out"]) + int(parent_rows.numel())
                    )
                return values

            (
                parent_rows,
                claim_ids,
                child_inclusion,
                layer_full_edges,
                layer_sampled_edges,
            ) = self._claim_edges(
                legal_mask,
                regret_values,
                histories,
                (depth - traverser) // 2,
            )
            full_edges = int(layer_full_edges.item())
            sampled_edges = int(layer_sampled_edges)
            stats["full_claim_edges"] = int(stats["full_claim_edges"]) + full_edges
            stats["sampled_claim_edges"] = (
                int(stats["sampled_claim_edges"]) + sampled_edges
            )

            if self.action_baseline == "call":
                baseline_values = torch.where(
                    legal_mask[:, 0],
                    terminal_values,
                    torch.zeros_like(terminal_values),
                )
            else:
                baseline_values = torch.zeros_like(terminal_values)
            action_values = baseline_values[:, None] * legal_mask
            call_legal = legal_mask[:, 0]
            action_values[call_legal, 0] = terminal_values[call_legal]

            edge_count = int(parent_rows.numel())
            chunk_size = (
                self.trainer.traverser_action_chunk_size
                or self.trainer.traversal_live_row_budget
                or max(edge_count, 1)
            )
            if edge_count:
                for start in range(0, edge_count, chunk_size):
                    end = min(start + chunk_size, edge_count)
                    edge_idx = torch.arange(
                        start,
                        end,
                        dtype=torch.long,
                        device=self.device,
                    )
                    chunk_parent_rows = parent_rows.index_select(0, edge_idx)
                    chunk_claim_ids = claim_ids.index_select(0, edge_idx)
                    chunk_inclusion = child_inclusion.index_select(0, edge_idx)
                    child_values = evaluate_frontier(
                        depth + 1,
                        deal_idx.index_select(0, chunk_parent_rows),
                        self.history.append(
                            self.history.select(histories, chunk_parent_rows),
                            chunk_claim_ids,
                        ),
                        chunk_claim_ids,
                        path_probability.index_select(0, chunk_parent_rows)
                        * chunk_inclusion,
                    )
                    child_baselines = baseline_values.index_select(
                        0,
                        chunk_parent_rows,
                    )
                    action_values[
                        chunk_parent_rows,
                        chunk_claim_ids + 1,
                    ] = child_baselines + (
                        child_values - child_baselines
                    ) / chunk_inclusion
                    stats["edge_chunks"] = int(stats["edge_chunks"]) + 1

            node_values = (strategy * action_values).sum(dim=1)
            instant_regret = (action_values - node_values[:, None]) * legal_mask
            old_scaled = torch.relu(regret_values) * legal_mask
            targets = torch.relu(
                previous_scale * old_scaled + instant_scale * instant_regret
            ) * legal_mask
            weights = path_probability.clamp_min(1e-12).reciprocal()
            regret_accumulator.append(
                features,
                targets,
                legal_mask,
                weights,
            )
            stats["regret_weight_sum"] = (
                float(stats["regret_weight_sum"]) + float(weights.sum().item())
            )
            stats["regret_weight_square_sum"] = (
                float(stats["regret_weight_square_sum"])
                + float(weights.square().sum().item())
            )
            stats["regret_weight_count"] = (
                int(stats["regret_weight_count"]) + int(weights.numel())
            )
            stats["max_regret_weight"] = max(
                float(stats["max_regret_weight"]),
                float(weights.max().item()) if weights.numel() else 0.0,
            )

            if profile:
                row["traverser_claim_edges_full"] = (
                    int(row["traverser_claim_edges_full"]) + full_edges
                )
                row["traverser_claim_edges_expanded"] = (
                    int(row["traverser_claim_edges_expanded"]) + sampled_edges
                )
                row["edge_chunks"] = int(row["edge_chunks"]) + (
                    (edge_count + chunk_size - 1) // chunk_size
                    if edge_count
                    else 0
                )
                row["regret_records"] = (
                    int(row["regret_records"]) + int(features.shape[0])
                )
                row["active_rows_out"] = (
                    int(row["active_rows_out"]) + int(edge_count)
                )
            return node_values

        root_values = evaluate_frontier(
            0,
            current_deal,
            current_history,
            current_last,
            current_path_probability,
        )
        regret_accumulator.flush()
        strategy_accumulator.flush()

        result: Dict[str, object] = {
            "regret_records": regret_accumulator.count,
            "strategy_records": strategy_accumulator.count,
            "full_claim_edges": int(stats["full_claim_edges"]),
            "sampled_claim_edges": int(stats["sampled_claim_edges"]),
            "regret_weight_sum": float(stats["regret_weight_sum"]),
            "regret_weight_square_sum": float(stats["regret_weight_square_sum"]),
            "regret_weight_count": int(stats["regret_weight_count"]),
            "max_regret_weight": float(stats["max_regret_weight"]),
            "streamed_edge_chunks": int(stats["edge_chunks"]),
            "streamed_row_splits": int(stats["row_splits"]),
        }
        if profile:
            result["depth_profile"] = list(profile_by_depth.values())
            result["peak_rows"] = int(stats["peak_rows"])
            result["root_values"] = root_values.detach().cpu()
            result["peak_allocated_bytes"] = (
                int(torch.cuda.max_memory_allocated(self.device))
                if self.device.type == "cuda"
                else 0
            )
            result["peak_reserved_bytes"] = (
                int(torch.cuda.max_memory_reserved(self.device))
                if self.device.type == "cuda"
                else 0
            )
        return result

    def run_traversals(
        self,
        traverser: int,
        batch_size: int,
        *,
        profile: bool = False,
        commit_records: bool = True,
    ) -> Dict[str, object]:
        if self.trainer.traversal_streaming:
            return self._run_traversals_streaming(
                traverser,
                batch_size,
                profile=profile,
                commit_records=commit_records,
            )

        use_cuda_events = profile and self.device.type == "cuda"
        if use_cuda_events:
            torch.cuda.reset_peak_memory_stats(self.device)
        depth_profile: List[Dict[str, object]] = []
        profile_by_depth: Dict[int, Dict[str, object]] = {}

        p1_counts, p2_counts, total_counts = self._sample_deals(batch_size)
        current_deal = torch.arange(
            batch_size,
            dtype=torch.long,
            device=self.device,
        )
        current_history = self.history.zeros(batch_size)
        current_last = torch.full(
            (batch_size,),
            -1,
            dtype=torch.long,
            device=self.device,
        )
        current_path_probability = torch.ones(
            batch_size,
            dtype=torch.float32,
            device=self.device,
        )

        layers: List[Dict[str, torch.Tensor | int | str]] = []
        strategy_features: List[torch.Tensor] = []
        strategy_targets: List[torch.Tensor] = []
        strategy_masks: List[torch.Tensor] = []
        strategy_path_probabilities: List[torch.Tensor] = []
        full_claim_edges = torch.zeros(
            (),
            dtype=torch.long,
            device=self.device,
        )
        sampled_claim_edges = torch.zeros(
            (),
            dtype=torch.long,
            device=self.device,
        )

        for depth in range(self.k + 1):
            if self.history.rows(current_history) == 0:
                break

            forward_start = None
            forward_end = None
            cpu_forward_start = 0.0
            if use_cuda_events:
                forward_start = torch.cuda.Event(enable_timing=True)
                forward_end = torch.cuda.Event(enable_timing=True)
                forward_start.record()
            elif profile:
                cpu_forward_start = time.perf_counter()

            actor = depth & 1
            profile_row: Dict[str, object] | None = None
            if profile:
                profile_row = {
                    "depth": depth,
                    "actor": actor,
                    "traverser": traverser,
                    "active_rows_in": self.history.rows(current_history),
                    "traverser_claim_edges_full": 0,
                    "traverser_claim_edges_expanded": 0,
                    "opponent_continuations": 0,
                    "regret_records": 0,
                    "strategy_records": 0,
                    "backup_s": 0.0,
                }
            features = self._features(
                actor,
                current_deal,
                current_history,
                p1_counts,
                p2_counts,
            )
            legal_mask = self.legal_masks.index_select(0, current_last + 1)
            regret_values, strategy = self._regrets_and_strategy(
                actor,
                features,
                legal_mask,
            )
            terminal_values = self._terminal_values(
                current_last,
                current_deal,
                caller=actor,
                traverser=traverser,
                total_counts=total_counts,
            )

            if actor == traverser:
                (
                    parent_rows,
                    claim_ids,
                    child_inclusion,
                    layer_full_edges,
                    layer_sampled_edges,
                ) = self._claim_edges(
                    legal_mask,
                    regret_values,
                    current_history,
                    (depth - traverser) // 2,
                )
                full_claim_edges += layer_full_edges
                sampled_claim_edges += layer_sampled_edges
                if self.action_baseline == "call":
                    baseline_values = torch.where(
                        legal_mask[:, 0],
                        terminal_values,
                        torch.zeros_like(terminal_values),
                    )
                else:
                    baseline_values = torch.zeros_like(terminal_values)
                layers.append(
                    {
                        "kind": "traverser",
                        "depth": depth,
                        "features": features,
                        "legal_mask": legal_mask,
                        "regret_values": regret_values,
                        "strategy": strategy,
                        "call_values": terminal_values,
                        "baseline_values": baseline_values,
                        "child_parent_rows": parent_rows,
                        "child_cols": claim_ids + 1,
                        "child_inclusion": child_inclusion,
                        "path_probability": current_path_probability,
                    }
                )
                if profile_row is not None:
                    profile_row["traverser_claim_edges_full_tensor"] = (
                        layer_full_edges
                    )
                    profile_row["traverser_claim_edges_expanded"] = int(
                        layer_sampled_edges
                    )
                    profile_row["regret_records"] = int(features.shape[0])

                current_deal = current_deal.index_select(0, parent_rows)
                current_history = self.history.append(
                    self.history.select(current_history, parent_rows),
                    claim_ids,
                )
                current_last = claim_ids
                current_path_probability = (
                    current_path_probability.index_select(0, parent_rows)
                    * child_inclusion
                )
                if profile_row is not None:
                    profile_row["active_rows_out"] = self.history.rows(
                        current_history
                    )
                    profile_row["allocated_bytes"] = int(
                        torch.cuda.memory_allocated(self.device)
                    ) if self.device.type == "cuda" else 0
                    profile_row["reserved_bytes"] = int(
                        torch.cuda.memory_reserved(self.device)
                    ) if self.device.type == "cuda" else 0
                    if use_cuda_events:
                        forward_end.record()
                        profile_row["forward_events"] = (
                            forward_start,
                            forward_end,
                        )
                    else:
                        profile_row["forward_s"] = (
                            time.perf_counter() - cpu_forward_start
                        )
                    depth_profile.append(profile_row)
                    profile_by_depth[depth] = profile_row
                continue

            strategy_features.append(features)
            strategy_targets.append(strategy)
            strategy_masks.append(legal_mask)
            strategy_path_probabilities.append(current_path_probability)

            sampled_cols = torch.multinomial(strategy, 1).squeeze(1)
            continues = sampled_cols > 0
            parent_rows = continues.nonzero(as_tuple=False).squeeze(1)
            child_positions = torch.full(
                (len(sampled_cols),),
                -1,
                dtype=torch.long,
                device=self.device,
            )
            child_positions[continues] = torch.arange(
                parent_rows.numel(),
                dtype=torch.long,
                device=self.device,
            )
            layers.append(
                {
                    "kind": "opponent",
                    "depth": depth,
                    "terminal_values": terminal_values,
                    "continues": continues,
                    "child_positions": child_positions,
                }
            )
            if profile_row is not None:
                profile_row["opponent_continuations"] = int(parent_rows.numel())
                profile_row["strategy_records"] = int(features.shape[0])

            claim_ids = sampled_cols.index_select(0, parent_rows) - 1
            current_deal = current_deal.index_select(0, parent_rows)
            current_history = self.history.append(
                self.history.select(current_history, parent_rows),
                claim_ids,
            )
            current_last = claim_ids
            current_path_probability = current_path_probability.index_select(
                0,
                parent_rows,
            )
            if profile_row is not None:
                profile_row["active_rows_out"] = self.history.rows(
                    current_history
                )
                profile_row["allocated_bytes"] = int(
                    torch.cuda.memory_allocated(self.device)
                ) if self.device.type == "cuda" else 0
                profile_row["reserved_bytes"] = int(
                    torch.cuda.memory_reserved(self.device)
                ) if self.device.type == "cuda" else 0
                if use_cuda_events:
                    forward_end.record()
                    profile_row["forward_events"] = (
                        forward_start,
                        forward_end,
                    )
                else:
                    profile_row["forward_s"] = (
                        time.perf_counter() - cpu_forward_start
                    )
                depth_profile.append(profile_row)
                profile_by_depth[depth] = profile_row

        next_values: torch.Tensor | None = None
        regret_features: List[torch.Tensor] = []
        regret_targets: List[torch.Tensor] = []
        regret_masks: List[torch.Tensor] = []
        regret_weights: List[torch.Tensor] = []
        iteration = max(float(self.trainer.iteration), 1.0)
        previous_scale = (iteration - 1.0) / iteration
        instant_scale = 1.0 / iteration

        for layer in reversed(layers):
            backup_start = None
            backup_end = None
            cpu_backup_start = 0.0
            if use_cuda_events:
                backup_start = torch.cuda.Event(enable_timing=True)
                backup_end = torch.cuda.Event(enable_timing=True)
                backup_start.record()
            elif profile:
                cpu_backup_start = time.perf_counter()

            if layer["kind"] == "opponent":
                terminal_values = layer["terminal_values"]
                continues = layer["continues"]
                child_positions = layer["child_positions"]
                values = terminal_values.clone()
                if next_values is not None and next_values.numel():
                    values[continues] = next_values.index_select(
                        0,
                        child_positions[continues],
                    )
                next_values = values
                if profile:
                    row = profile_by_depth[int(layer["depth"])]
                    if use_cuda_events:
                        backup_end.record()
                        row["backup_events"] = (backup_start, backup_end)
                    else:
                        row["backup_s"] = time.perf_counter() - cpu_backup_start
                continue

            strategy = layer["strategy"]
            legal_mask = layer["legal_mask"]
            # Horvitz-Thompson action-value estimate. Unsampled actions retain
            # the baseline; sampled actions receive the inverse-inclusion
            # correction. With inclusion=1 this is exactly full expansion.
            baseline_values = layer["baseline_values"]
            action_values = (
                baseline_values[:, None] * legal_mask
            )
            call_legal = legal_mask[:, 0]
            action_values[call_legal, 0] = layer["call_values"][call_legal]
            child_parent_rows = layer["child_parent_rows"]
            if next_values is not None and next_values.numel():
                child_baselines = baseline_values.index_select(
                    0,
                    child_parent_rows,
                )
                action_values[
                    child_parent_rows,
                    layer["child_cols"],
                ] = child_baselines + (
                    next_values - child_baselines
                ) / layer["child_inclusion"]

            node_values = (strategy * action_values).sum(dim=1)
            instant_regret = (action_values - node_values[:, None]) * legal_mask
            old_scaled = torch.relu(layer["regret_values"]) * legal_mask
            targets = torch.relu(
                previous_scale * old_scaled + instant_scale * instant_regret
            ) * legal_mask
            regret_features.append(layer["features"])
            regret_targets.append(targets)
            regret_masks.append(legal_mask)
            # Deeper infosets exist in the sampled traversal only when every
            # preceding traverser action was selected. Inverse path reach
            # restores their contribution to the full-expansion regression
            # objective.
            regret_weights.append(
                layer["path_probability"].clamp_min(1e-12).reciprocal()
            )
            next_values = node_values
            if profile:
                row = profile_by_depth[int(layer["depth"])]
                if use_cuda_events:
                    backup_end.record()
                    row["backup_events"] = (backup_start, backup_end)
                else:
                    row["backup_s"] = time.perf_counter() - cpu_backup_start

        regret_count = 0
        regret_weight_sum = 0.0
        regret_weight_square_sum = 0.0
        max_regret_weight = 0.0
        if regret_features:
            features = torch.cat(regret_features, dim=0)
            targets = torch.cat(regret_targets, dim=0)
            masks = torch.cat(regret_masks, dim=0)
            weights = torch.cat(regret_weights, dim=0)
            regret_count = int(features.shape[0])
            regret_weight_sum = float(weights.sum().item())
            regret_weight_square_sum = float(weights.square().sum().item())
            max_regret_weight = float(weights.max().item())
            if commit_records:
                self.trainer._add_device_records(
                    self.trainer.regret_buffers[traverser],
                    self.trainer.regret_validation_buffers[traverser],
                    features,
                    targets,
                    masks,
                    weights,
                )

        strategy_count = 0
        if strategy_features:
            features = torch.cat(strategy_features, dim=0)
            targets = torch.cat(strategy_targets, dim=0)
            masks = torch.cat(strategy_masks, dim=0)
            path_weights = torch.cat(
                strategy_path_probabilities,
                dim=0,
            ).clamp_min(1e-12).reciprocal()
            strategy_count = int(features.shape[0])
            actor_pid = 1 - traverser
            iteration_weight = (
                1.0
                if self.trainer.strategy_weighting == "uniform"
                else float(self.trainer.iteration)
            )
            if commit_records:
                self.trainer._add_device_records(
                    self.trainer.strategy_buffers[actor_pid],
                    self.trainer.strategy_validation_buffers[actor_pid],
                    features,
                    targets,
                    masks,
                    path_weights * iteration_weight,
                )

        result: Dict[str, object] = {
            "regret_records": regret_count,
            "strategy_records": strategy_count,
            "full_claim_edges": int(full_claim_edges.item()),
            "sampled_claim_edges": int(sampled_claim_edges.item()),
            "regret_weight_sum": regret_weight_sum,
            "regret_weight_square_sum": regret_weight_square_sum,
            "regret_weight_count": regret_count,
            "max_regret_weight": max_regret_weight,
        }
        if profile:
            if use_cuda_events:
                torch.cuda.synchronize(self.device)
            for row in depth_profile:
                full_edges = row.pop(
                    "traverser_claim_edges_full_tensor",
                    None,
                )
                if full_edges is not None:
                    row["traverser_claim_edges_full"] = int(
                        full_edges.item()
                    )
                events = row.pop("forward_events", None)
                if events is not None:
                    row["forward_s"] = events[0].elapsed_time(events[1]) / 1000.0
                backup_events = row.pop("backup_events", None)
                if backup_events is not None:
                    row["backup_s"] = (
                        backup_events[0].elapsed_time(backup_events[1]) / 1000.0
                    )
            result["depth_profile"] = depth_profile
            result["peak_rows"] = max(
                (
                    max(
                        int(row["active_rows_in"]),
                        int(row["active_rows_out"]),
                    )
                    for row in depth_profile
                ),
                default=0,
            )
            result["peak_allocated_bytes"] = (
                int(torch.cuda.max_memory_allocated(self.device))
                if self.device.type == "cuda"
                else 0
            )
            result["peak_reserved_bytes"] = (
                int(torch.cuda.max_memory_reserved(self.device))
                if self.device.type == "cuda"
                else 0
            )
            if next_values is not None:
                result["root_values"] = next_values.detach().cpu()
        return result
