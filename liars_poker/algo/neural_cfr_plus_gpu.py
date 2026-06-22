from __future__ import annotations

from typing import Dict, List

import torch

from liars_poker.core import card_rank, generate_deck
from liars_poker.infoset import CALL


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
        self.action_priority_count = trainer.traverser_action_priority_count
        self.action_baseline = trainer.traverser_action_baseline

        deck_ranks = [
            card_rank(card, self.spec) - 1 for card in generate_deck(self.spec)
        ]
        self.deck_ranks = torch.tensor(
            deck_ranks,
            dtype=torch.long,
            device=self.device,
        )
        self.history_shifts = torch.arange(
            self.k,
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
        hids: torch.Tensor,
        p1_counts: torch.Tensor,
        p2_counts: torch.Tensor,
    ) -> torch.Tensor:
        hand_counts = (p1_counts if actor == 0 else p2_counts).index_select(
            0,
            deal_idx,
        )
        history = ((hids[:, None] >> self.history_shifts[None, :]) & 1).float()
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
        if (
            (
                self.action_sample_count is None
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
                self.action_sample_count,
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
            random_scores = torch.rand(
                claim_legal.shape,
                dtype=torch.float32,
                device=self.device,
            ).masked_fill(~random_eligible, -1.0)
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
            scores = torch.rand(
                claim_legal.shape,
                dtype=torch.float32,
                device=self.device,
            ).masked_fill(~claim_legal, -1.0)
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

    def run_traversals(self, traverser: int, batch_size: int) -> Dict[str, int]:
        p1_counts, p2_counts, total_counts = self._sample_deals(batch_size)
        current_deal = torch.arange(
            batch_size,
            dtype=torch.long,
            device=self.device,
        )
        current_hid = torch.zeros(
            batch_size,
            dtype=torch.long,
            device=self.device,
        )
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
            if current_hid.numel() == 0:
                break

            actor = depth & 1
            features = self._features(
                actor,
                current_deal,
                current_hid,
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

                current_deal = current_deal.index_select(0, parent_rows)
                current_hid = current_hid.index_select(0, parent_rows) | (
                    torch.ones_like(claim_ids) << claim_ids
                )
                current_last = claim_ids
                current_path_probability = (
                    current_path_probability.index_select(0, parent_rows)
                    * child_inclusion
                )
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
                    "terminal_values": terminal_values,
                    "continues": continues,
                    "child_positions": child_positions,
                }
            )

            claim_ids = sampled_cols.index_select(0, parent_rows) - 1
            current_deal = current_deal.index_select(0, parent_rows)
            current_hid = current_hid.index_select(0, parent_rows) | (
                torch.ones_like(claim_ids) << claim_ids
            )
            current_last = claim_ids
            current_path_probability = current_path_probability.index_select(
                0,
                parent_rows,
            )

        next_values: torch.Tensor | None = None
        regret_features: List[torch.Tensor] = []
        regret_targets: List[torch.Tensor] = []
        regret_masks: List[torch.Tensor] = []
        regret_weights: List[torch.Tensor] = []
        iteration = float(self.trainer.iteration)
        previous_scale = (iteration - 1.0) / iteration
        instant_scale = 1.0 / iteration

        for layer in reversed(layers):
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
            self.trainer._add_device_records(
                self.trainer.strategy_buffers[actor_pid],
                self.trainer.strategy_validation_buffers[actor_pid],
                features,
                targets,
                masks,
                path_weights * iteration_weight,
            )

        return {
            "regret_records": regret_count,
            "strategy_records": strategy_count,
            "full_claim_edges": int(full_claim_edges.item()),
            "sampled_claim_edges": int(sampled_claim_edges.item()),
            "regret_weight_sum": regret_weight_sum,
            "regret_weight_square_sum": regret_weight_square_sum,
            "regret_weight_count": regret_count,
            "max_regret_weight": max_regret_weight,
        }
