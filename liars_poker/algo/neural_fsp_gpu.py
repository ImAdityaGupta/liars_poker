from __future__ import annotations

from typing import Dict

import torch

from liars_poker.core import GameSpec, card_rank, generate_deck
from liars_poker.infoset import CALL
from liars_poker.policies.action_conditioned import (
    ActionConditionedPolicy,
    ActionConditionedQPolicy,
)
from liars_poker.policies.neural import InfosetEncoder
from liars_poker.env import rules_for_spec


class CompactStrategyReservoir:
    """Device-resident Algorithm-R reservoir for sampled policy actions."""

    def __init__(
        self,
        capacity: int,
        input_dim: int,
        device: str | torch.device,
    ) -> None:
        self.capacity = int(capacity)
        self.input_dim = int(input_dim)
        self.device = torch.device(device)
        self.features = torch.empty(
            (self.capacity, self.input_dim),
            dtype=torch.uint8,
            device=self.device,
        )
        self.actions = torch.empty(
            self.capacity,
            dtype=torch.int16,
            device=self.device,
        )
        self.legal_rows = torch.empty(
            self.capacity,
            dtype=torch.int16,
            device=self.device,
        )
        self.size = 0
        self.seen = 0

    def add_many(
        self,
        features: torch.Tensor,
        actions: torch.Tensor,
        legal_rows: torch.Tensor,
    ) -> None:
        features = features.to(self.device, dtype=torch.uint8)
        actions = actions.to(self.device, dtype=torch.int16)
        legal_rows = legal_rows.to(self.device, dtype=torch.int16)
        n = int(features.shape[0])
        if n == 0:
            return

        room = min(n, self.capacity - self.size)
        if room:
            start = self.size
            stop = start + room
            self.features[start:stop].copy_(features[:room])
            self.actions[start:stop].copy_(actions[:room])
            self.legal_rows[start:stop].copy_(legal_rows[:room])
            self.size = stop
            self.seen += room

        remaining = n - room
        if remaining <= 0:
            return

        bounds = torch.arange(
            self.seen + 1,
            self.seen + remaining + 1,
            dtype=torch.float64,
            device=self.device,
        )
        slots = torch.floor(torch.rand(remaining, device=self.device) * bounds).long()
        accepted = slots < self.capacity
        self.seen += remaining
        if not bool(accepted.any()):
            return

        accepted_slots = slots[accepted]
        accepted_sources = torch.arange(
            remaining,
            dtype=torch.long,
            device=self.device,
        )[accepted]
        unique_slots, inverse = torch.unique(
            accepted_slots,
            sorted=False,
            return_inverse=True,
        )
        last_source = torch.full(
            (len(unique_slots),),
            -1,
            dtype=torch.long,
            device=self.device,
        )
        last_source.scatter_reduce_(
            0,
            inverse,
            accepted_sources,
            reduce="amax",
            include_self=True,
        )
        source = room + last_source
        self.features[unique_slots] = features[source]
        self.actions[unique_slots] = actions[source]
        self.legal_rows[unique_slots] = legal_rows[source]

    def sample(
        self,
        batch_size: int,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n = min(int(batch_size), self.size)
        indices = torch.randint(
            self.size,
            (n,),
            device=self.device,
            generator=generator,
        )
        return (
            self.features.index_select(0, indices).float(),
            self.actions.index_select(0, indices).long(),
            self.legal_rows.index_select(0, indices).long(),
        )

    def state_dict(self) -> Dict[str, object]:
        return {
            "capacity": self.capacity,
            "input_dim": self.input_dim,
            "size": self.size,
            "seen": self.seen,
            "features": self.features[: self.size].detach().cpu(),
            "actions": self.actions[: self.size].detach().cpu(),
            "legal_rows": self.legal_rows[: self.size].detach().cpu(),
        }

    @classmethod
    def from_state_dict(
        cls,
        state: Dict[str, object],
        *,
        device: str | torch.device,
    ) -> "CompactStrategyReservoir":
        buffer = cls(
            int(state["capacity"]),
            int(state["input_dim"]),
            device,
        )
        buffer.size = int(state["size"])
        buffer.seen = int(state["seen"])
        buffer.features[: buffer.size].copy_(
            torch.as_tensor(state["features"], device=buffer.device)
        )
        buffer.actions[: buffer.size].copy_(
            torch.as_tensor(state["actions"], device=buffer.device)
        )
        buffer.legal_rows[: buffer.size].copy_(
            torch.as_tensor(state["legal_rows"], device=buffer.device)
        )
        return buffer


class GPUFSPStrategyCollector:
    """External-sampling collector for reach-correct FSP strategy records."""

    def __init__(
        self,
        spec: GameSpec,
        device: str | torch.device,
    ) -> None:
        self.spec = spec
        self.rules = rules_for_spec(spec)
        self.encoder = InfosetEncoder(spec)
        self.device = torch.device(device)
        self.k = self.encoder.k
        self.action_dim = self.encoder.action_dim
        self.ranks = spec.ranks
        self.history_shifts = torch.arange(
            self.k,
            dtype=torch.long,
            device=self.device,
        )
        deck_ranks = [card_rank(card, spec) - 1 for card in generate_deck(spec)]
        self.deck_ranks = torch.tensor(
            deck_ranks,
            dtype=torch.long,
            device=self.device,
        )
        self.legal_masks = self._build_legal_masks()

    def _build_legal_masks(self) -> torch.Tensor:
        masks = torch.zeros(
            (self.k + 1, self.action_dim),
            dtype=torch.bool,
            device=self.device,
        )
        for last_claim in range(-1, self.k):
            legal = self.rules.legal_actions_from_last(
                None if last_claim < 0 else last_claim
            )
            cols = [0 if action == CALL else action + 1 for action in legal]
            masks[last_claim + 1, cols] = True
        return masks

    def _sample_hands(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        draw_count = 2 * self.spec.hand_size
        order = torch.rand(
            (n, int(self.deck_ranks.numel())),
            device=self.device,
        ).topk(draw_count, dim=1).indices
        ranks = self.deck_ranks[order]
        p1_ranks = ranks[:, : self.spec.hand_size]
        p2_ranks = ranks[:, self.spec.hand_size :]
        p1 = torch.zeros(
            (n, self.ranks),
            dtype=torch.float32,
            device=self.device,
        )
        p2 = torch.zeros_like(p1)
        p1.scatter_add_(
            1,
            p1_ranks,
            torch.ones_like(p1_ranks, dtype=torch.float32),
        )
        p2.scatter_add_(
            1,
            p2_ranks,
            torch.ones_like(p2_ranks, dtype=torch.float32),
        )
        return p1, p2

    def _features(
        self,
        role: int,
        deal_idx: torch.Tensor,
        hids: torch.Tensor,
        p1_counts: torch.Tensor,
        p2_counts: torch.Tensor,
    ) -> torch.Tensor:
        hand = (p1_counts if role == 0 else p2_counts).index_select(0, deal_idx)
        history = ((hids[:, None] >> self.history_shifts[None, :]) & 1).float()
        return torch.cat((hand, history), dim=1)

    def _policy_actions(
        self,
        policy: ActionConditionedPolicy | ActionConditionedQPolicy,
        role: int,
        features: torch.Tensor,
        legal_mask: torch.Tensor,
        generator: torch.Generator,
    ) -> torch.Tensor:
        model = policy._model(role)
        with torch.inference_mode():
            scores = model.score_all(features, policy.action_features).float()
            scores = scores.masked_fill(~legal_mask, -torch.inf)
            if isinstance(policy, ActionConditionedPolicy):
                return torch.multinomial(
                    torch.softmax(scores, dim=1),
                    1,
                    generator=generator,
                ).squeeze(1)
            return scores.argmax(dim=1)

    def _collect_batch(
        self,
        policy: ActionConditionedPolicy | ActionConditionedQPolicy,
        role: int,
        episodes: int,
        training_buffer: CompactStrategyReservoir,
        validation_buffer: CompactStrategyReservoir | None,
        validation_fraction: float,
        generator: torch.Generator,
    ) -> Dict[str, int]:
        p1_counts, p2_counts = self._sample_hands(episodes)
        deal_idx = torch.arange(episodes, device=self.device)
        hids = torch.zeros(episodes, dtype=torch.long, device=self.device)
        last_claim = torch.full(
            (episodes,),
            -1,
            dtype=torch.long,
            device=self.device,
        )
        records = 0
        peak_states = episodes

        for depth in range(self.k + 1):
            if deal_idx.numel() == 0:
                break
            actor = depth & 1
            legal = self.legal_masks.index_select(0, last_claim + 1)

            if actor == role:
                features = self._features(
                    role,
                    deal_idx,
                    hids,
                    p1_counts,
                    p2_counts,
                )
                actions = self._policy_actions(
                    policy,
                    role,
                    features,
                    legal,
                    generator,
                )
                legal_rows = last_claim + 1
                if validation_buffer is not None and validation_fraction > 0.0:
                    use_validation = (
                        torch.rand(
                            len(features),
                            device=self.device,
                            generator=generator,
                        )
                        < validation_fraction
                    )
                    validation_buffer.add_many(
                        features[use_validation],
                        actions[use_validation],
                        legal_rows[use_validation],
                    )
                    use_training = ~use_validation
                    training_buffer.add_many(
                        features[use_training],
                        actions[use_training],
                        legal_rows[use_training],
                    )
                else:
                    training_buffer.add_many(features, actions, legal_rows)
                records += len(features)

                continues = actions > 0
                deal_idx = deal_idx[continues]
                hids = hids[continues]
                claims = actions[continues] - 1
                hids = hids | (torch.ones_like(claims) << claims)
                last_claim = claims
                continue

            claim_edges = legal[:, 1:].nonzero(as_tuple=False)
            parent_rows = claim_edges[:, 0]
            claims = claim_edges[:, 1]
            deal_idx = deal_idx.index_select(0, parent_rows)
            hids = hids.index_select(0, parent_rows) | (
                torch.ones_like(claims) << claims
            )
            last_claim = claims
            peak_states = max(peak_states, int(deal_idx.numel()))

        return {"records": records, "peak_states": peak_states}

    def collect(
        self,
        policy: ActionConditionedPolicy | ActionConditionedQPolicy,
        role: int,
        episodes: int,
        training_buffer: CompactStrategyReservoir,
        *,
        validation_buffer: CompactStrategyReservoir | None = None,
        validation_fraction: float = 0.0,
        batch_size: int = 256,
        generator: torch.Generator,
    ) -> Dict[str, int]:
        totals = {"episodes": 0, "records": 0, "peak_states": 0}
        remaining = int(episodes)
        while remaining:
            n = min(remaining, int(batch_size))
            stats = self._collect_batch(
                policy,
                role,
                n,
                training_buffer,
                validation_buffer,
                validation_fraction,
                generator,
            )
            totals["episodes"] += n
            totals["records"] += stats["records"]
            totals["peak_states"] = max(totals["peak_states"], stats["peak_states"])
            remaining -= n
        return totals
