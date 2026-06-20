from __future__ import annotations

import copy
import random
import time
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from liars_poker.core import GameSpec, card_rank, generate_deck
from liars_poker.env import rules_for_spec
from liars_poker.infoset import CALL
from liars_poker.policies.base import Policy
from liars_poker.policies.neural import InfosetEncoder, NeuralMLP, NeuralPolicy
from liars_poker.policies.neural_q import NeuralQPolicy
from liars_poker.policies.action_conditioned import ActionConditionedQPolicy
from liars_poker.policies.tabular_dense import DenseTabularPolicy


class DeviceTransitionReplay:
    """Fixed-capacity device-resident ring buffer for responder transitions."""

    def __init__(
        self,
        capacity: int,
        input_dim: int,
        action_dim: int,
        device: torch.device,
    ) -> None:
        self.capacity = int(capacity)
        self.input_dim = int(input_dim)
        self.action_dim = int(action_dim)
        self.device = device
        self.features = torch.empty((capacity, input_dim), dtype=torch.float32, device=device)
        self.actions = torch.empty(capacity, dtype=torch.long, device=device)
        self.rewards = torch.empty(capacity, dtype=torch.float32, device=device)
        self.next_features = torch.empty_like(self.features)
        self.next_legal_masks = torch.empty(
            (capacity, action_dim),
            dtype=torch.bool,
            device=device,
        )
        self.dones = torch.empty(capacity, dtype=torch.bool, device=device)
        self.size = 0
        self.position = 0
        self.seen = 0

    def add_many(
        self,
        features: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_features: torch.Tensor,
        next_legal_masks: torch.Tensor,
        dones: torch.Tensor,
    ) -> None:
        n = int(features.shape[0])
        if n <= 0:
            return
        self.seen += n
        if n >= self.capacity:
            features = features[-self.capacity :]
            actions = actions[-self.capacity :]
            rewards = rewards[-self.capacity :]
            next_features = next_features[-self.capacity :]
            next_legal_masks = next_legal_masks[-self.capacity :]
            dones = dones[-self.capacity :]
            n = self.capacity

        first = min(n, self.capacity - self.position)
        second = n - first
        sl = slice(self.position, self.position + first)
        self.features[sl].copy_(features[:first])
        self.actions[sl].copy_(actions[:first])
        self.rewards[sl].copy_(rewards[:first])
        self.next_features[sl].copy_(next_features[:first])
        self.next_legal_masks[sl].copy_(next_legal_masks[:first])
        self.dones[sl].copy_(dones[:first])
        if second:
            sl = slice(0, second)
            self.features[sl].copy_(features[first:])
            self.actions[sl].copy_(actions[first:])
            self.rewards[sl].copy_(rewards[first:])
            self.next_features[sl].copy_(next_features[first:])
            self.next_legal_masks[sl].copy_(next_legal_masks[first:])
            self.dones[sl].copy_(dones[first:])

        self.position = (self.position + n) % self.capacity
        self.size = min(self.capacity, self.size + n)

    def sample(
        self,
        batch_size: int,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, ...]:
        indices = torch.randint(
            self.size,
            (int(batch_size),),
            device=self.device,
            generator=generator,
        )
        return (
            self.features.index_select(0, indices),
            self.actions.index_select(0, indices),
            self.rewards.index_select(0, indices),
            self.next_features.index_select(0, indices),
            self.next_legal_masks.index_select(0, indices),
            self.dones.index_select(0, indices),
        )

    def state_dict(self) -> Dict[str, object]:
        used = self.size
        return {
            "capacity": self.capacity,
            "input_dim": self.input_dim,
            "action_dim": self.action_dim,
            "size": self.size,
            "position": self.position,
            "seen": self.seen,
            "features": self.features[:used].detach().cpu(),
            "actions": self.actions[:used].detach().cpu(),
            "rewards": self.rewards[:used].detach().cpu(),
            "next_features": self.next_features[:used].detach().cpu(),
            "next_legal_masks": self.next_legal_masks[:used].detach().cpu(),
            "dones": self.dones[:used].detach().cpu(),
        }

    @classmethod
    def from_state_dict(
        cls,
        state: Dict[str, object],
        device: torch.device,
    ) -> "DeviceTransitionReplay":
        buffer = cls(
            int(state["capacity"]),
            int(state["input_dim"]),
            int(state["action_dim"]),
            device,
        )
        buffer.size = int(state["size"])
        buffer.position = int(state["position"])
        buffer.seen = int(state["seen"])
        for name in (
            "features",
            "actions",
            "rewards",
            "next_features",
            "next_legal_masks",
            "dones",
        ):
            getattr(buffer, name)[: buffer.size].copy_(
                torch.as_tensor(state[name], device=device)
            )
        return buffer


class _FixedOpponent:
    def __init__(
        self,
        spec: GameSpec,
        policy: Policy,
        device: torch.device,
        legal_masks: torch.Tensor,
    ) -> None:
        self.spec = spec
        self.device = device
        self.legal_masks = legal_masks
        self.kind = ""

        if isinstance(policy, DenseTabularPolicy):
            if policy.spec != spec:
                raise ValueError("Opponent policy spec mismatch.")
            if not spec.suit_symmetry:
                raise ValueError(
                    "GPU dense-opponent lookup currently requires suit_symmetry=True."
                )
            self.kind = "dense"
            self.S = torch.as_tensor(policy.S, device=device)
            base = spec.hand_size + 1
            powers = torch.tensor(
                [base**rank for rank in range(spec.ranks)],
                dtype=torch.long,
                device=device,
            )
            self.hand_code_powers = powers
            lookup = torch.full(
                (base**spec.ranks,),
                -1,
                dtype=torch.long,
                device=device,
            )
            for idx, hand in enumerate(policy.hands):
                counts = [0] * spec.ranks
                for card in hand:
                    counts[card_rank(card, spec) - 1] += 1
                code = sum(counts[r] * (base**r) for r in range(spec.ranks))
                lookup[code] = idx
            self.hand_lookup = lookup
            return

        if isinstance(policy, (NeuralPolicy, NeuralQPolicy, ActionConditionedQPolicy)):
            if policy.spec != spec:
                raise ValueError("Opponent policy spec mismatch.")
            if isinstance(policy, ActionConditionedQPolicy):
                self.kind = "action_conditioned_q"
                self.action_features = policy.action_features.to(device)
            else:
                self.kind = "neural_q" if isinstance(policy, NeuralQPolicy) else "neural"
            self.models = [
                copy.deepcopy(policy.model_p1).to(device).eval(),
                copy.deepcopy(policy.model_p2).to(device).eval(),
            ]
            return

        raise TypeError(
            "NeuralBRTrainer currently supports DenseTabularPolicy, NeuralPolicy, "
            "NeuralQPolicy, or ActionConditionedQPolicy opponents."
        )

    def probabilities(
        self,
        actor: int,
        features: torch.Tensor,
        hids: torch.Tensor,
        hand_counts: torch.Tensor,
        legal_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.kind == "dense":
            codes = (hand_counts.long() * self.hand_code_powers).sum(dim=1)
            hand_idx = self.hand_lookup.index_select(0, codes)
            if bool((hand_idx < 0).any()):
                raise RuntimeError("Failed to map a sampled hand into dense policy indices.")
            probs = self.S[hids, hand_idx].float() * legal_mask
        else:
            with torch.inference_mode():
                if self.kind == "action_conditioned_q":
                    values = self.models[actor].score_all(
                        features,
                        self.action_features,
                    ).float()
                else:
                    values = self.models[actor](features).float()
            if self.kind in {"neural_q", "action_conditioned_q"}:
                best = values.masked_fill(~legal_mask, -torch.inf).argmax(dim=1)
                probs = torch.zeros_like(values).scatter_(1, best[:, None], 1.0)
            else:
                probs = torch.softmax(
                    values.masked_fill(~legal_mask, -torch.inf),
                    dim=1,
                )
        totals = probs.sum(dim=1, keepdim=True)
        return probs / totals.clamp_min(1e-12)


class NeuralBRTrainer:
    """Role-separated Double-DQN best responders against a frozen opponent."""

    CHECKPOINT_VERSION = 1

    KIND_RANK_HIGH = 0
    KIND_PAIR = 1
    KIND_TWO_PAIR = 2
    KIND_TRIPS = 3
    KIND_FULL_HOUSE = 4
    KIND_QUADS = 5

    def __init__(
        self,
        spec: GameSpec,
        opponent: Policy,
        *,
        hidden_sizes: Sequence[int] = (512, 512),
        device: str | torch.device | None = None,
        expansion: str = "all_actions",
        replay_capacity: int = 1_000_000,
        batch_size: int = 4096,
        learning_rate: float = 1e-3,
        train_steps: int = 100,
        warmup_transitions: int = 20_000,
        target_update_every: int = 500,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_decisions: int = 500_000,
        seed: int = 0,
    ) -> None:
        if expansion not in {"sampled", "all_actions"}:
            raise ValueError("expansion must be 'sampled' or 'all_actions'.")
        self.spec = spec
        self.rules = rules_for_spec(spec)
        self.encoder = InfosetEncoder(spec)
        self.device = torch.device(
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.hidden_sizes = tuple(int(size) for size in hidden_sizes)
        self.expansion = expansion
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self.train_steps = int(train_steps)
        self.warmup_transitions = int(warmup_transitions)
        self.target_update_every = int(target_update_every)
        self.epsilon_start = float(epsilon_start)
        self.epsilon_end = float(epsilon_end)
        self.epsilon_decay_decisions = int(epsilon_decay_decisions)
        self.seed = int(seed)
        self.iteration = 0
        self.decisions_seen = [0, 0]
        self.optimizer_steps = [0, 0]

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(seed)

        self.q_nets = [
            NeuralMLP(
                self.encoder.input_dim,
                self.encoder.action_dim,
                self.hidden_sizes,
            ).to(self.device)
            for _ in range(2)
        ]
        self.target_nets = [copy.deepcopy(model).eval() for model in self.q_nets]
        self.optimizers = [
            torch.optim.Adam(model.parameters(), lr=self.learning_rate)
            for model in self.q_nets
        ]
        self.replay = [
            DeviceTransitionReplay(
                replay_capacity,
                self.encoder.input_dim,
                self.encoder.action_dim,
                self.device,
            )
            for _ in range(2)
        ]

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
        self._build_claim_tensors()
        self.opponent = _FixedOpponent(
            spec,
            opponent,
            self.device,
            self.legal_masks,
        )

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

    def _build_claim_tensors(self) -> None:
        kinds = []
        rank_a = []
        rank_b = []
        for kind, value in self.rules.claims:
            if kind == "RankHigh":
                kinds.append(self.KIND_RANK_HIGH)
                rank_a.append(value - 1)
                rank_b.append(-1)
            elif kind == "Pair":
                kinds.append(self.KIND_PAIR)
                rank_a.append(value - 1)
                rank_b.append(-1)
            elif kind == "TwoPair":
                low, high = self.rules.two_pair_ranks[value]
                kinds.append(self.KIND_TWO_PAIR)
                rank_a.append(low - 1)
                rank_b.append(high - 1)
            elif kind == "Trips":
                kinds.append(self.KIND_TRIPS)
                rank_a.append(value - 1)
                rank_b.append(-1)
            elif kind == "FullHouse":
                trip, pair = self.rules.full_house_ranks[value]
                kinds.append(self.KIND_FULL_HOUSE)
                rank_a.append(trip - 1)
                rank_b.append(pair - 1)
            elif kind == "Quads":
                kinds.append(self.KIND_QUADS)
                rank_a.append(value - 1)
                rank_b.append(-1)
            else:
                raise ValueError(f"Unsupported claim kind: {kind}")
        self.claim_kind = torch.tensor(kinds, dtype=torch.long, device=self.device)
        self.claim_rank_a = torch.tensor(rank_a, dtype=torch.long, device=self.device)
        self.claim_rank_b = torch.tensor(rank_b, dtype=torch.long, device=self.device)

    def _sample_deals(
        self,
        n: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        draw_count = 2 * self.spec.hand_size
        order = torch.rand(
            (n, int(self.deck_ranks.numel())),
            device=self.device,
            generator=self.generator,
        ).topk(draw_count, dim=1).indices
        ranks = self.deck_ranks[order]
        p1_ranks = ranks[:, : self.spec.hand_size]
        p2_ranks = ranks[:, self.spec.hand_size :]
        p1 = torch.zeros((n, self.ranks), dtype=torch.float32, device=self.device)
        p2 = torch.zeros_like(p1)
        p1.scatter_add_(1, p1_ranks, torch.ones_like(p1_ranks, dtype=torch.float32))
        p2.scatter_add_(1, p2_ranks, torch.ones_like(p2_ranks, dtype=torch.float32))
        return p1, p2, p1 + p2

    def _features(
        self,
        actor: int,
        deal_idx: torch.Tensor,
        hids: torch.Tensor,
        p1_counts: torch.Tensor,
        p2_counts: torch.Tensor,
    ) -> torch.Tensor:
        hand = (p1_counts if actor == 0 else p2_counts).index_select(0, deal_idx)
        history = ((hids[:, None] >> self.history_shifts[None, :]) & 1).float()
        return torch.cat((hand, history), dim=1)

    def _terminal_rewards(
        self,
        last_claim: torch.Tensor,
        deal_idx: torch.Tensor,
        *,
        caller: int,
        responder: int,
        total_counts: torch.Tensor,
    ) -> torch.Tensor:
        safe_claim = last_claim.clamp_min(0)
        counts = total_counts.index_select(0, deal_idx)
        rows = torch.arange(len(last_claim), device=self.device)
        kind = self.claim_kind.index_select(0, safe_claim)
        rank_a = self.claim_rank_a.index_select(0, safe_claim)
        rank_b = self.claim_rank_b.index_select(0, safe_claim)
        count_a = counts[rows, rank_a]
        count_b = counts[rows, rank_b.clamp_min(0)]
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
            winner == responder,
            torch.ones(len(last_claim), device=self.device),
            -torch.ones(len(last_claim), device=self.device),
        )

    def _opponent_actions(
        self,
        actor: int,
        deal_idx: torch.Tensor,
        hids: torch.Tensor,
        last_claim: torch.Tensor,
        p1_counts: torch.Tensor,
        p2_counts: torch.Tensor,
    ) -> torch.Tensor:
        features = self._features(actor, deal_idx, hids, p1_counts, p2_counts)
        legal = self.legal_masks.index_select(0, last_claim + 1)
        hand_counts = (p1_counts if actor == 0 else p2_counts).index_select(
            0,
            deal_idx,
        )
        probs = self.opponent.probabilities(
            actor,
            features,
            hids,
            hand_counts,
            legal,
        )
        return torch.multinomial(probs, 1, generator=self.generator).squeeze(1)

    def epsilon(self, role: int) -> float:
        fraction = min(
            1.0,
            self.decisions_seen[role] / max(1, self.epsilon_decay_decisions),
        )
        return self.epsilon_start + fraction * (
            self.epsilon_end - self.epsilon_start
        )

    def _select_actions(
        self,
        role: int,
        features: torch.Tensor,
        legal_mask: torch.Tensor,
        epsilon: float,
    ) -> torch.Tensor:
        with torch.inference_mode():
            q = self.q_nets[role](features).float()
        greedy = q.masked_fill(~legal_mask, -torch.inf).argmax(dim=1)
        if epsilon <= 0.0:
            return greedy
        random_probs = legal_mask.float()
        random_probs /= random_probs.sum(dim=1, keepdim=True)
        random_cols = torch.multinomial(
            random_probs,
            1,
            generator=self.generator,
        ).squeeze(1)
        explore = torch.rand(
            len(features),
            device=self.device,
            generator=self.generator,
        ) < epsilon
        return torch.where(explore, random_cols, greedy)

    def _advance_actions(
        self,
        role: int,
        deal_idx: torch.Tensor,
        hids: torch.Tensor,
        last_claim: torch.Tensor,
        action_cols: torch.Tensor,
        p1_counts: torch.Tensor,
        p2_counts: torch.Tensor,
        total_counts: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        n = len(action_cols)
        rewards = torch.zeros(n, dtype=torch.float32, device=self.device)
        dones = torch.zeros(n, dtype=torch.bool, device=self.device)
        next_hids = hids.clone()
        next_last = last_claim.clone()

        calls = action_cols == 0
        if bool(calls.any()):
            rewards[calls] = self._terminal_rewards(
                last_claim[calls],
                deal_idx[calls],
                caller=role,
                responder=role,
                total_counts=total_counts,
            )
            dones[calls] = True

        raises = ~calls
        if bool(raises.any()):
            rows = raises.nonzero(as_tuple=False).squeeze(1)
            claims = action_cols.index_select(0, rows) - 1
            raised_hids = hids.index_select(0, rows) | (
                torch.ones_like(claims) << claims
            )
            branch_deals = deal_idx.index_select(0, rows)
            opponent_role = 1 - role
            opponent_cols = self._opponent_actions(
                opponent_role,
                branch_deals,
                raised_hids,
                claims,
                p1_counts,
                p2_counts,
            )
            opponent_calls = opponent_cols == 0
            if bool(opponent_calls.any()):
                call_rows = rows[opponent_calls]
                rewards[call_rows] = self._terminal_rewards(
                    claims[opponent_calls],
                    branch_deals[opponent_calls],
                    caller=opponent_role,
                    responder=role,
                    total_counts=total_counts,
                )
                dones[call_rows] = True

            opponent_raises = ~opponent_calls
            if bool(opponent_raises.any()):
                continue_rows = rows[opponent_raises]
                opponent_claims = opponent_cols[opponent_raises] - 1
                next_hids[continue_rows] = raised_hids[opponent_raises] | (
                    torch.ones_like(opponent_claims) << opponent_claims
                )
                next_last[continue_rows] = opponent_claims

        next_features = torch.zeros(
            (n, self.encoder.input_dim),
            dtype=torch.float32,
            device=self.device,
        )
        next_masks = torch.zeros(
            (n, self.action_dim),
            dtype=torch.bool,
            device=self.device,
        )
        continues = ~dones
        if bool(continues.any()):
            next_features[continues] = self._features(
                role,
                deal_idx[continues],
                next_hids[continues],
                p1_counts,
                p2_counts,
            )
            next_masks[continues] = self.legal_masks.index_select(
                0,
                next_last[continues] + 1,
            )
        return rewards, dones, next_hids, next_last, next_features, next_masks

    def _run_batch(
        self,
        role: int,
        episodes: int,
        *,
        collect: bool,
        expansion: str,
        epsilon: float,
    ) -> Dict[str, int]:
        p1_counts, p2_counts, total_counts = self._sample_deals(episodes)
        deal_idx = torch.arange(episodes, device=self.device)
        hids = torch.zeros(episodes, dtype=torch.long, device=self.device)
        last_claim = torch.full(
            (episodes,),
            -1,
            dtype=torch.long,
            device=self.device,
        )

        if role == 1:
            opponent_cols = self._opponent_actions(
                0,
                deal_idx,
                hids,
                last_claim,
                p1_counts,
                p2_counts,
            )
            claims = opponent_cols - 1
            hids = torch.ones_like(claims) << claims
            last_claim = claims

        wins = 0
        transitions = 0
        decisions = 0
        active_deals = deal_idx
        active_hids = hids
        active_last = last_claim

        while active_deals.numel():
            features = self._features(
                role,
                active_deals,
                active_hids,
                p1_counts,
                p2_counts,
            )
            legal = self.legal_masks.index_select(0, active_last + 1)
            chosen_cols = self._select_actions(role, features, legal, epsilon)
            n_states = len(features)
            decisions += n_states

            if expansion == "all_actions":
                edges = legal.nonzero(as_tuple=False)
                parent_rows = edges[:, 0]
                action_cols = edges[:, 1]
                branch_deals = active_deals.index_select(0, parent_rows)
                branch_hids = active_hids.index_select(0, parent_rows)
                branch_last = active_last.index_select(0, parent_rows)
                branch_features = features.index_select(0, parent_rows)
                branch_index = torch.full(
                    (n_states, self.action_dim),
                    -1,
                    dtype=torch.long,
                    device=self.device,
                )
                branch_index[parent_rows, action_cols] = torch.arange(
                    len(action_cols),
                    device=self.device,
                )
                selected = branch_index[
                    torch.arange(n_states, device=self.device),
                    chosen_cols,
                ]
            else:
                action_cols = chosen_cols
                branch_deals = active_deals
                branch_hids = active_hids
                branch_last = active_last
                branch_features = features
                selected = torch.arange(n_states, device=self.device)

            (
                rewards,
                dones,
                next_hids,
                next_last,
                next_features,
                next_masks,
            ) = self._advance_actions(
                role,
                branch_deals,
                branch_hids,
                branch_last,
                action_cols,
                p1_counts,
                p2_counts,
                total_counts,
            )

            if collect:
                self.replay[role].add_many(
                    branch_features,
                    action_cols,
                    rewards,
                    next_features,
                    next_masks,
                    dones,
                )
            transitions += len(action_cols)

            selected_done = dones.index_select(0, selected)
            selected_rewards = rewards.index_select(0, selected)
            wins += int(((selected_done) & (selected_rewards > 0)).sum().item())
            keep = ~selected_done
            selected = selected[keep]
            active_deals = branch_deals.index_select(0, selected)
            active_hids = next_hids.index_select(0, selected)
            active_last = next_last.index_select(0, selected)

        if collect:
            self.decisions_seen[role] += decisions
        return {
            "episodes": episodes,
            "wins": wins,
            "decisions": decisions,
            "transitions": transitions,
        }

    def collect_role(
        self,
        role: int,
        episodes: int,
        *,
        rollout_batch_size: int = 4096,
    ) -> Dict[str, int | float]:
        start = time.perf_counter()
        totals = {"episodes": 0, "wins": 0, "decisions": 0, "transitions": 0}
        remaining = int(episodes)
        epsilon = self.epsilon(role)
        while remaining:
            batch = min(remaining, int(rollout_batch_size))
            stats = self._run_batch(
                role,
                batch,
                collect=True,
                expansion=self.expansion,
                epsilon=epsilon,
            )
            for key in totals:
                totals[key] += stats[key]
            remaining -= batch
        return {
            **totals,
            "epsilon": epsilon,
            "collect_s": time.perf_counter() - start,
        }

    def train_role(self, role: int, *, steps: int | None = None) -> float:
        buffer = self.replay[role]
        if buffer.size < max(self.warmup_transitions, self.batch_size):
            return float("nan")
        model = self.q_nets[role]
        target_model = self.target_nets[role]
        optimizer = self.optimizers[role]
        losses = []
        model.train()
        for _ in range(self.train_steps if steps is None else int(steps)):
            (
                features,
                actions,
                rewards,
                next_features,
                next_masks,
                dones,
            ) = buffer.sample(self.batch_size, self.generator)
            predicted = model(features).gather(1, actions[:, None]).squeeze(1)
            with torch.no_grad():
                online_next = model(next_features).masked_fill(
                    ~next_masks,
                    -torch.inf,
                )
                next_actions = online_next.argmax(dim=1)
                target_next = target_model(next_features).gather(
                    1,
                    next_actions[:, None],
                ).squeeze(1)
                target_next = torch.where(dones, torch.zeros_like(target_next), target_next)
                targets = rewards + target_next
            loss = F.smooth_l1_loss(predicted, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            self.optimizer_steps[role] += 1
            if self.optimizer_steps[role] % self.target_update_every == 0:
                target_model.load_state_dict(model.state_dict())
            losses.append(float(loss.detach()))
        model.eval()
        return float(np.mean(losses))

    def run_iteration(
        self,
        *,
        episodes_per_role: int = 4096,
        rollout_batch_size: int = 4096,
    ) -> Dict[str, object]:
        self.iteration += 1
        role_records = []
        for role in (0, 1):
            collected = self.collect_role(
                role,
                episodes_per_role,
                rollout_batch_size=rollout_batch_size,
            )
            start = time.perf_counter()
            loss = self.train_role(role)
            collected["fit_s"] = time.perf_counter() - start
            collected["loss"] = loss
            collected["replay_size"] = self.replay[role].size
            collected["replay_seen"] = self.replay[role].seen
            role_records.append(collected)
        return {"iter": self.iteration, "roles": role_records}

    def evaluate_role(
        self,
        role: int,
        episodes: int,
        *,
        rollout_batch_size: int = 8192,
    ) -> Dict[str, float]:
        totals = {"episodes": 0, "wins": 0, "decisions": 0, "transitions": 0}
        remaining = int(episodes)
        while remaining:
            batch = min(remaining, int(rollout_batch_size))
            stats = self._run_batch(
                role,
                batch,
                collect=False,
                expansion="sampled",
                epsilon=0.0,
            )
            for key in totals:
                totals[key] += stats[key]
            remaining -= batch
        return {
            **totals,
            "win_rate": totals["wins"] / max(1, totals["episodes"]),
        }

    def policy(self) -> NeuralQPolicy:
        policy = NeuralQPolicy(
            self.spec,
            hidden_sizes=self.hidden_sizes,
            device=self.device,
        )
        policy.model_p1.load_state_dict(self.q_nets[0].state_dict())
        policy.model_p2.load_state_dict(self.q_nets[1].state_dict())
        return policy.eval()

    def checkpoint_state(self) -> Dict[str, object]:
        return {
            "version": self.CHECKPOINT_VERSION,
            "spec_json": self.spec.to_json(),
            "config": {
                "hidden_sizes": self.hidden_sizes,
                "expansion": self.expansion,
                "replay_capacity": self.replay[0].capacity,
                "batch_size": self.batch_size,
                "learning_rate": self.learning_rate,
                "train_steps": self.train_steps,
                "warmup_transitions": self.warmup_transitions,
                "target_update_every": self.target_update_every,
                "epsilon_start": self.epsilon_start,
                "epsilon_end": self.epsilon_end,
                "epsilon_decay_decisions": self.epsilon_decay_decisions,
                "seed": self.seed,
            },
            "iteration": self.iteration,
            "decisions_seen": self.decisions_seen,
            "optimizer_steps": self.optimizer_steps,
            "q_nets": [model.state_dict() for model in self.q_nets],
            "target_nets": [model.state_dict() for model in self.target_nets],
            "optimizers": [optimizer.state_dict() for optimizer in self.optimizers],
            "replay": [buffer.state_dict() for buffer in self.replay],
            "generator_state": self.generator.get_state().cpu(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
            "python_rng_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(),
        }

    def save_checkpoint(self, path: str | Path) -> None:
        torch.save(self.checkpoint_state(), Path(path))

    @classmethod
    def load_checkpoint(
        cls,
        path: str | Path,
        opponent: Policy,
        *,
        device: str | torch.device | None = None,
    ) -> "NeuralBRTrainer":
        state = torch.load(Path(path), map_location="cpu", weights_only=False)
        import json

        spec_data = json.loads(state["spec_json"])
        spec = GameSpec(
            ranks=int(spec_data["ranks"]),
            suits=int(spec_data["suits"]),
            hand_size=int(spec_data["hand_size"]),
            claim_kinds=tuple(spec_data["claim_kinds"]),
            suit_symmetry=bool(spec_data["suit_symmetry"]),
        )
        trainer = cls(spec, opponent, device=device, **state["config"])
        trainer.iteration = int(state["iteration"])
        trainer.decisions_seen = list(state["decisions_seen"])
        trainer.optimizer_steps = list(state["optimizer_steps"])
        for model, saved in zip(trainer.q_nets, state["q_nets"]):
            model.load_state_dict(saved)
        for model, saved in zip(trainer.target_nets, state["target_nets"]):
            model.load_state_dict(saved)
        for optimizer, saved in zip(trainer.optimizers, state["optimizers"]):
            optimizer.load_state_dict(saved)
        trainer.replay = [
            DeviceTransitionReplay.from_state_dict(saved, trainer.device)
            for saved in state["replay"]
        ]
        trainer.generator.set_state(state["generator_state"])
        torch.set_rng_state(state["torch_rng_state"])
        if state["cuda_rng_state"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda_rng_state"])
        random.setstate(state["python_rng_state"])
        np.random.set_state(state["numpy_rng_state"])
        return trainer
