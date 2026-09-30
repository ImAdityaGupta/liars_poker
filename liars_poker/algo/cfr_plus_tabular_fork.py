"""Fast 18-claim continuation with stored regrets and neural averaging.

The source regret networks become frozen priors. Their output is cached on
first visit; thereafter only sampled, visited information sets change.
The original strategy networks and their replay buffers continue unchanged.
"""

from __future__ import annotations

from itertools import combinations_with_replacement
from pathlib import Path

import torch

from liars_poker.algo.deep_cfr_plus import DeepCFRPlusTrainer
from liars_poker.algo.regret_readers import TableRegretReader


class TabularRegretFork(DeepCFRPlusTrainer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if (self.spec.ranks, self.spec.hand_size, self.encoder.k) != (4, 2, 18):
            raise ValueError("The compact regret table is specific to the 18-claim game")
        if self.device.type != "cpu" or self.traversal_backend != "gpu_native":
            raise ValueError("This fork requires CPU gpu_native traversal")
        if (self.regret_target_mode != "aggregate_then_clip"
                or self.regret_increment_reach_mode != "none"
                or self.regret_accumulation_mode not in {"normalized", "cumulative"}):
            raise ValueError("This fork requires conditional aggregate-then-clip targets")
        if self.validation_fraction:
            raise ValueError("This fork requires validation_fraction=0")
        if (self.traverser_action_sample_count is not None
                or self.traverser_action_sample_fraction is not None
                or self.traverser_action_sample_schedule is not None):
            raise ValueError("This fork requires full traverser-action expansion")

        self.regret_table: torch.Tensor | None = None
        self.table_initialized: torch.Tensor | None = None
        counts = []
        for hand in combinations_with_replacement(range(self.spec.ranks), 2):
            row = [0] * self.spec.ranks
            for rank in hand:
                row[rank] += 1
            counts.append(row)
        self.hand_counts = torch.tensor(counts, dtype=torch.float32)
        self.n_rank_hands = len(counts)
        self.hand_powers = 3 ** torch.arange(self.spec.ranks, dtype=torch.long)
        self.history_powers = 2 ** torch.arange(self.encoder.k, dtype=torch.long)
        hand_lut = torch.full((3 ** self.spec.ranks,), -1, dtype=torch.long)
        for index, row in enumerate(self.hand_counts.long()):
            hand_lut[int((row * self.hand_powers).sum())] = index
        self.hand_lut = hand_lut

    def _table_indices(self, features: torch.Tensor) -> torch.Tensor:
        if features.device.type != "cpu":
            raise ValueError("Regret table lookup must stay on CPU")
        hand_code = features[:, :self.spec.ranks].long() @ self.hand_powers
        history = features[:, self.spec.ranks:].long() @ self.history_powers
        return history * self.n_rank_hands + self.hand_lut[hand_code]

    def activate_regret_table(self) -> None:
        """Set up lazy caching of the source network's frozen predictions."""
        if self.regret_table is not None:
            raise RuntimeError("Regret table has already been initialized")
        rows = (1 << self.encoder.k) * self.n_rank_hands
        self.regret_table = torch.empty((rows, self.encoder.action_dim), dtype=torch.float32)
        self.table_initialized = torch.zeros(rows, dtype=torch.bool)
        self.regret_reader = TableRegretReader(
            self.regret_table,
            self.table_initialized,
            self._table_indices,
            seed_from_network=lambda pid, x: self._forward(self.regret_nets[pid], x),
            before_read=self._before_table_read,
        )

    def _before_table_read(self, keys: torch.Tensor) -> None:
        """Hook for table variants that discount old rows when read."""

    def _train_regret(self, pid: int, traversals_per_player: int) -> float:
        _ = traversals_per_player
        if self.regret_table is None:
            raise RuntimeError("Compile or restore the regret table first")
        buffer = self.regret_buffers[pid]
        n = buffer.size
        if n != buffer.seen:
            raise RuntimeError("Regret records overflowed; cannot form the complete per-iteration mean")
        if not n:
            return 0.0
        # The traverser has stored old + sampled conditional advantage for
        # each visit. Mean raw targets at identical infosets, then clip once.
        keys, inverse = torch.unique(
            self._table_indices(buffer.features[:n]), return_inverse=True
        )
        weights = buffer.weights[:n].double()
        totals = torch.zeros(len(keys), dtype=torch.float64)
        totals.index_add_(0, inverse, weights)
        sums = torch.zeros((len(keys), self.encoder.action_dim), dtype=torch.float64)
        sums.index_add_(0, inverse, buffer.targets[:n].double() * weights[:, None])
        updated = (sums / totals.clamp_min(1e-12)[:, None]).clamp_min_(0).float()
        with torch.no_grad():
            self.regret_table[keys] = updated
            self.table_initialized[keys] = True
        return 0.0

    def checkpoint_dict(self) -> dict:
        if self.regret_table is None or self.table_initialized is None:
            raise RuntimeError("Cannot checkpoint an uninitialized fork")
        state = super().checkpoint_dict()
        keys = self.table_initialized.nonzero(as_tuple=False).squeeze(1)
        state["tabular_regret_fork"] = {
            "format": 3,
            "keys": keys,
            "values": self.regret_table.index_select(0, keys),
        }
        return state

    @classmethod
    def load_fork_checkpoint(cls, path: str | Path) -> "TabularRegretFork":
        # The base loader restores all neural strategy state and RNG state.
        # The second read restores the table without altering that state.
        trainer = super().load_checkpoint(path, device="cpu")
        state = torch.load(path, map_location="cpu", weights_only=False)
        fork_state = state.get("tabular_regret_fork")
        if fork_state is None or fork_state.get("format") != 3:
            raise ValueError("Not a tabular regret fork checkpoint")
        keys = fork_state["keys"]
        values = fork_state["values"]
        rows = (1 << trainer.encoder.k) * trainer.n_rank_hands
        if (keys.dtype != torch.long or keys.ndim != 1
                or values.dtype != torch.float32
                or tuple(values.shape) != (len(keys), trainer.encoder.action_dim)
                or (len(keys) and (int(keys.min()) < 0 or int(keys.max()) >= rows))):
            raise ValueError("Invalid tabular regret table in checkpoint")
        trainer.activate_regret_table()
        trainer.regret_table[keys] = values
        trainer.table_initialized[keys] = True
        return trainer
