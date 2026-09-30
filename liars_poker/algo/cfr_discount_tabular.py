"""From-scratch sampled conditional CFR variants for the 18-claim game.

Traversal records contain sampled conditional advantages directly. The table
groups visits by information set and applies one update to each visited row.
No iteration normalization is used.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from liars_poker.algo.cfr_plus_tabular_fork import TabularRegretFork


RULES = {"cfr", "cfr_plus", "dcfr_plus", "dcfr_exact", "dcfr_visited"}


class TabularDiscountTrainer(TabularRegretFork):
    """Sparse-checkpointed regret table with exact or visited-only discount."""

    def __init__(self, *args, update_rule: str = "cfr_plus", **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if update_rule not in RULES:
            raise ValueError(f"Unknown update rule: {update_rule}")
        if self.regret_accumulation_mode != "cumulative":
            raise ValueError("Discount comparison requires cumulative regret units")
        self.update_rule = update_rule
        self.last_discount_iteration: torch.Tensor | None = None
        self._log_positive = np.zeros(1024, dtype=np.float64)
        self._log_negative = np.zeros(1024, dtype=np.float64)
        self._prefix_built_through = 0

    def activate_regret_table(self) -> None:
        # No frozen network prior: every information set begins with zero regret.
        super().activate_regret_table()
        self.regret_table.zero_()
        self.last_discount_iteration = torch.zeros(
            len(self.regret_table), dtype=torch.int32
        )
        self.regret_reader.seed_from_network = None

    def _ensure_prefix(self, through: int) -> None:
        if through <= self._prefix_built_through:
            return
        if through >= len(self._log_positive):
            capacity = max(through + 1, 2 * len(self._log_positive))
            self._log_positive = np.pad(
                self._log_positive, (0, capacity - len(self._log_positive))
            )
            self._log_negative = np.pad(
                self._log_negative, (0, capacity - len(self._log_negative))
            )
        for t in range(self._prefix_built_through + 1, through + 1):
            if self.update_rule == "dcfr_plus":
                # d_1 = 0, but no nonzero row exists before iteration 1.
                positive_log = -math.log1p(1.0 / ((t - 1) ** 2)) if t > 1 else 0.0
                negative_log = 0.0
            else:
                # DCFR discounts the *old* row before the current increment.
                # No nonzero row exists before t=1.
                positive_log = -math.log1p((t - 1) ** -1.5) if t > 1 else 0.0
                negative_log = -math.log(2.0) if t > 1 else 0.0
            self._log_positive[t] = self._log_positive[t - 1] + positive_log
            self._log_negative[t] = self._log_negative[t - 1] + negative_log
        self._prefix_built_through = through

    def _materialize(self, keys: torch.Tensor, through: int) -> None:
        if self.update_rule not in {"dcfr_plus", "dcfr_exact"} or through <= 0:
            return
        unique = torch.unique(keys)
        last = self.last_discount_iteration[unique]
        stale = self.table_initialized[unique] & (last < through)
        if not stale.any():
            return
        unique, last = unique[stale], last[stale]
        self._ensure_prefix(through)
        old_steps = last.numpy().astype(np.int64)
        positive_factor = torch.from_numpy(
            np.exp(self._log_positive[through] - self._log_positive[old_steps])
        ).float()
        values = self.regret_table[unique]
        if self.update_rule == "dcfr_plus":
            updated = values * positive_factor[:, None]
        else:
            negative_factor = torch.from_numpy(
                np.exp(self._log_negative[through] - self._log_negative[old_steps])
            ).float()
            updated = torch.where(values >= 0, values * positive_factor[:, None],
                                  values * negative_factor[:, None])
        self.regret_table[unique] = updated
        self.last_discount_iteration[unique] = through

    def _before_table_read(self, keys: torch.Tensor) -> None:
        # At iteration t, unvisited rows have received discounts through t-1.
        # A row updated earlier in the current alternating iteration has
        # last_discount_iteration=t and is already current.
        self._materialize(keys, self.iteration - 1)

    def make_regret_record(
        self, old_raw: torch.Tensor, advantage: torch.Tensor,
        legal_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Table rules consume conditional advantages directly, with no prior."""
        _ = old_raw
        return advantage * legal_mask

    def _train_regret(self, pid: int, traversals_per_player: int) -> float:
        _ = traversals_per_player
        if self.regret_table is None:
            raise RuntimeError("Activate the zero table before training")
        buffer = self.regret_buffers[pid]
        n = buffer.size
        if n != buffer.seen:
            raise RuntimeError("Regret buffer overflowed within one player update")
        if n == 0:
            return 0.0
        features = torch.as_tensor(buffer.features[:n])
        advantages = torch.as_tensor(buffer.targets[:n])
        weights = torch.as_tensor(buffer.weights[:n])
        if not torch.all(weights == 1):
            raise RuntimeError("Arm-4 conditional mean requires unit visit weights")
        keys, inverse = torch.unique(
            self._table_indices(features), sorted=True, return_inverse=True
        )
        totals = torch.zeros(len(keys), dtype=torch.float64)
        totals.index_add_(0, inverse, torch.ones(n, dtype=torch.float64))
        sums = torch.zeros((len(keys), self.encoder.action_dim), dtype=torch.float64)
        sums.index_add_(0, inverse, advantages.double())
        increment = (sums / totals.clamp_min(1e-12)[:, None]).float()

        self._materialize(keys, self.iteration - 1)
        old = self.regret_table.index_select(0, keys)
        t = self.iteration
        if self.update_rule == "cfr":
            updated = old + increment
        elif self.update_rule == "cfr_plus":
            updated = (old + increment).clamp_min(0)
        elif self.update_rule == "dcfr_plus":
            discount = 0.0 if t == 1 else (t - 1) ** 2 / ((t - 1) ** 2 + 1)
            updated = (discount * old + increment).clamp_min(0)
        else:
            positive_discount = 0.0 if t == 1 else 1.0 / (1.0 + (t - 1) ** -1.5)
            discounted_old = torch.where(old >= 0, old * positive_discount,
                                         old * 0.5)
            updated = discounted_old + increment
        with torch.no_grad():
            self.regret_table[keys] = updated
            self.table_initialized[keys] = True
            self.last_discount_iteration[keys] = t
        return 0.0

    def checkpoint_dict(self) -> dict:
        state = super().checkpoint_dict()
        keys = state["tabular_regret_fork"]["keys"]
        state["tabular_discount"] = {
            "format": 1,
            "update_rule": self.update_rule,
            "last_discount_iteration": self.last_discount_iteration[keys],
        }
        return state

    @classmethod
    def load_fork_checkpoint(cls, path: str | Path) -> "TabularDiscountTrainer":
        trainer = super().load_fork_checkpoint(path)
        state = torch.load(path, map_location="cpu", weights_only=False)
        extra = state.get("tabular_discount")
        if extra is None or extra.get("format") != 1 or extra.get("update_rule") not in RULES:
            raise ValueError("Not a tabular discount checkpoint")
        keys = state["tabular_regret_fork"]["keys"]
        last = extra["last_discount_iteration"]
        if (last.dtype != torch.int32 or last.shape != keys.shape
                or (len(last) and (int(last.min()) < 1 or int(last.max()) > trainer.iteration))):
            raise ValueError("Invalid lazy-discount checkpoint state")
        trainer.update_rule = extra["update_rule"]
        trainer.last_discount_iteration[keys] = last
        trainer._experiment_progress = state.get("experiment_progress")
        return trainer
