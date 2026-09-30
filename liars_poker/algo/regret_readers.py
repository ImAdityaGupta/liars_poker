"""Regret-value sources used by traversal and current-policy evaluation.

Readers do not own replay buffers or the average-strategy network. They make
the source of current regrets explicit without changing checkpoint formats.
"""

from __future__ import annotations

from typing import Callable, Protocol

import torch


class RegretReader(Protocol):
    def read(self, pid: int, features: torch.Tensor) -> torch.Tensor: ...


class NetworkRegretReader:
    def __init__(self, networks, forward: Callable) -> None:
        self.networks = networks
        self.forward = forward

    def read(self, pid: int, features: torch.Tensor) -> torch.Tensor:
        return self.forward(self.networks[pid], features)


class TableRegretReader:
    """Dense table with optional lazy network seeding and read-time decay."""

    def __init__(
        self,
        table: torch.Tensor,
        initialized: torch.Tensor,
        key_for: Callable[[torch.Tensor], torch.Tensor],
        *,
        seed_from_network: Callable[[int, torch.Tensor], torch.Tensor] | None = None,
        before_read: Callable[[torch.Tensor], None] | None = None,
    ) -> None:
        self.table = table
        self.initialized = initialized
        self.key_for = key_for
        self.seed_from_network = seed_from_network
        self.before_read = before_read

    def read(self, pid: int, features: torch.Tensor) -> torch.Tensor:
        single = features.ndim == 1
        batch = features.unsqueeze(0) if single else features
        keys = self.key_for(batch)
        if self.seed_from_network is not None:
            missing = ~self.initialized.index_select(0, keys)
            if missing.any():
                missing_rows = missing.nonzero(as_tuple=False).squeeze(1)
                unique, inverse = torch.unique(
                    keys.index_select(0, missing_rows), return_inverse=True
                )
                first = torch.full((len(unique),), len(keys), dtype=torch.long)
                first.scatter_reduce_(0, inverse, missing_rows, reduce="amin")
                with torch.inference_mode():
                    predictions = self.seed_from_network(
                        pid, batch.index_select(0, first)
                    ).float().relu()
                    self.table[unique] = predictions
                    self.initialized[unique] = True
        if self.before_read is not None:
            self.before_read(keys)
        values = self.table.index_select(0, keys)
        return values[0] if single else values
