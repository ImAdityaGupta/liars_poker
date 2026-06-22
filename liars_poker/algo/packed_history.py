from __future__ import annotations

import math

import torch


class PackedHistory:
    """Packed claim-history bits with a scalar fast path through 63 claims."""

    BITS_PER_WORD = 63

    def __init__(
        self,
        claim_count: int,
        device: str | torch.device,
    ) -> None:
        self.claim_count = int(claim_count)
        if self.claim_count <= 0:
            raise ValueError("claim_count must be positive.")
        self.device = torch.device(device)
        self.word_count = math.ceil(self.claim_count / self.BITS_PER_WORD)
        claim_ids = torch.arange(
            self.claim_count,
            dtype=torch.long,
            device=self.device,
        )
        self.claim_words = torch.div(
            claim_ids,
            self.BITS_PER_WORD,
            rounding_mode="floor",
        )
        self.claim_masks = torch.bitwise_left_shift(
            torch.ones_like(claim_ids),
            claim_ids.remainder(self.BITS_PER_WORD),
        )

    @property
    def scalar(self) -> bool:
        return self.word_count == 1

    def zeros(self, rows: int) -> torch.Tensor:
        shape = (int(rows),) if self.scalar else (int(rows), self.word_count)
        return torch.zeros(shape, dtype=torch.long, device=self.device)

    @staticmethod
    def rows(history: torch.Tensor) -> int:
        return int(history.shape[0])

    @staticmethod
    def select(history: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
        return history.index_select(0, rows)

    @staticmethod
    def repeat_interleave(history: torch.Tensor, repeats: int) -> torch.Tensor:
        return history.repeat_interleave(int(repeats), dim=0)

    def append(
        self,
        history: torch.Tensor,
        claim_ids: torch.Tensor,
    ) -> torch.Tensor:
        claim_ids = claim_ids.long()
        masks = self.claim_masks.index_select(0, claim_ids)
        if self.scalar:
            return torch.bitwise_or(history, masks)

        result = history.clone()
        words = self.claim_words.index_select(0, claim_ids)
        rows = torch.arange(
            len(claim_ids),
            dtype=torch.long,
            device=self.device,
        )
        result[rows, words] = torch.bitwise_or(
            result[rows, words],
            masks,
        )
        return result

    def from_claims(self, claim_ids: torch.Tensor) -> torch.Tensor:
        return self.append(self.zeros(len(claim_ids)), claim_ids)

    def features(
        self,
        history: torch.Tensor,
        *,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        if self.scalar:
            bits = torch.bitwise_and(
                history[:, None],
                self.claim_masks[None, :],
            )
        else:
            words = history.index_select(1, self.claim_words)
            bits = torch.bitwise_and(words, self.claim_masks[None, :])
        return bits.ne(0).to(dtype=dtype)

