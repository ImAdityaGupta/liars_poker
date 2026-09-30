"""Regret-target arithmetic shared by recursive and batched CFR+ traversals.

The caller supplies raw old regrets and sampled conditional advantages.
Grouping, replay weighting, and fitting remain in the trainer. The tabular
discount trainer stores advantages instead of targets through the same record
interface; neither path allocates an extra per-record history column.
"""

from __future__ import annotations

import numpy as np
import torch


def make_regret_target(
    old_raw: np.ndarray | torch.Tensor,
    advantage: np.ndarray | torch.Tensor,
    legal_mask: np.ndarray | torch.Tensor,
    *,
    iteration: int,
    accumulation_mode: str,
    target_mode: str,
) -> np.ndarray | torch.Tensor:
    """Apply the existing prior scaling and record-level clipping rules."""
    if isinstance(old_raw, torch.Tensor):
        old_positive = torch.relu(old_raw) * legal_mask
    else:
        old_positive = np.maximum(old_raw, 0.0)
        old_positive[~legal_mask] = 0.0
    # The batched traverser is also called directly before the first trainer
    # iteration by inspection tools; it historically treated that as t=1.
    t = max(float(iteration), 1.0)
    if accumulation_mode == "cumulative":
        prior = old_positive if t > 1 else (
            torch.zeros_like(old_positive)
            if isinstance(old_positive, torch.Tensor)
            else np.zeros_like(old_positive)
        )
        raw = prior + advantage
    elif accumulation_mode == "normalized":
        previous_scale = (t - 1.0) / t
        if isinstance(old_positive, torch.Tensor):
            raw = previous_scale * old_positive + (1.0 / t) * advantage
        else:
            # Preserve the recursive path's in-place float32 accumulation.
            raw = previous_scale * old_positive
            raw += advantage / (iteration if iteration > 0 else t)
    else:
        raise ValueError(f"Unknown regret accumulation mode: {accumulation_mode}")

    if isinstance(raw, torch.Tensor):
        target = torch.relu(raw) if target_mode == "clip_each_record" else raw
        return target * legal_mask
    target = np.maximum(raw, 0.0) if target_mode == "clip_each_record" else raw
    target = target.astype(np.float32)
    target[~legal_mask] = 0.0
    return target
