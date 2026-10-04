#!/usr/bin/env python3
"""Read-only summary of saved regret tables for the vanilla CFR comparison."""

from __future__ import annotations

import argparse
from pathlib import Path
import torch


def summarize(path: Path) -> None:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    table = payload["tabular_regret_fork"]
    keys = table["keys"]
    values = table["values"]
    n_rank_hands = 10
    history = keys // n_rank_hands
    last = torch.floor(torch.log2(history.clamp_min(1).float())).long()
    last[history == 0] = -1
    claims = torch.arange(values.shape[1] - 1)
    legal = torch.empty_like(values, dtype=torch.bool)
    legal[:, 0] = history > 0
    legal[:, 1:] = claims[None, :] > last[:, None]
    positive = ((values > 0) & legal).any(dim=1)
    legal_values = values[legal]
    print({
        "arm": path.parent.name,
        "iteration": payload["iteration"],
        "initialized_rows": len(keys),
        "no_positive_legal_fraction": float((~positive).float().mean()),
        "negative_legal_fraction": float((legal_values < 0).float().mean()),
        "mean_legal_regret": float(legal_values.mean()),
        "median_legal_regret": float(legal_values.median()),
        "min_legal_regret": float(legal_values.min()),
        "max_legal_regret": float(legal_values.max()),
    }, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="+", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    for path in args.checkpoints:
        summarize(path)


if __name__ == "__main__":
    main()
