#!/usr/bin/env python3
"""Evaluate exact and online-neural averages from one exact4096 checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
REPO = Path("/root/liars_poker_discount") if Path("/root/liars_poker_discount").exists() else ROOT
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import torch

from liars_poker.serialization import save_policy
from run_cfr_plus_18_offline_average_fitting import (
    evaluate,
    exact_average,
    neural_policy,
)
from liars_poker.core import GameSpec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    args.output.mkdir(parents=True, exist_ok=True)

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    spec_data = dict(state["spec"])
    spec_data["claim_kinds"] = tuple(spec_data["claim_kinds"])
    spec = GameSpec(**spec_data)
    progress = state["experiment_progress"]
    iteration = int(state["iteration"])
    measured_min = float(progress["measured_training_s"]) / 60.0
    reference = exact_average(spec, state["exact_average_observer"])
    online = neural_policy(spec, state["strategy_nets"], "cpu")

    rows = []
    for name, policy in (("exact", reference), ("online", online)):
        save_policy(policy, str(args.output / name))
        metrics = evaluate(policy, reference)
        row = {"name": name, "iteration": iteration,
               "measured_training_min": measured_min, **metrics}
        rows.append(row)
        print(json.dumps(row), flush=True)

    (args.output / "results.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    source = {
        "checkpoint": str(args.checkpoint),
        "iteration": iteration,
        "measured_training_min": measured_min,
        "strategy_buffer_sizes": [int(x["size"]) for x in state["strategy_buffers"]],
        "strategy_buffer_seen": [int(x["seen"]) for x in state["strategy_buffers"]],
        "evaluation_threads": args.threads,
    }
    (args.output / "source.json").write_text(json.dumps(source, indent=2),
                                               encoding="utf-8")


if __name__ == "__main__":
    main()
