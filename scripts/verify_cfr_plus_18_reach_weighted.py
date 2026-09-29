#!/usr/bin/env python3
"""Check visit-fraction target algebra and a real CPU traversal/checkpoint."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from liars_poker.algo.deep_cfr_plus import DeepCFRPlusTrainer, DeviceRecentBuffer
from liars_poker.core import GameSpec


class FixedOld(torch.nn.Module):
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.tensor([0.8, 0.4], device=features.device).expand(len(features), -1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-spec-roots", default="",
                        help="comma-separated traversal budgets for one real 18-claim iteration each")
    parser.add_argument("--full-spec-cumulative", action="store_true",
                        help="smoke all three cumulative-regret configurations on the 18-claim game")
    args = parser.parse_args()
    torch.set_num_threads(1)
    buffer = DeviceRecentBuffer(8, 2, 2, torch.device("cpu"))
    features = torch.tensor([[0., 1.], [0., 1.], [1., 0.]])
    old = torch.tensor([0.8, 0.4])
    fresh = torch.tensor([[1., -1.], [3., 1.], [-2., 2.]])
    raw = 0.75 * old + 0.25 * fresh
    buffer.add_many(features, raw, torch.ones_like(raw, dtype=torch.bool),
                    torch.ones(3))
    DeepCFRPlusTrainer._aggregate_regret_targets(
        buffer, model=FixedOld(), iteration=4, roots=4,
    )
    expected_a = torch.relu(0.75 * old + (2 / 4) * 0.25 * fresh[:2].mean(0))
    expected_b = torch.relu(0.75 * old + (1 / 4) * 0.25 * fresh[2])
    torch.testing.assert_close(buffer.targets[:2], expected_a.expand(2, -1))
    torch.testing.assert_close(buffer.targets[2], expected_b)
    print("PASS grouped target weights only the fresh increment by N/K", flush=True)

    cumulative = DeviceRecentBuffer(8, 2, 2, torch.device("cpu"))
    cumulative.add_many(features, old + fresh, torch.ones_like(fresh, dtype=torch.bool),
                        torch.ones(3))
    DeepCFRPlusTrainer._aggregate_regret_targets(
        cumulative, model=FixedOld(), iteration=4, roots=4,
        accumulation_mode="cumulative",
    )
    expected_cum_a = torch.relu(old + (2 / 4) * fresh[:2].mean(0))
    expected_cum_b = torch.relu(old + (1 / 4) * fresh[2])
    torch.testing.assert_close(cumulative.targets[:2], expected_cum_a.expand(2, -1))
    torch.testing.assert_close(cumulative.targets[2], expected_cum_b)
    print("PASS cumulative target applies N/K only to the fresh increment", flush=True)

    spec = GameSpec(ranks=2, suits=2, hand_size=1,
                    claim_kinds=("RankHigh", "Pair"), suit_symmetry=True)
    trainer = DeepCFRPlusTrainer(
        spec, device="cpu", seed=17,
        regret_hidden_sizes=(16, 16), strategy_hidden_sizes=(16, 16),
        regret_buffer_capacity=10_000, strategy_buffer_capacity=10_000,
        batch_size=16, regret_train_steps=1, strategy_train_steps=1,
        regret_target_mode="aggregate_then_clip",
        regret_increment_reach_mode="visit_fraction",
        regret_accumulation_mode="cumulative",
        traversal_backend="gpu_native", traversal_batch_size=8,
        validation_fraction=0.0, fused_optimizer=False,
    )
    row = trainer.run_iteration(traversals_per_player=8)
    assert row["iteration"] == 1 and all(math.isfinite(x) for x in row["regret_loss"])
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "checkpoint.pt"
        trainer.save_checkpoint(path)
        restored = DeepCFRPlusTrainer.load_checkpoint(path, device="cpu")
        assert restored.iteration == 1
        assert restored.regret_increment_reach_mode == "visit_fraction"
        assert restored.regret_accumulation_mode == "cumulative"
    print("PASS real traversal, fitting, and checkpoint restore", flush=True)

    if args.full_spec_cumulative:
        from run_cfr_plus_18_target_order_cpu_overnight import make_trainer

        torch.set_num_threads(8)
        configs = ((1024, "visit_fraction", 500_000),
                   (4096, "visit_fraction", 4_000_000),
                   (4096, "none", 4_000_000))
        for roots, reach_mode, capacity in configs:
            trainer = make_trainer(
                "aggregate_then_clip", 17, reach_mode, capacity, "cumulative"
            )
            row = trainer.run_iteration(traversals_per_player=roots)
            assert row["iteration"] == 1
            assert all(math.isfinite(x) for x in row["regret_loss"])
            assert all(buffer.seen <= buffer.capacity for buffer in trainer.regret_buffers)
            print(f"PASS cumulative 18-claim iteration K={roots}, "
                  f"reach={reach_mode}, regret records="
                  f"{row['new_regret_records']}", flush=True)
            del trainer
    elif args.full_spec_roots:
        from run_cfr_plus_18_target_order_cpu_overnight import make_trainer

        torch.set_num_threads(8)
        for roots in (int(value) for value in args.full_spec_roots.split(",")):
            trainer = make_trainer("aggregate_then_clip", 17, "visit_fraction")
            row = trainer.run_iteration(traversals_per_player=roots)
            assert row["iteration"] == 1
            assert all(math.isfinite(x) for x in row["regret_loss"])
            assert all(buffer.seen <= buffer.capacity for buffer in trainer.regret_buffers)
            print(f"PASS 18-claim iteration K={roots}, regret records="
                  f"{row['new_regret_records']}", flush=True)
            del trainer


if __name__ == "__main__":
    main()
