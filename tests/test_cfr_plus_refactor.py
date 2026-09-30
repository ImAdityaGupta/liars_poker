"""Behaviour checks for shared CFR+ targets and explicit regret readers."""

from __future__ import annotations

import gc
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from liars_poker.algo.cfr_plus_targets import make_regret_target
from liars_poker.algo.deep_cfr_plus import DeepCFRPlusTrainer
from liars_poker.algo.cfr_discount_tabular import TabularDiscountTrainer
from liars_poker.algo.cfr_plus_tabular_fork import TabularRegretFork
from liars_poker.algo.regret_readers import NetworkRegretReader, TableRegretReader
from liars_poker.core import GameSpec


class TargetEquivalence(unittest.TestCase):
    def test_numpy_target_matches_previous_formulas(self) -> None:
        old = np.array([-3.0, 2.25, 4.0], dtype=np.float32)
        advantage = np.array([0.5, -3.0, 7.0], dtype=np.float32)
        mask = np.array([True, True, False])
        for iteration in (1, 2, 19):
            for units in ("normalized", "cumulative"):
                for mode in ("clip_each_record", "aggregate_then_clip", "clip_on_read"):
                    with self.subTest(iteration=iteration, units=units, mode=mode):
                        old_scaled = np.maximum(old, 0.0)
                        old_scaled[~mask] = 0.0
                        if units == "cumulative":
                            expected = (old_scaled if iteration > 1 else np.zeros_like(old_scaled)) + advantage
                        else:
                            expected = ((iteration - 1.0) / iteration) * old_scaled
                            expected += advantage / iteration
                        if mode == "clip_each_record":
                            expected = np.maximum(expected, 0.0)
                        expected = expected.astype(np.float32)
                        expected[~mask] = 0.0
                        actual = make_regret_target(
                            old, advantage, mask, iteration=iteration,
                            accumulation_mode=units, target_mode=mode,
                        )
                        np.testing.assert_array_equal(actual, expected)

    def test_torch_target_matches_previous_formulas(self) -> None:
        old = torch.tensor([[-3.0, 2.25, 4.0]])
        advantage = torch.tensor([[0.5, -3.0, 7.0]])
        mask = torch.tensor([[True, True, False]])
        for iteration in (0, 1, 2, 19):
            t = max(float(iteration), 1.0)
            for units in ("normalized", "cumulative"):
                for mode in ("clip_each_record", "aggregate_then_clip", "clip_on_read"):
                    with self.subTest(iteration=iteration, units=units, mode=mode):
                        old_scaled = torch.relu(old) * mask
                        if units == "cumulative":
                            prior = old_scaled if t > 1 else torch.zeros_like(old_scaled)
                            raw = prior + advantage
                        else:
                            raw = ((t - 1.0) / t) * old_scaled + (1.0 / t) * advantage
                        expected = (torch.relu(raw) if mode == "clip_each_record" else raw) * mask
                        actual = make_regret_target(
                            old, advantage, mask, iteration=iteration,
                            accumulation_mode=units, target_mode=mode,
                        )
                        self.assertTrue(torch.equal(actual, expected))


class ReaderEquivalence(unittest.TestCase):
    def test_network_reader_uses_selected_network(self) -> None:
        networks = [torch.nn.Linear(2, 2, bias=False), torch.nn.Linear(2, 2, bias=False)]
        with torch.no_grad():
            networks[0].weight.fill_(1.0)
            networks[1].weight.fill_(2.0)
        reader = NetworkRegretReader(networks, lambda network, x: network(x))
        x = torch.tensor([[1.0, 3.0]])
        self.assertTrue(torch.equal(reader.read(0, x), torch.tensor([[4.0, 4.0]])))
        self.assertTrue(torch.equal(reader.read(1, x), torch.tensor([[8.0, 8.0]])))

    def test_table_reader_seeds_duplicates_once_and_applies_read_hook(self) -> None:
        table = torch.zeros(4, 2)
        initialized = torch.zeros(4, dtype=torch.bool)
        seeded = []
        hooks = []

        def seed(pid, x):
            seeded.append((pid, x.clone()))
            return torch.stack((-x[:, 0], x[:, 0]), dim=1)

        reader = TableRegretReader(
            table, initialized, lambda x: x[:, 0].long(),
            seed_from_network=seed,
            before_read=lambda keys: hooks.append(keys.clone()),
        )
        features = torch.tensor([[1.0], [2.0], [1.0]])
        actual = reader.read(1, features)
        self.assertEqual(len(seeded), 1)
        self.assertEqual(len(seeded[0][1]), 2)
        self.assertTrue(torch.equal(actual, torch.tensor([[0., 1.], [0., 2.], [0., 1.]])))
        self.assertTrue(torch.equal(hooks[0], torch.tensor([1, 2, 1])))
        reader.read(1, features[0])
        self.assertEqual(len(seeded), 1)


class TabularDiscountRecords(unittest.TestCase):
    def test_discount_rows_consume_advantage_without_old_target(self) -> None:
        for rule in ("cfr", "cfr_plus", "dcfr_plus", "dcfr_exact", "dcfr_visited"):
            with self.subTest(rule=rule):
                trainer = object.__new__(TabularDiscountTrainer)
                trainer.update_rule = rule
                trainer.iteration = 1
                trainer.encoder = SimpleNamespace(action_dim=2)
                trainer.regret_table = torch.zeros((3, 2))
                trainer.table_initialized = torch.zeros(3, dtype=torch.bool)
                trainer.last_discount_iteration = torch.zeros(3, dtype=torch.int32)
                trainer._log_positive = np.zeros(8, dtype=np.float64)
                trainer._log_negative = np.zeros(8, dtype=np.float64)
                trainer._prefix_built_through = 0
                trainer._table_indices = lambda features: features[:, 0].long()
                trainer.regret_buffers = [SimpleNamespace(
                    size=2, seen=2,
                    features=torch.tensor([[1.], [1.]]),
                    targets=torch.tensor([[3., -2.], [1., 0.]]),
                    weights=torch.ones(2),
                )]
                record = trainer.make_regret_record(
                    torch.tensor([[1000., -1000.]]),
                    torch.tensor([[2., -1.]]),
                    torch.tensor([[True, True]]),
                )
                self.assertTrue(torch.equal(record, torch.tensor([[2., -1.]])))
                trainer._train_regret(0, 2)
                expected = torch.tensor([2., -1.]) if rule in {
                    "cfr", "dcfr_exact", "dcfr_visited"
                } else torch.tensor([2., 0.])
                self.assertTrue(torch.equal(trainer.regret_table[1], expected))
                trainer.iteration = 3
                trainer._materialize(torch.tensor([1]), through=2)
                if rule == "dcfr_exact":
                    expected = torch.tensor([1., -0.5])
                elif rule == "dcfr_plus":
                    expected = torch.tensor([1., 0.])
                self.assertTrue(torch.equal(trainer.regret_table[1], expected))


class TrainerCheckpoint(unittest.TestCase):
    def test_cpu_batched_iteration_and_resume(self) -> None:
        torch.set_num_threads(1)
        spec = GameSpec(ranks=3, suits=2, hand_size=1,
                        claim_kinds=("RankHigh", "Pair"), suit_symmetry=True)
        trainer = DeepCFRPlusTrainer(
            spec, device="cpu", seed=17,
            regret_hidden_sizes=(16,), strategy_hidden_sizes=(16,),
            regret_buffer_capacity=10_000, strategy_buffer_capacity=1_000,
            learning_rate=1e-3, batch_size=8,
            regret_train_steps=1, strategy_train_steps=1,
            regret_target_mode="clip_each_record", traversal_backend="gpu_native",
            traversal_batch_size=4, validation_fraction=0,
        )
        first = trainer.run_iteration(traversals_per_player=4)
        self.assertEqual(first["iteration"], 1)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "trainer.pt"
            trainer.save_checkpoint(checkpoint)
            restored = DeepCFRPlusTrainer.load_checkpoint(checkpoint, device="cpu")
        self.assertEqual(restored.iteration, trainer.iteration)
        for old, new in zip(trainer.regret_nets, restored.regret_nets):
            for p_old, p_new in zip(old.parameters(), new.parameters()):
                self.assertTrue(torch.equal(p_old, p_new))
        second = restored.run_iteration(traversals_per_player=4)
        self.assertEqual(second["iteration"], 2)

    def test_streamed_cumulative_clip_on_read(self) -> None:
        torch.set_num_threads(1)
        spec = GameSpec(ranks=3, suits=2, hand_size=1,
                        claim_kinds=("RankHigh", "Pair"), suit_symmetry=True)
        trainer = DeepCFRPlusTrainer(
            spec, device="cpu", seed=29,
            regret_hidden_sizes=(16,), strategy_hidden_sizes=(16,),
            regret_buffer_capacity=10_000, strategy_buffer_capacity=1_000,
            learning_rate=1e-3, batch_size=8,
            regret_train_steps=1, strategy_train_steps=1,
            regret_target_mode="clip_on_read", regret_positive_weight=0,
            regret_accumulation_mode="cumulative",
            strategy_weighting="quadratic",
            traversal_backend="gpu_native", traversal_streaming=True,
            traversal_batch_size=4, validation_fraction=0,
        )
        first = trainer.run_iteration(traversals_per_player=4)
        self.assertEqual(first["iteration"], 1)
        self.assertGreater(sum(first["new_regret_records"]), 0)
        self.assertEqual(trainer._strategy_record_weight(), 1.0)
        trainer.run_iteration(traversals_per_player=4)
        self.assertEqual(trainer._strategy_record_weight(), 4.0)

    def test_18_claim_table_reader_checkpoint_round_trip(self) -> None:
        torch.set_num_threads(1)
        spec = GameSpec(ranks=4, suits=4, hand_size=2,
                        claim_kinds=("RankHigh", "Pair", "TwoPair", "Trips"),
                        suit_symmetry=True)
        trainer = TabularRegretFork(
            spec, device="cpu", seed=7,
            regret_hidden_sizes=(16,), strategy_hidden_sizes=(16,),
            regret_buffer_capacity=100, strategy_buffer_capacity=100,
            regret_train_steps=0, strategy_train_steps=0,
            regret_target_mode="aggregate_then_clip",
            regret_accumulation_mode="cumulative",
            traversal_backend="gpu_native", traversal_batch_size=1,
            validation_fraction=0,
        )
        trainer.activate_regret_table()
        features = torch.zeros((1, trainer.encoder.input_dim))
        features[0, 0] = 2.0
        first = trainer.regret_values_tensor(0, features).clone()
        key = trainer._table_indices(features)[0]
        self.assertTrue(bool(trainer.table_initialized[key]))
        self.assertTrue(torch.equal(first, trainer.regret_values_tensor(0, features)))
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "fork.pt"
            trainer.save_checkpoint(checkpoint)
            del trainer
            gc.collect()
            restored = TabularRegretFork.load_fork_checkpoint(checkpoint)
            self.assertTrue(torch.equal(first, restored.regret_values_tensor(0, features)))
            self.assertTrue(bool(restored.table_initialized[key]))

    def test_discount_table_lazy_state_checkpoint_round_trip(self) -> None:
        torch.set_num_threads(1)
        spec = GameSpec(ranks=4, suits=4, hand_size=2,
                        claim_kinds=("RankHigh", "Pair", "TwoPair", "Trips"),
                        suit_symmetry=True)
        trainer = TabularDiscountTrainer(
            spec, device="cpu", seed=7, update_rule="dcfr_exact",
            regret_hidden_sizes=(16,), strategy_hidden_sizes=(16,),
            regret_buffer_capacity=100_000, strategy_buffer_capacity=1_000,
            regret_train_steps=0, strategy_train_steps=0,
            regret_target_mode="aggregate_then_clip",
            regret_accumulation_mode="cumulative",
            traversal_backend="gpu_native", traversal_batch_size=1,
            validation_fraction=0,
        )
        trainer.activate_regret_table()
        features = torch.zeros((1, trainer.encoder.input_dim))
        features[0, 0] = 2.0
        key = trainer._table_indices(features)[0]
        self.assertFalse(bool(trainer.table_initialized[key]))
        self.assertTrue(torch.equal(
            trainer.regret_values_tensor(0, features),
            torch.zeros((1, trainer.encoder.action_dim)),
        ))
        trainer.regret_table[key, :2] = torch.tensor([2., -1.])
        trainer.table_initialized[key] = True
        trainer.last_discount_iteration[key] = 1
        trainer.iteration = 3
        before = trainer.regret_values_tensor(0, features).clone()
        self.assertTrue(torch.equal(before[0, :2], torch.tensor([1., -0.5])))
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "discount.pt"
            trainer.save_checkpoint(checkpoint)
            del trainer
            gc.collect()
            restored = TabularDiscountTrainer.load_fork_checkpoint(checkpoint)
            self.assertEqual(restored.update_rule, "dcfr_exact")
            self.assertEqual(int(restored.last_discount_iteration[key]), 2)
            self.assertTrue(torch.equal(before, restored.regret_values_tensor(0, features)))
            row = restored.run_iteration(traversals_per_player=1)
            self.assertEqual(row["iteration"], 4)
            self.assertGreater(sum(row["new_regret_records"]), 0)


if __name__ == "__main__":
    unittest.main()
