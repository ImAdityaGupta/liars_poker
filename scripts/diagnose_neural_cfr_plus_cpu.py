#!/usr/bin/env python3
"""CPU-only frozen-policy oracle check for the neural CFR+ GPU-native traverser.

The root strategy is temporarily made one-hot for each legal action. Replaying
the same Torch RNG stream reconstructs the action-value vector produced by the
actual traverser, without changing its production code. Future strategies are
uniform. This script does no neural fitting and needs no GPU.
"""

from __future__ import annotations

import argparse
from functools import lru_cache
import json
from pathlib import Path
import sys
import types

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from liars_poker.algo.deep_cfr_plus import DeepCFRPlusTrainer
from liars_poker.algo.cfr_plus_dense import CFRPlusDense
from liars_poker.algo.neural_cfr_plus_gpu import GPUDeepCFRPlusTraverser
from liars_poker.core import GameSpec
from liars_poker.env import resolve_call_winner, rules_for_spec
from liars_poker.infoset import CALL


SPEC = GameSpec(
    ranks=3,
    suits=2,
    hand_size=1,
    claim_kinds=("RankHigh", "Pair"),
    suit_symmetry=True,
)


def exact_root_action_values() -> np.ndarray:
    """Exact values under uniform continuation, conditional on P1's rank."""
    rules = rules_for_spec(SPEC)
    actions = rules.legal_actions_from_last(None)

    @lru_cache(maxsize=None)
    def continuation(history: tuple[int, ...], p1_rank: int, p2_rank: int) -> float:
        if history[-1] == CALL:
            winner = resolve_call_winner(
                SPEC,
                history,
                (p1_rank,),
                (p2_rank,),
            )
            return 1.0 if winner == "P1" else -1.0
        legal = rules.legal_actions_from_last(history[-1])
        return sum(
            continuation(history + (action,), p1_rank, p2_rank)
            for action in legal
        ) / len(legal)

    result = np.zeros((SPEC.ranks, len(actions)), dtype=np.float64)
    for p1_rank in range(1, SPEC.ranks + 1):
        # Given a particular P1 rank, the remaining physical cards consist of
        # two copies of each rank, except for one copy of P1's rank.
        remaining = {
            rank: SPEC.suits - int(rank == p1_rank)
            for rank in range(1, SPEC.ranks + 1)
        }
        for p2_rank, multiplicity in remaining.items():
            chance = multiplicity / sum(remaining.values())
            for col, action in enumerate(actions):
                result[p1_rank - 1, col] += chance * continuation(
                    (action,), p1_rank, p2_rank
                )
    return result


def check_exact_oracle(exact: np.ndarray) -> float:
    """The dense CFR+ first update must agree up to hand-chance scaling."""
    solver = CFRPlusDense(SPEC)
    solver.iterate()
    expected = np.maximum(exact - exact.mean(axis=1, keepdims=True), 0.0)
    # Dense CFR+ sums over the five physically distinct remaining cards.
    actual = solver.R0[0, :, 1:]
    error = float(np.abs(actual - 5.0 * expected).max())
    if error > 1e-10:
        raise AssertionError(f"Exact oracle disagrees with dense CFR+: {error}")
    return error


def make_frozen_traverser(cap: int | None, seed: int, *, streaming: bool):
    trainer = DeepCFRPlusTrainer(
        SPEC,
        regret_hidden_sizes=(8,),
        strategy_hidden_sizes=(8,),
        device="cpu",
        seed=seed,
        regret_buffer_capacity=100_000,
        strategy_buffer_capacity=100_000,
        regret_train_steps=0,
        strategy_train_steps=0,
        traversal_backend="gpu_native",
        traversal_batch_size=256,
        traverser_action_sample_count=cap,
        traverser_action_sample_mode="random",
        traversal_streaming=streaming,
        traverser_action_chunk_size=256,
        validation_fraction=0.0,
        device_replay=True,
        fused_optimizer=False,
    )
    trainer.iteration = 1
    traverser = GPUDeepCFRPlusTraverser(trainer)
    trainer._gpu_traverser = traverser
    traverser.root_choice = None

    def fixed_strategy(self, actor, features, legal_mask):
        values = torch.zeros_like(legal_mask, dtype=torch.float32)
        strategy = legal_mask.float()
        strategy /= strategy.sum(dim=1, keepdim=True)
        if actor == 0 and self.root_choice is not None:
            is_root = features[:, SPEC.ranks :].sum(dim=1) == 0
            strategy[is_root] = 0.0
            strategy[is_root, self.root_choice + 1] = 1.0
        return values, strategy

    traverser._regrets_and_strategy = types.MethodType(fixed_strategy, traverser)
    original_sample_deals = traverser._sample_deals

    def capture_deals(self, batch_size):
        p1_counts, p2_counts, totals = original_sample_deals(batch_size)
        self.root_hands = p1_counts.argmax(dim=1).cpu().numpy() + 1
        return p1_counts, p2_counts, totals

    traverser._sample_deals = types.MethodType(capture_deals, traverser)
    return trainer, traverser


def estimate_root_action_values(
    traverser,
    *,
    samples: int,
    batch_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    actions = rules_for_spec(SPEC).legal_actions_from_last(None)
    estimated = np.empty((samples, len(actions)), dtype=np.float64)
    hands = np.empty(samples, dtype=np.int64)
    totals = {"full_edges": 0, "sampled_edges": 0, "root_batches": 0}

    with torch.inference_mode():
        for start in range(0, samples, batch_size):
            n = min(batch_size, samples - start)
            batch_seed = seed + start // batch_size
            reference_hands = None
            for col, action in enumerate(actions):
                traverser.root_choice = action
                torch.manual_seed(batch_seed)
                stats = traverser.run_traversals(
                    0, n, profile=True, commit_records=False
                )
                estimated[start : start + n, col] = stats["root_values"].numpy()
                if reference_hands is None:
                    reference_hands = traverser.root_hands.copy()
                    totals["full_edges"] += stats["full_claim_edges"]
                    totals["sampled_edges"] += stats["sampled_claim_edges"]
                    totals["root_batches"] += 1
                elif not np.array_equal(reference_hands, traverser.root_hands):
                    raise AssertionError("Random-stream replay changed the root deals")
            hands[start : start + n] = reference_hands
    return estimated, hands, totals


def check_production_first_iteration_target(traverser, seed: int) -> float:
    """Confirm the reconstructed vector gives the production root target."""
    batch_size = 32
    values, _, _ = estimate_root_action_values(
        traverser, samples=batch_size, batch_size=batch_size, seed=seed
    )
    traverser.root_choice = None  # uniform root strategy
    traverser.trainer.regret_buffers[0].clear()
    torch.manual_seed(seed)
    with torch.inference_mode():
        traverser.run_traversals(0, batch_size, commit_records=True)
    buffer = traverser.trainer.regret_buffers[0]
    features = buffer.features[: buffer.size].cpu().numpy()
    targets = buffer.targets[: buffer.size].cpu().numpy()
    is_root = (features[:, SPEC.ranks :] == 0).all(axis=1)
    root_targets = targets[is_root, 1:]
    if root_targets.shape != values.shape:
        raise AssertionError(
            f"Expected {values.shape} root targets, got {root_targets.shape}"
        )
    expected = np.maximum(values - values.mean(axis=1, keepdims=True), 0.0)
    return float(np.abs(root_targets - expected).max())


def summarize(
    estimated: np.ndarray,
    hands: np.ndarray,
    exact: np.ndarray,
    totals: dict[str, float],
) -> dict:
    actions = rules_for_spec(SPEC).legal_actions_from_last(None)
    rows = []
    for p1_rank in range(1, SPEC.ranks + 1):
        subset = estimated[hands == p1_rank]
        expected = exact[p1_rank - 1]
        true_regret = expected - expected.mean()
        sample_regret = subset - subset.mean(axis=1, keepdims=True)
        mean_values = subset.mean(axis=0)
        se_values = subset.std(axis=0, ddof=1) / np.sqrt(len(subset))
        for col, action in enumerate(actions):
            rows.append(
                {
                    "hand_rank": p1_rank,
                    "action": rules_for_spec(SPEC).render_action(action),
                    "n": len(subset),
                    "exact_value": float(expected[col]),
                    "sample_mean_value": float(mean_values[col]),
                    "sample_se": float(se_values[col]),
                    "value_z": float(
                        (mean_values[col] - expected[col]) / se_values[col]
                        if se_values[col] > 0.0 else 0.0
                    ),
                    "exact_positive_regret": float(max(true_regret[col], 0.0)),
                    "mean_clipped_sample_regret": float(
                        np.maximum(sample_regret[:, col], 0.0).mean()
                    ),
                    "mean_sample_regret": float(sample_regret[:, col].mean()),
                }
            )

    exact_per_row = exact[hands - 1]
    true_regret_per_row = exact_per_row - exact_per_row.mean(axis=1, keepdims=True)
    sampled_regret = estimated - estimated.mean(axis=1, keepdims=True)
    return {
        "samples": int(len(hands)),
        "sampled_edge_fraction": (
            totals["sampled_edges"] / totals["full_edges"]
        ),
        "max_abs_value_error": max(
            abs(row["sample_mean_value"] - row["exact_value"]) for row in rows
        ),
        "max_abs_value_z": max(abs(row["value_z"]) for row in rows),
        "mean_exact_positive_regret": float(
            np.maximum(true_regret_per_row, 0.0).mean()
        ),
        "mean_clipped_sample_regret": float(
            np.maximum(sampled_regret, 0.0).mean()
        ),
        "mean_clipping_gap": float(
            np.maximum(sampled_regret, 0.0).mean()
            - np.maximum(true_regret_per_row, 0.0).mean()
        ),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--caps", type=str, default="full,2")
    parser.add_argument("--paths", type=str, default="old,streamed")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.samples <= 0 or args.batch_size <= 0:
        parser.error("samples and batch-size must be positive")
    torch.set_num_threads(1)

    exact = exact_root_action_values()
    oracle_error = check_exact_oracle(exact)
    results = {
        "spec": SPEC.to_json(),
        "seed": args.seed,
        "dense_oracle_max_error": oracle_error,
        "cases": {},
    }
    for path in args.paths.split(","):
        path = path.strip()
        if path not in {"old", "streamed"}:
            parser.error(f"Unknown path: {path}")
        for label in args.caps.split(","):
            label = label.strip()
            cap = None if label == "full" else int(label)
            trainer, traverser = make_frozen_traverser(
                cap, args.seed, streaming=(path == "streamed")
            )
            values, hands, totals = estimate_root_action_values(
                traverser,
                samples=args.samples,
                batch_size=args.batch_size,
                seed=args.seed,
            )
            case = summarize(values, hands, exact, totals)
            case["first_iteration_target_max_error"] = (
                check_production_first_iteration_target(traverser, args.seed)
            )
            key = f"{path}_{label}"
            results["cases"][key] = case
            print(
                f"{key:>14}: n={case['samples']} "
                f"edges={case['sampled_edge_fraction']:.3f} "
                f"max |value error|={case['max_abs_value_error']:.4f} "
                f"max |z|={case['max_abs_value_z']:.2f} "
                f"mean clipping gap={case['mean_clipping_gap']:.4f} "
                f"target error={case['first_iteration_target_max_error']:.2g}",
                flush=True,
            )
            del trainer
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print("Wrote", args.output)


if __name__ == "__main__":
    main()
