#!/usr/bin/env python3
"""Train tiny production neural CFR+ runs beside an exact shadow ledger.

The shadow follows the neural trainer's alternating frozen policies. It never
controls training. This separates regret-policy drift from average-policy fit.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from functools import lru_cache
import json
from pathlib import Path
import sys
import time
import types

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from liars_poker.algo.cfr_plus_dense import CFRPlusDense
from liars_poker.algo.deep_cfr_plus import DeepCFRPlusTrainer, DeviceRecentBuffer
from liars_poker.env import resolve_call_winner, rules_for_spec
from liars_poker.infoset import CALL
from liars_poker.policies.neural import compile_neural_to_dense
from liars_poker.policies.tabular_dense import DenseTabularPolicy
from scripts.compare_sampled_cfr_plus_tabular_cpu import exact_exploitability
from scripts.diagnose_neural_cfr_plus_cpu import SPEC, exact_root_action_values


def verify_aggregation_math() -> None:
    """Check weighted grouping, clipping order, and legal-action masking."""
    buffer = DeviceRecentBuffer(3, 2, 3, "cpu")
    buffer.add_many(
        torch.tensor([[1., 0.], [1., 0.], [0., 1.]]),
        torch.tensor([[1., -2., 0.], [-1., 2., 0.], [2., -1., 5.]]),
        torch.tensor([[1, 1, 1], [1, 1, 1], [1, 1, 0]], dtype=torch.bool),
        torch.tensor([1., 3., 2.]),
    )
    DeepCFRPlusTrainer._aggregate_regret_targets(buffer)
    expected = torch.tensor([[0., 1., 0.], [0., 1., 0.], [2., 0., 0.]])
    if not torch.allclose(buffer.targets[:3], expected):
        raise AssertionError("Weighted aggregate-then-clip target is incorrect")


def exact_root_values_under_policy(policy: DenseTabularPolicy) -> np.ndarray:
    """Root Q(a) for P1 under the compiled frozen policy, by private rank."""
    rules = rules_for_spec(SPEC)
    actions = rules.legal_actions_from_last(None)

    @lru_cache(maxsize=None)
    def continuation(history: tuple[int, ...], p1: int, p2: int) -> float:
        if history[-1] == CALL:
            winner = resolve_call_winner(SPEC, history, (p1,), (p2,))
            return 1.0 if winner == "P1" else -1.0
        actor = len(history) & 1
        hand_idx = policy.hand_to_idx[((p1 if actor == 0 else p2),)]
        hid = sum(1 << claim for claim in history)
        legal = rules.legal_actions_from_last(history[-1])
        return sum(
            float(policy.S[hid, hand_idx, 0 if action == CALL else action + 1])
            * continuation(history + (action,), p1, p2)
            for action in legal
        )

    out = np.zeros((SPEC.ranks, len(actions)), dtype=np.float64)
    for p1 in range(1, SPEC.ranks + 1):
        for p2 in range(1, SPEC.ranks + 1):
            remaining = SPEC.suits - int(p1 == p2)
            chance = remaining / (SPEC.ranks * SPEC.suits - 1)
            for j, action in enumerate(actions):
                out[p1 - 1, j] += chance * continuation((action,), p1, p2)
    return out


def audit_root_targets(trainer: DeepCFRPlusTrainer, frozen: DenseTabularPolicy) -> tuple[dict, np.ndarray]:
    """Compare actual P1 root records with the exact one-step target."""
    values = exact_root_values_under_policy(frozen)
    sigma = frozen.S[0, :, 1:]
    instant = values - (sigma * values).sum(axis=1, keepdims=True)
    root_features = trainer.encoder.encode_hands(frozen.hands, ())
    old = np.maximum(trainer._regret_values_from_features(0, root_features), 0.0)
    t = trainer.iteration
    exact_target = np.maximum((t - 1) / t * old[:, 1:] + instant / t, 0.0)

    buffer = trainer.regret_buffers[0]
    x = buffer.features[: buffer.size].cpu().numpy()
    targets = buffer.targets[: buffer.size].cpu().numpy()
    weights = buffer.weights[: buffer.size].cpu().numpy()
    is_root = (x[:, SPEC.ranks :] == 0).all(axis=1)
    ranks = x[is_root, : SPEC.ranks].argmax(axis=1)
    observed = targets[is_root, 1:]
    observed_weights = weights[is_root]
    errors = []
    signed = []
    for rank_idx in range(SPEC.ranks):
        have_rank = ranks == rank_idx
        if not have_rank.any():
            continue
        mean_target = np.average(
            observed[have_rank], axis=0, weights=observed_weights[have_rank]
        )
        delta = mean_target - exact_target[rank_idx]
        errors.extend(abs(delta))
        signed.extend(delta)
    result = {
        "root_target_n": int(is_root.sum()),
        "root_target_mean_abs_error": float(np.mean(errors)),
        "root_target_mean_signed_error": float(np.mean(signed)),
    }
    return result, np.concatenate([np.zeros((SPEC.ranks, 1)), exact_target], axis=1)


def verify_root_oracle_with_dense_update(policy: DenseTabularPolicy) -> None:
    """Cross-check a nonuniform frozen policy against exact dense values."""
    reference = CFRPlusDense(SPEC)
    reference.S[:] = policy.S
    reference._recompute_likelihoods()
    dense_root_value = reference._update_player(0, weight=0.0)[0] / 5.0
    root_actions = exact_root_values_under_policy(policy)
    enumerated = (policy.S[0, :, 1:] * root_actions).sum(axis=1)
    if not np.allclose(dense_root_value, enumerated, atol=1e-10):
        raise AssertionError("Root oracle disagrees with dense CFR+ on a frozen policy")


def root_shadow_policy_tv(current, shadow: CFRPlusDense) -> float:
    """Mean total-variation gap for P1's root policies over private hands."""
    regrets = np.maximum(shadow.R0[0], 0.0) * shadow.legal_mask[0]
    totals = regrets.sum(axis=1, keepdims=True)
    reference = np.tile(shadow.uniform_rows[0], (shadow.n_hands, 1))
    np.divide(regrets, totals, out=reference, where=totals > 0)
    return float(0.5 * np.abs(current.S[0] - reference).sum(axis=1).mean())


def run_one(*, cap: int | None, seed: int, clip_mode: str, args, all_rows: list[dict]) -> None:
    trainer = DeepCFRPlusTrainer(
        SPEC,
        device="cpu",
        seed=seed,
        regret_hidden_sizes=(args.width, args.width),
        strategy_hidden_sizes=(args.width, args.width),
        regret_buffer_capacity=100_000,
        strategy_buffer_capacity=100_000,
        learning_rate=args.learning_rate,
        batch_size=128,
        regret_train_steps=args.regret_steps,
        strategy_train_steps=args.strategy_steps,
        regret_positive_weight=0.5,
        regret_target_mode=clip_mode,
        strategy_weighting="linear",
        traversal_backend="gpu_native",
        traversal_batch_size=min(args.traversals, 128),
        traverser_action_sample_count=cap,
        traverser_action_sample_mode="random",
        traverser_action_baseline="none",
        traversal_streaming=False,
        validation_fraction=0.0,
        device_replay=True,
        fused_optimizer=False,
    )
    shadow = CFRPlusDense(SPEC)
    original_train_regret = trainer._train_regret
    shadow_updates = []
    root_audits = []

    if not np.allclose(
        exact_root_values_under_policy(DenseTabularPolicy(SPEC)),
        exact_root_action_values(),
        atol=1e-12,
    ):
        raise AssertionError("Frozen-policy root oracle disagrees with uniform oracle")

    def update_shadow_then_fit(self, pid: int) -> float:
        # run_iteration traverses this player before invoking _train_regret.
        # No fitting overlaps that traversal, so these are its frozen policies.
        if clip_mode == "aggregate_then_clip":
            # Audit the actual post-aggregation target rather than the raw
            # un-clipped records temporarily held in the regret buffer.
            self._aggregate_regret_targets(self.regret_buffers[pid])
        # Compiling a snapshot creates temporary MLPs before loading weights.
        # Keep that observer-only initialization from changing later samples.
        with (nullcontext() if args.observer_consumes_rng else torch.random.fork_rng(devices=[])):
            frozen = self.current_policy_dense()
        exact_target = None
        if pid == 0:
            if self.iteration == 1:
                verify_root_oracle_with_dense_update(frozen)
            audit, exact_target = audit_root_targets(self, frozen)
        shadow.S[:] = frozen.S
        shadow._recompute_likelihoods()
        old = (shadow.R0 if pid == 0 else shadow.R1).copy()
        shadow._update_player(pid, weight=float(self.iteration))
        new = shadow.R0 if pid == 0 else shadow.R1
        shadow_updates.append(float(np.abs(new - old).sum()))
        loss = original_train_regret(pid)
        if pid == 0:
            features = self.encoder.encode_hands(frozen.hands, ())
            fitted = np.maximum(self._regret_values_from_features(0, features), 0.0)
            audit["root_fitted_to_exact_target_mae"] = float(
                np.abs(fitted[:, 1:] - exact_target[:, 1:]).mean()
            )
            root_audits.append(audit)
        return loss

    trainer._train_regret = types.MethodType(update_shadow_then_fit, trainer)
    for iteration in range(1, args.iterations + 1):
        start = time.perf_counter()
        record = trainer.run_iteration(traversals_per_player=args.traversals)
        elapsed = time.perf_counter() - start
        if iteration != 1 and iteration % args.eval_every:
            continue
        with (nullcontext() if args.observer_consumes_rng else torch.random.fork_rng(devices=[])):
            current = trainer.current_policy_dense()
            learned_average = compile_neural_to_dense(trainer.average_policy())
        exact_average = shadow.average_policy()
        row = {
            "clip_mode": clip_mode,
            "cap": "full" if cap is None else cap,
            "seed": seed,
            "iteration": iteration,
            "current_exploitability": exact_exploitability(current),
            "learned_average_exploitability": exact_exploitability(learned_average),
            "shadow_average_exploitability": exact_exploitability(exact_average),
            "root_shadow_policy_tv": root_shadow_policy_tv(current, shadow),
            "regret_loss": float(np.mean(record["regret_loss"])),
            "strategy_loss": float(np.mean(record["strategy_loss"])),
            "sampled_edge_fraction": record["action_sampling"]["claim_edge_fraction"],
            "regret_records": record["new_regret_records"],
            "shadow_update_l1": shadow_updates[-2:],
            **root_audits[-1],
            "last_iteration_wall_s": elapsed,
        }
        all_rows.append(row)
        print(
            f"mode={clip_mode} cap={row['cap']} seed={seed} iter={iteration} "
            f"current={row['current_exploitability']:.4f} "
            f"learned_avg={row['learned_average_exploitability']:.4f} "
            f"exact_avg={row['shadow_average_exploitability']:.4f} "
            f"root_TV={row['root_shadow_policy_tv']:.3f}",
            flush=True,
        )
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(all_rows, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--traversals", type=int, default=32)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--regret-steps", type=int, default=8)
    parser.add_argument("--strategy-steps", type=int, default=4)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--caps", default="full,2")
    parser.add_argument("--clip-modes", default="clip_each_record")
    parser.add_argument(
        "--observer-consumes-rng", action="store_true",
        help="Reproduce the earlier shadow experiment's snapshot RNG behavior.",
    )
    parser.add_argument("--seeds", default="17,23")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.iterations, args.traversals, args.eval_every, args.width) <= 0:
        parser.error("iterations, traversals, eval-every and width must be positive")
    if min(args.regret_steps, args.strategy_steps) < 0:
        parser.error("training steps must be nonnegative")
    torch.set_num_threads(1)
    verify_aggregation_math()
    all_rows = []
    for clip_mode in args.clip_modes.split(","):
        clip_mode = clip_mode.strip()
        if clip_mode not in {"clip_each_record", "aggregate_then_clip"}:
            parser.error(f"Unknown clip mode: {clip_mode}")
        for cap_label in args.caps.split(","):
            cap_label = cap_label.strip()
            cap = None if cap_label == "full" else int(cap_label)
            for seed_label in args.seeds.split(","):
                # Save each completed monitor, including earlier configurations.
                run_one(
                    cap=cap, seed=int(seed_label), clip_mode=clip_mode,
                    args=args, all_rows=all_rows,
                )


if __name__ == "__main__":
    main()
