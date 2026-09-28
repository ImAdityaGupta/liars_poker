#!/usr/bin/env python3
"""Audit sampled neural CFR+ regret targets against exact values by depth."""

from __future__ import annotations

import argparse
from collections import defaultdict
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

from liars_poker.algo.cfr_plus_dense import CFRPlusDense
from liars_poker.algo.deep_cfr_plus import DeepCFRPlusTrainer
from liars_poker.env import resolve_call_winner
from liars_poker.infoset import CALL
from scripts.diagnose_neural_cfr_plus_cpu import SPEC
from scripts.shadow_neural_cfr_plus_cpu import exact_root_values_under_policy


def exact_values(frozen, pid: int):
    """Dense counterfactual action values divided by opponent reach mass."""
    exact = CFRPlusDense(SPEC)
    exact.S[:] = frozen.S
    exact._recompute_likelihoods()
    state_values = exact._update_player(pid, weight=0.0)
    opponent_reach = exact.L1 if pid == 0 else exact.L0
    blocker = exact.A0 if pid == 0 else exact.A1

    def at(hid: int, hand_idx: int) -> np.ndarray:
        if int(exact.popcount[hid] & 1) != pid:
            raise AssertionError("Information set belongs to the wrong player")
        mass = (blocker @ opponent_reach[hid])[hand_idx]
        if mass <= 0:
            raise AssertionError("Visited information set has zero opponent reach")
        out = np.zeros(exact.A, dtype=np.float64)
        for action in exact.legal_actions[hid]:
            if action == CALL:
                out[0] = exact._terminal_utility(
                    hid, opponent_reach[hid], caller=pid, player=pid
                )[hand_idx] / mass
            else:
                child = hid | (1 << action)
                out[action + 1] = state_values[child, hand_idx] / mass
        return out

    return exact, at


def verify_depth_oracle(frozen, exact: CFRPlusDense, q_at, pid: int) -> None:
    """Cross-check dense conditional values by enumerating hidden ranks."""
    @lru_cache(maxsize=None)
    def continuation(history: tuple[int, ...], p1: int, p2: int) -> float:
        if history and history[-1] == CALL:
            winner = resolve_call_winner(SPEC, history, (p1,), (p2,))
            return 1.0 if winner == ("P1" if pid == 0 else "P2") else -1.0
        hid = sum(1 << action for action in history)
        actor = len(history) & 1
        hand_idx = exact.hand_to_idx[((p1 if actor == 0 else p2),)]
        sigma = frozen.S[hid, hand_idx]
        return sum(
            float(sigma[0 if action == CALL else action + 1])
            * continuation(history + (action,), p1, p2)
            for action in exact.legal_actions[hid]
        )

    for hid in range(exact.H):
        history = tuple(action for action in range(exact.k) if hid & (1 << action))
        if (len(history) & 1) != pid:
            continue
        for own_rank in range(1, SPEC.ranks + 1):
            hand_idx = exact.hand_to_idx[(own_rank,)]
            posterior = []
            for opp_rank in range(1, SPEC.ranks + 1):
                chance = (SPEC.suits - int(own_rank == opp_rank)) / (SPEC.ranks * SPEC.suits - 1)
                likelihood = 1.0
                for i, action in enumerate(history):
                    if (i & 1) != pid:
                        prefix_hid = sum(1 << prev for prev in history[:i])
                        opp_idx = exact.hand_to_idx[(opp_rank,)]
                        likelihood *= frozen.S[prefix_hid, opp_idx, action + 1]
                posterior.append(chance * likelihood)
            mass = sum(posterior)
            if mass <= 1e-12:
                continue
            q = q_at(hid, hand_idx)
            for action in exact.legal_actions[hid]:
                col = 0 if action == CALL else action + 1
                brute = 0.0
                for opp_rank, probability in enumerate(posterior, start=1):
                    hands = (own_rank, opp_rank) if pid == 0 else (opp_rank, own_rank)
                    brute += probability * continuation(history + (action,), *hands)
                if not np.isclose(q[col], brute / mass, atol=1e-10):
                    raise AssertionError(
                        f"Depth oracle mismatch: pid={pid} hid={hid} rank={own_rank} action={action}"
                    )


def audit_before_fit(trainer, frozen, pid: int) -> list[dict]:
    exact, q_at = exact_values(frozen, pid)
    if trainer.iteration == 1:
        verify_depth_oracle(frozen, exact, q_at, pid)
    if pid == 0 and trainer.iteration == 1:
        root = exact_root_values_under_policy(frozen)
        dense_root = np.array([
            q_at(0, exact.hand_to_idx[(rank,)])[1:]
            for rank in range(1, SPEC.ranks + 1)
        ])
        if not np.allclose(root, dense_root, atol=1e-10):
            raise AssertionError("Dense conditional values disagree with root oracle")

    buffer = trainer.regret_buffers[pid]
    n = buffer.size
    features = buffer.features[:n].detach().cpu().numpy()
    observed = buffer.targets[:n].detach().cpu().numpy()
    weights = buffer.weights[:n].detach().cpu().numpy().astype(np.float64)
    histories = features[:, SPEC.ranks :]
    powers = 1 << np.arange(histories.shape[1], dtype=np.int64)
    hids = (histories.astype(np.int64) @ powers).astype(int)
    hand_ranks = features[:, : SPEC.ranks].argmax(axis=1) + 1

    indices: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, (hid, rank) in enumerate(zip(hids, hand_ranks)):
        indices[(int(hid), int(rank))].append(i)

    scale_old = (trainer.iteration - 1) / trainer.iteration
    old = trainer._regret_values_from_features(pid, features)
    audit = []
    for (hid, rank), row_indices in indices.items():
        hand_idx = exact.hand_to_idx[(rank,)]
        mask = exact.legal_mask[hid]
        if not np.array_equal(
            mask, buffer.legal_masks[row_indices[0]].detach().cpu().numpy()
        ):
            raise AssertionError("Regret record legal mask disagrees with dense rules")
        q = q_at(hid, hand_idx)
        sigma = frozen.S[hid, hand_idx]
        instant = (q - float(np.dot(sigma, q))) * mask
        expected = np.maximum(
            scale_old * np.maximum(old[row_indices[0]], 0.0)
            + instant / trainer.iteration,
            0.0,
        ) * mask
        mean_observed = np.average(
            observed[row_indices], axis=0, weights=weights[row_indices]
        )
        audit.append({
            "depth": int(exact.popcount[hid]),
            "feature": features[row_indices[0]],
            "mask": mask,
            "expected": expected,
            "target_error": (mean_observed - expected)[mask] * trainer.iteration,
            "records": len(row_indices),
        })
    return audit


def summarize_after_fit(trainer, pid: int, audit: list[dict], mode: str, seed: int) -> list[dict]:
    if not audit:
        return []
    x = np.stack([entry["feature"] for entry in audit])
    fitted = np.maximum(trainer._regret_values_from_features(pid, x), 0.0)
    by_depth: dict[int, dict[str, list]] = defaultdict(
        lambda: {"target": [], "fitted": [], "infosets": [], "records": []}
    )
    for entry, pred in zip(audit, fitted):
        group = by_depth[entry["depth"]]
        group["target"].extend(entry["target_error"])
        group["fitted"].extend(
            ((pred - entry["expected"]) * trainer.iteration)[entry["mask"]]
        )
        group["infosets"].append(1)
        group["records"].append(entry["records"])
    rows = []
    for depth, group in sorted(by_depth.items()):
        target = np.asarray(group["target"])
        fitted_error = np.asarray(group["fitted"])
        rows.append({
            "mode": mode,
            "seed": seed,
            "player": pid,
            "iteration": trainer.iteration,
            "depth": depth,
            "infosets": len(group["infosets"]),
            "records": int(sum(group["records"])),
            "target_signed_error_x_t": float(target.mean()),
            "target_abs_error_x_t": float(np.abs(target).mean()),
            "fitted_abs_error_x_t": float(np.abs(fitted_error).mean()),
        })
    return rows


def run_one(mode: str, seed: int, args, all_rows: list[dict]) -> None:
    trainer = DeepCFRPlusTrainer(
        SPEC,
        device="cpu",
        seed=seed,
        regret_hidden_sizes=(32, 32),
        strategy_hidden_sizes=(32, 32),
        regret_buffer_capacity=100_000,
        strategy_buffer_capacity=100_000,
        learning_rate=1e-3,
        batch_size=128,
        regret_train_steps=8,
        strategy_train_steps=4,
        regret_positive_weight=0.5,
        regret_target_mode=mode,
        strategy_weighting="linear",
        traversal_backend="gpu_native",
        traversal_batch_size=32,
        traverser_action_sample_count=2,
        traverser_action_sample_mode="random",
        traverser_action_baseline="none",
        traversal_streaming=False,
        validation_fraction=0.0,
        device_replay=True,
        fused_optimizer=False,
    )
    original_train = trainer._train_regret
    audit_iters = set(args.audit_iterations)

    def audit_then_fit(self, pid: int) -> float:
        if self.iteration not in audit_iters:
            return original_train(pid)
        if mode == "aggregate_then_clip":
            self._aggregate_regret_targets(self.regret_buffers[pid])
        with torch.random.fork_rng(devices=[]):
            frozen = self.current_policy_dense()
        audit = audit_before_fit(self, frozen, pid)
        loss = original_train(pid)
        rows = summarize_after_fit(self, pid, audit, mode, seed)
        all_rows.extend(rows)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(all_rows, indent=2), encoding="utf-8")
        print(
            f"mode={mode} seed={seed} iter={self.iteration} p{pid + 1} "
            f"depths={[row['depth'] for row in rows]}", flush=True
        )
        return loss

    trainer._train_regret = types.MethodType(audit_then_fit, trainer)
    for _ in range(args.iterations):
        trainer.run_iteration(traversals_per_player=32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--audit-iterations", default="1,50,150,300")
    parser.add_argument("--seeds", default="17,23")
    parser.add_argument("--output", type=Path, default=Path("docs/data/neural_depth_audit_300.json"))
    args = parser.parse_args()
    args.audit_iterations = [int(value) for value in args.audit_iterations.split(",")]
    if args.iterations <= 0 or any(it < 1 or it > args.iterations for it in args.audit_iterations):
        parser.error("Audit iterations must be within the training run")
    torch.set_num_threads(1)
    rows: list[dict] = []
    for mode in ("clip_each_record", "aggregate_then_clip"):
        for seed in [int(value) for value in args.seeds.split(",")]:
            run_one(mode, seed, args, rows)


if __name__ == "__main__":
    main()
