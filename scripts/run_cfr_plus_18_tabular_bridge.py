#!/usr/bin/env python3
"""Resumable 18-claim tabular CFR+ reach/value bridge.

One rolling checkpoint per arm. Evaluations are numbers in JSONL, not a
growing collection of dense policy snapshots. See --help for launch options.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from liars_poker.algo.br_exact_dense_to_dense import adjustment_factor, best_response_dense
from liars_poker.algo.cfr_plus_dense import CFRPlusDense
from liars_poker.core import GameSpec, generate_deck
from liars_poker.infoset import CALL
from liars_poker.policies.tabular_dense import DenseTabularPolicy


SPEC = GameSpec(
    ranks=4, suits=4, hand_size=2,
    claim_kinds=("RankHigh", "Pair", "TwoPair", "Trips"),
    suit_symmetry=True,
)
ARMS = (
    "exact", "sample_reach", "ignore_reach", "sample_value", "sample_both",
    "conditional", "exact_reach_gated", "unit_reach_gated",
)
SAMPLED = frozenset((
    "sample_reach", "sample_value", "sample_both", "conditional",
    "exact_reach_gated", "unit_reach_gated",
))
MIN_FREE_GIB = 8.0


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as out:
        out.write(json.dumps(row, allow_nan=False) + "\n")
        out.flush()


def write_json(path: Path, row: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(row, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows


def check_disk(path: Path, extra_bytes: int = 0) -> None:
    free = shutil.disk_usage(path).free - extra_bytes
    if free < MIN_FREE_GIB * 1024**3:
        raise RuntimeError(
            f"disk guard: only {free / 1024**3:.2f} GiB would remain at {path}; "
            "keeping the previous checkpoint intact"
        )


class TabularBridge:
    def __init__(self, arm: str, seed: int, roots: int):
        if arm not in ARMS or roots <= 0:
            raise ValueError((arm, roots))
        self.arm = arm
        self.seed = seed
        self.roots = roots
        self.solver = CFRPlusDense(SPEC)
        self.rng = np.random.default_rng(seed)
        self.deck = np.asarray(generate_deck(SPEC), dtype=np.int8)
        self.total_deals = math.comb(len(self.deck), SPEC.hand_size)
        self.remaining_deals = math.comb(len(self.deck) - SPEC.hand_size, SPEC.hand_size)
        self.hand_prob = np.asarray(
            [adjustment_factor(SPEC, (), h) / self.total_deals for h in self.solver.hands],
            dtype=np.float64,
        )
        assert np.isclose(self.hand_prob.sum(), 1.0)
        assert np.allclose(self.solver.A0.sum(axis=1), self.remaining_deals)
        assert np.allclose(self.solver.A1.sum(axis=1), self.remaining_deals)
        self.row_scale = self.hand_prob / self.remaining_deals

    def _deal(self) -> tuple[int, int]:
        cards = self.deck[self.rng.permutation(len(self.deck))[: 2 * SPEC.hand_size]]
        h0 = tuple(sorted(int(x) for x in cards[: SPEC.hand_size]))
        h1 = tuple(sorted(int(x) for x in cards[SPEC.hand_size :]))
        return self.solver.hand_to_idx[h0], self.solver.hand_to_idx[h1]

    def _traverse(
        self, hid: int, h0: int, h1: int, traverser: int,
        counts: np.ndarray, sums: np.ndarray, touched: list[tuple[int, int]],
    ) -> float:
        s = self.solver
        actor = int(s.popcount[hid] & 1)
        actions = s.legal_actions[hid]
        hands = (h0, h1)

        def child_value(action: int) -> float:
            if action == CALL:
                truthful = bool(s._truth_mats[int(s.last_claim[hid])][h0, h1])
                won = (traverser != actor) if truthful else (traverser == actor)
                return 1.0 if won else -1.0
            return self._traverse(hid | (1 << action), h0, h1, traverser, counts, sums, touched)

        if actor != traverser:
            pick = self.rng.random()
            for action, col in zip(actions, s.legal_cols[hid]):
                pick -= float(s.S[hid, hands[actor], col])
                if pick <= 0.0:
                    return child_value(action)
            return child_value(actions[-1])

        hand = hands[traverser]
        cols = s.legal_cols[hid]
        values = np.asarray([child_value(action) for action in actions], dtype=np.float64)
        probs = s.S[hid, hand, list(cols)]
        node_value = float(np.dot(values, probs))
        if counts[hid, hand] == 0:
            touched.append((hid, hand))
        counts[hid, hand] += 1
        sums[hid, hand, list(cols)] += values - node_value
        return node_value

    def _sample(self, player: int) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
        s = self.solver
        counts = np.zeros((s.H, s.n_hands), dtype=np.uint32)
        sums = np.zeros_like(s.S)
        touched: list[tuple[int, int]] = []
        for _ in range(self.roots):
            h0, h1 = self._deal()
            self._traverse(0, h0, h1, player, counts, sums, touched)
        return counts, sums, touched

    def _exact_update(
        self, player: int, counts: np.ndarray | None, sums: np.ndarray | None,
    ) -> None:
        s = self.solver
        A = s.A0 if player == 0 else s.A1

        def transform(hid: int, opp_reach: np.ndarray, raw: np.ndarray) -> np.ndarray:
            # Dense CFR+ sums unnormalised opponent-hand multiplicities. Our
            # sampled roots include the chance of the player's own hand too.
            q_dense = A @ opp_reach
            q = self.row_scale * q_dense
            if self.arm == "sample_value":
                self._q_for_stats[hid] = q
            if self.arm == "exact":
                return raw * self.row_scale[None, :]
            if self.arm == "exact_reach_gated":
                assert counts is not None
                return raw * self.row_scale[None, :] * (counts[hid][None, :] > 0)
            if self.arm == "sample_value":
                assert counts is not None and sums is not None
                g_sample = np.divide(
                    sums[hid, :, s.legal_cols[hid]],
                    counts[hid][None, :],
                    out=np.zeros_like(raw), where=counts[hid][None, :] > 0,
                )
                return q[None, :] * g_sample
            g = np.divide(raw, q_dense[None, :], out=np.zeros_like(raw), where=q_dense[None, :] > 0)
            if self.arm == "ignore_reach":
                return g
            if self.arm == "unit_reach_gated":
                assert counts is not None
                return g * (counts[hid][None, :] > 0)
            assert counts is not None
            return g * (counts[hid][None, :] / self.roots)

        s._update_player(player, weight=float(s.iteration), increment_transform=transform)

    def _sampled_update(
        self, player: int, counts: np.ndarray, sums: np.ndarray,
        touched: list[tuple[int, int]],
    ) -> None:
        s = self.solver
        R = s.R0 if player == 0 else s.R1
        SS = s.SS0 if player == 0 else s.SS1
        Lp = s.L0 if player == 0 else s.L1
        # The same exact own-reach/linear-weight average as CFRPlusDense.
        for pc in range(player, s.k + 1, 2):
            for hid in s.hids_by_popcount[pc]:
                if not s.legal_actions[hid]:
                    continue
                cols = s.legal_cols[hid]
                SS[hid, :, cols] += float(s.iteration) * (Lp[hid][None, :] * s.S[hid, :, cols])
        for hid, hand in touched:
            cols = s.legal_cols[hid]
            increment = sums[hid, hand, list(cols)]
            if self.arm == "sample_both":
                increment = increment / self.roots
            else:
                increment = increment / counts[hid, hand]
            R[hid, hand, list(cols)] = np.maximum(R[hid, hand, list(cols)] + increment, 0.0)

    def iterate(self) -> dict:
        s = self.solver
        s.iteration += 1
        sampling_s = exact_s = 0.0
        total_visits = unique_rows = 0
        if self.arm == "sample_value":
            self._q_for_stats = np.zeros((s.H, s.n_hands), dtype=np.float64)
            self._count_for_stats = np.zeros((s.H, s.n_hands), dtype=np.uint32)
        for player in (0, 1):
            s._update_strategy_for_player(player)
            s._recompute_likelihoods()
            counts = sums = touched = None
            if self.arm in SAMPLED:
                start = time.perf_counter()
                counts, sums, touched = self._sample(player)
                sampling_s += time.perf_counter() - start
                total_visits += int(counts.sum())
                unique_rows += len(touched)
                if self.arm == "sample_value":
                    self._count_for_stats += counts
            start = time.perf_counter()
            if self.arm in ("sample_both", "conditional"):
                assert counts is not None and sums is not None and touched is not None
                self._sampled_update(player, counts, sums, touched)
            else:
                self._exact_update(player, counts, sums)
            exact_s += time.perf_counter() - start
        result = {"sampling_s": sampling_s, "update_s": exact_s,
                  "sample_visits": total_visits, "sample_unique_rows": unique_rows}
        if self.arm == "sample_value":
            expected_hits = self.roots * self._q_for_stats
            bins = []
            for low, high in ((0.0, 0.1), (0.1, 1.0), (1.0, 10.0), (10.0, math.inf)):
                in_bin = (self._q_for_stats > 0) & (expected_hits >= low) & (expected_hits < high)
                bins.append((int(in_bin.sum()), int((in_bin & (self._count_for_stats == 0)).sum())))
            result["reach_no_visit_bins"] = {
                label: {"infosets": total, "unvisited": missed}
                for label, (total, missed) in zip(("<0.1", "0.1-1", "1-10", ">=10"), bins)
            }
        return result

    def policy(self, kind: str) -> DenseTabularPolicy:
        s = self.solver
        if kind == "average":
            return s.average_policy()
        if kind != "current":
            raise ValueError(kind)
        # Evaluate post-update regret matching without changing S. The dense
        # alternating solver intentionally retains the other player's last
        # refreshed strategy between player updates.
        policy = DenseTabularPolicy(SPEC)
        for hid in range(s.H):
            if not s.legal_actions[hid]:
                continue
            player = int(s.popcount[hid] & 1)
            regrets = s.R0[hid] if player == 0 else s.R1[hid]
            positive = np.maximum(regrets, 0.0) * s.legal_mask[hid]
            totals = positive.sum(axis=1)
            good = totals > 0
            if np.any(good):
                policy.S[hid, good] = (positive[good] / totals[good, None]).astype(np.float32)
        policy.recompute_likelihoods()
        return policy

    def evaluate(self, kind: str) -> dict:
        start = time.perf_counter()
        policy = self.policy(kind)
        _, meta = best_response_dense(SPEC, policy, store_state_values=False)
        p0, p1 = meta["computer"].exploitability()
        return {"kind": kind, "p_first": float(p0), "p_second": float(p1),
                "exploitability": float(p0 + p1 - 1.0),
                "evaluation_s": time.perf_counter() - start}

    def checkpoint(self, path: Path, measured_s: float, next_eval_s: float, next_checkpoint_s: float) -> None:
        import fcntl

        s = self.solver
        lock = path.parent.parent / ".checkpoint.lock"
        lock.touch(exist_ok=True)
        with lock.open("rb") as lock_handle:
            fcntl.flock(lock_handle, fcntl.LOCK_EX)
            raw_bytes = sum(a.nbytes for a in (s.R0, s.R1, s.SS0, s.SS1))
            check_disk(path.parent, extra_bytes=raw_bytes + 1024**3)
            tmp = path.with_name(path.name + ".tmp")
            meta = {
                "arm": self.arm, "seed": self.seed, "roots": self.roots,
                "iteration": s.iteration, "measured_s": measured_s,
                "next_eval_s": next_eval_s, "next_checkpoint_s": next_checkpoint_s,
                "rng_state": self.rng.bit_generator.state,
            }
            try:
                with tmp.open("wb") as out:
                    np.savez_compressed(out, R0=s.R0, R1=s.R1, SS0=s.SS0, SS1=s.SS1,
                                        meta=np.bytes_(json.dumps(meta)))
                os.replace(tmp, path)
            finally:
                tmp.unlink(missing_ok=True)

    def restore(self, path: Path) -> dict:
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(data["meta"].item().decode("utf-8"))
            if (meta["arm"], meta["seed"], meta["roots"]) != (self.arm, self.seed, self.roots):
                raise ValueError("checkpoint arm/seed/roots mismatch")
            s = self.solver
            for name in ("R0", "R1", "SS0", "SS1"):
                arr = data[name]
                if arr.shape != getattr(s, name).shape or arr.dtype != s.dtype:
                    raise ValueError(f"checkpoint {name} shape/dtype mismatch")
                setattr(s, name, arr)
        self.solver.iteration = int(meta["iteration"])
        self.rng.bit_generator.state = meta["rng_state"]
        return meta


def trim_to_checkpoint(arm_dir: Path, iteration: int) -> None:
    """Keep logs from a lost tail, but exclude them from the active curves."""
    for name in ("training.jsonl", "evaluations.jsonl"):
        path = arm_dir / name
        rows = read_jsonl(path)
        valid = [r for r in rows if int(r.get("iteration", 0)) <= iteration]
        stale = [r for r in rows if int(r.get("iteration", 0)) > iteration]
        if stale:
            with (arm_dir / f"{name}.superseded").open("a", encoding="utf-8") as out:
                for row in stale:
                    out.write(json.dumps(row) + "\n")
            with path.open("w", encoding="utf-8") as out:
                for row in valid:
                    out.write(json.dumps(row) + "\n")


def run_arm(args: argparse.Namespace) -> None:
    arm_dir = args.output_root / args.arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = arm_dir / "latest_checkpoint.npz"
    manifest_path = arm_dir / "manifest.json"
    target_s = 60.0 * args.minutes
    trainer = TabularBridge(args.arm, args.seed, args.roots)
    measured_s = 0.0
    next_eval_s = args.eval_minutes * 60.0
    next_checkpoint_s = args.checkpoint_minutes * 60.0
    if checkpoint_path.exists():
        if not args.resume:
            raise RuntimeError(f"checkpoint exists at {checkpoint_path}; use --resume")
        prior_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        prior_target = float(prior_manifest["target_minutes"])
        if args.minutes < prior_target:
            raise ValueError("Cannot shorten the target of a resumed bridge arm")
        meta = trainer.restore(checkpoint_path)
        if args.minutes > prior_target:
            prior_manifest.setdefault("extensions", []).append({
                "utc": utc(), "from_minutes": prior_target,
                "to_minutes": args.minutes,
            })
            prior_manifest["target_minutes"] = args.minutes
            write_json(manifest_path, prior_manifest)
        measured_s = float(meta["measured_s"])
        next_eval_s = float(meta["next_eval_s"])
        next_checkpoint_s = float(meta["next_checkpoint_s"])
        trim_to_checkpoint(arm_dir, trainer.solver.iteration)
        append_jsonl(arm_dir / "events.jsonl", {"event": "resume", "utc": utc(),
                      "iteration": trainer.solver.iteration, "measured_training_min": measured_s / 60})
    else:
        if args.resume and manifest_path.exists():
            raise RuntimeError(f"cannot resume {arm_dir}: no valid checkpoint")
        write_json(manifest_path, {
            "arm": args.arm, "seed": args.seed, "roots": args.roots,
            "target_minutes": args.minutes, "eval_minutes": args.eval_minutes,
            "checkpoint_minutes": args.checkpoint_minutes, "spec": SPEC.to_json(),
            "created_utc": utc(),
        })
        trainer.checkpoint(checkpoint_path, 0.0, next_eval_s, next_checkpoint_s)
    state_path = arm_dir / "state.json"
    status = "running"
    try:
        while measured_s < target_s:
            start = time.perf_counter()
            stats = trainer.iterate()
            iteration_s = time.perf_counter() - start
            measured_s += iteration_s
            row = {"utc": utc(), "arm": args.arm, "iteration": trainer.solver.iteration,
                   "measured_training_min": measured_s / 60.0,
                   "iteration_s": iteration_s, **stats}
            append_jsonl(arm_dir / "training.jsonl", row)
            print(f"{args.arm} train={measured_s/60:.1f}m iter={trainer.solver.iteration} "
                  f"iter_s={iteration_s:.2f} sample={stats['sampling_s']:.2f} "
                  f"update={stats['update_s']:.2f}", flush=True)

            if measured_s >= next_eval_s:
                for kind in ("average", "current"):
                    result = trainer.evaluate(kind)
                    append_jsonl(arm_dir / "evaluations.jsonl", {
                        "utc": utc(), "arm": args.arm, "iteration": trainer.solver.iteration,
                        "measured_training_min": measured_s / 60.0, **result,
                    })
                    print(f"{args.arm} {kind} exact={result['exploitability']:.6f}", flush=True)
                while measured_s >= next_eval_s:
                    next_eval_s += args.eval_minutes * 60.0

            if measured_s >= next_checkpoint_s or measured_s >= target_s:
                while measured_s >= next_checkpoint_s:
                    next_checkpoint_s += args.checkpoint_minutes * 60.0
                trainer.checkpoint(checkpoint_path, measured_s, next_eval_s, next_checkpoint_s)
                print(f"{args.arm} checkpoint={checkpoint_path.stat().st_size/1024**3:.2f} GiB", flush=True)
            write_json(state_path, {"arm": args.arm, "status": status,
                                    "iteration": trainer.solver.iteration,
                                    "measured_training_min": measured_s / 60,
                                    "updated_utc": utc()})
        status = "complete"
    except BaseException as exc:
        status = "interrupted"
        append_jsonl(arm_dir / "events.jsonl", {"event": "error", "utc": utc(),
                     "iteration": trainer.solver.iteration, "message": str(exc)})
        raise
    finally:
        write_json(state_path, {"arm": args.arm, "status": status,
                                "iteration": trainer.solver.iteration,
                                "measured_training_min": measured_s / 60,
                                "updated_utc": utc()})


def audit_roots(seed: int, draws: int) -> None:
    """Verify unconditional hand and shallow counterfactual reach units."""
    t = TabularBridge("exact", seed, 1)
    s = t.solver
    hand_counts = np.zeros(s.n_hands, dtype=np.int64)
    first_claim_counts = np.zeros(s.n_hands, dtype=np.int64)
    root_col = s.legal_cols[0][0]
    root_action = s.legal_actions[0][0]
    if root_action == CALL:
        raise AssertionError("first root action must be a claim")
    for _ in range(draws):
        h0, h1 = t._deal()
        hand_counts[h0] += 1
        if t.rng.random() < s.S[0, h0, root_col]:
            first_claim_counts[h1] += 1
    p0 = t.hand_prob
    # For player 1 at the first opponent claim, own hand is h1.
    q1 = t.row_scale * (s.A1 @ s.S[0, :, root_col])
    for label, seen, expected in (("root", hand_counts, p0),
                                  ("first_opponent_claim", first_claim_counts, q1)):
        empirical = seen / draws
        sigma = np.sqrt(np.maximum(expected * (1 - expected) / draws, 1e-15))
        max_z = float(np.max(np.abs(empirical - expected) / sigma))
        print(f"audit {label}: max_z={max_z:.3f} sum_expected={expected.sum():.6f} "
              f"sum_observed={empirical.sum():.6f}", flush=True)
        if max_z > 6.0:
            raise AssertionError(f"{label} reach normalization failed: max_z={max_z:.2f}")


def supervise(args: argparse.Namespace) -> None:
    args.output_root.mkdir(parents=True, exist_ok=True)
    selected = tuple(args.arms.split(",")) if args.arms else ARMS
    if not selected or len(set(selected)) != len(selected) or any(a not in ARMS for a in selected):
        raise ValueError(f"Invalid --arms: {args.arms}")
    run_path = args.output_root / "run.json"
    previous = json.loads(run_path.read_text(encoding="utf-8")) if run_path.exists() else {}
    if previous and (int(previous["seed"]) != args.seed or int(previous["roots"]) != args.roots):
        raise ValueError("Existing bridge run has different seed or roots")
    if previous and args.minutes < float(previous["target_minutes"]):
        raise ValueError("Cannot shorten the target of an existing bridge run")
    extensions = list(previous.get("extensions", []))
    if previous and args.minutes > float(previous["target_minutes"]):
        extensions.append({"utc": utc(), "from_minutes": float(previous["target_minutes"]),
                           "to_minutes": args.minutes, "arms": list(selected)})
    all_arms = tuple(dict.fromkeys((*previous.get("arms", ()), *selected)))
    write_json(run_path, {
        "arms": all_arms, "seed": args.seed, "roots": args.roots,
        "target_minutes": args.minutes,
        "extensions": extensions,
        "created_or_resumed_utc": previous.get("created_or_resumed_utc", utc()),
    })
    env = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = str(args.threads_per_arm)
    processes = []
    for arm in selected:
        cmd = [sys.executable, "-u", str(Path(__file__).resolve()),
               "--arm", arm, "--output-root", str(args.output_root),
               "--seed", str(args.seed), "--roots", str(args.roots),
               "--minutes", str(args.minutes), "--eval-minutes", str(args.eval_minutes),
               "--checkpoint-minutes", str(args.checkpoint_minutes)]
        if args.resume:
            cmd.append("--resume")
        log_path = args.output_root / arm / "console.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as log:
            processes.append((arm, subprocess.Popen(cmd, cwd=ROOT, env=env,
                                                     stdout=log, stderr=subprocess.STDOUT)))
        print(f"started {arm} pid={processes[-1][1].pid} log={log_path}", flush=True)
    failures = []
    for arm, process in processes:
        code = process.wait()
        print(f"finished {arm} exit={code}", flush=True)
        if code:
            failures.append((arm, code))
    if failures:
        raise RuntimeError(f"arms failed: {failures}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root", type=Path, default=Path("artifacts/cfr_plus_18_tabular_bridge"))
    p.add_argument("--arm", choices=ARMS, help="run one arm")
    p.add_argument("--arms", help="comma-separated arms to supervise; default: all")
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--roots", type=int, default=1024)
    p.add_argument("--minutes", type=float, default=300.0)
    p.add_argument("--eval-minutes", type=float, default=15.0)
    p.add_argument("--checkpoint-minutes", type=float, default=30.0)
    p.add_argument("--threads-per-arm", type=int, default=1)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--audit-roots", type=int, metavar="DRAWS")
    args = p.parse_args()
    if args.audit_roots:
        audit_roots(args.seed, args.audit_roots)
        return
    if min(args.roots, args.minutes, args.eval_minutes, args.checkpoint_minutes, args.threads_per_arm) <= 0:
        p.error("roots, minutes, eval/checkpoint cadence, and threads must be positive")
    if args.arm and args.arms:
        p.error("use --arm or --arms, not both")
    if args.arm:
        run_arm(args)
    else:
        supervise(args)


if __name__ == "__main__":
    main()
