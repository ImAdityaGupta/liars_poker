#!/usr/bin/env python3
"""Fresh O/E/S/N longitudinal audit or exact-g-at-visited-sets continuation."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from types import MethodType

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from liars_poker.algo.br_exact_dense_to_dense import best_response_dense
from liars_poker.algo.deep_cfr_plus import DeepCFRPlusTrainer
from liars_poker.policies.neural import compile_neural_to_dense
from liars_poker.serialization import save_policy
from scripts.audit_cfr_plus_18_late_update import exact_tables
from scripts.run_cfr_plus_18_target_order_cpu_overnight import SPEC, make_trainer


def append(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, allow_nan=False) + "\n")
        f.flush()


def atomic_json(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def exact_exploitability(dense) -> dict:
    _, meta = best_response_dense(SPEC, dense, store_state_values=False)
    first, second = meta["computer"].exploitability()
    return {"p_first": float(first), "p_second": float(second),
            "exploitability": float(first + second - 1.0)}


def replace_with_exact_g(self: DeepCFRPlusTrainer, pid: int, roots: int) -> float:
    """Override only the regret target at the sampled infosets."""
    if self.regret_target_mode != "aggregate_then_clip" or self.regret_increment_reach_mode != "none":
        raise ValueError("Exact-g continuation requires conditional aggregate-then-clip targets")
    buffer = self.regret_buffers[pid]
    if buffer.seen != buffer.size:
        raise RuntimeError("The regret buffer overflowed before exact-g replacement")
    n = buffer.size
    if not n:
        return 0.0
    self._aggregate_regret_targets(buffer)
    solver, q, g, qg = exact_tables(self, pid)
    del q, qg
    features = buffer.features[:n]
    unique, inverse = torch.unique(features, dim=0, return_inverse=True)
    ranks = self.spec.ranks
    hand_base = (3 ** np.arange(ranks)).astype(np.int64)
    hand_codes = self.encoder.encode_hands(solver.hands, ())[:, :ranks].astype(np.int64) @ hand_base
    hand_lookup = np.full(3 ** ranks, -1, dtype=np.int32)
    hand_lookup[hand_codes] = np.arange(solver.n_hands)
    encoded = unique.numpy()
    hands = hand_lookup[encoded[:, :ranks].astype(np.int64) @ hand_base]
    if np.any(hands < 0):
        raise AssertionError("Sampled hand not found in exact solver")
    bits = (1 << np.arange(self.encoder.k)).astype(np.int64)
    hids = encoded[:, ranks:].astype(np.int64) @ bits
    if np.any((solver.popcount[hids] & 1) != pid):
        raise AssertionError("Sampled infoset belongs to the wrong player")
    with torch.inference_mode():
        old = torch.relu(self.regret_nets[pid](unique)).numpy()
    if self.regret_accumulation_mode == "cumulative":
        target = old + g[hids, hands]
    else:
        t = self.iteration
        target = ((t - 1) / t) * old + g[hids, hands] / t
    target = np.maximum(target, 0).astype(np.float32)
    target *= solver.legal_mask[hids]
    buffer.targets[:n] = torch.from_numpy(target).index_select(0, inverse)
    return self._train_model(self.regret_nets[pid], self.regret_optimizers[pid],
                             buffer, self.regret_train_steps, strategy_loss=False)


def summarize_audit(audit_dir: Path) -> dict:
    import csv
    with (audit_dir / "metrics.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out: dict = {}
    for scope in ("visited", "all", "unvisited"):
        out[scope] = {}
        for pair in ("old_vs_exact_g", "exact_g_vs_old",
                     "exact_g_vs_sampled", "sampled_vs_fitted",
                     "exact_g_vs_fitted", "exact_g_vs_exact_qg"):
            row = next((r for r in rows if r["scope"] == scope
                        and r["expected_visits_bin"] == "all" and r["pair"] == pair), None)
            if row:
                out[scope][pair] = {
                    key: float(row[key]) for key in (
                        "mean_tv", "reach_weighted_tv", "inv_neg_log_reach_weighted_tv",
                        "mean_kl_nats", "reach_weighted_kl_nats", "regret_rmse",
                        "reach_weighted_regret_rmse", "support_mismatch_fraction")
                }
                out[scope][pair]["infosets"] = int(row["infosets"])
                out[scope][pair]["reach_weight_sum"] = float(row["reach_weight_sum"])
    return out


def monitor_point(trainer: DeepCFRPlusTrainer, out: Path, *, label: str,
                  training_s: float, mode: str, threads: int) -> None:
    snapshot = out / "snapshots" / label
    snapshot.mkdir(parents=True, exist_ok=True)
    save_policy(trainer.average_policy(), str(snapshot / "average_policy"))
    start = time.perf_counter()
    average_eval = exact_exploitability(compile_neural_to_dense(trainer.average_policy(), batch_size=65_536))
    current_eval = exact_exploitability(trainer.current_policy_dense())
    row = {"utc": datetime.now(timezone.utc).isoformat(), "mode": mode, "label": label,
           "training_min": training_s / 60, "iteration": trainer.iteration,
           "average": average_eval, "current": current_eval,
           "exact_eval_s": time.perf_counter() - start}
    if mode == "normal":
        # Audit a loaded copy. The live trainer and its RNG state never receive the audit update.
        audit_dir = out / "audits" / label
        audit_dir.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env.update({"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2",
                    "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "1",
                    "NUMEXPR_NUM_THREADS": "1"})
        cmd = [sys.executable, "-u", str(ROOT / "scripts/audit_cfr_plus_18_late_update.py"),
               "--checkpoint", str(out / "checkpoints" / f"{label}.pt"),
               "--output-dir", str(audit_dir), "--player", "0", "--roots", "4096",
               "--threads", "2", "--save-infoset-metrics"]
        if not (audit_dir / "summary.json").exists():
            with (audit_dir / "console.log").open("w", encoding="utf-8") as log:
                subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                               check=True, timeout=900)
        row["audit"] = summarize_audit(audit_dir)
    append(out / "monitors.jsonl", row)
    print(f"[{mode}] {label} iter={trainer.iteration} avg={average_eval['exploitability']:.6f} "
          f"current={current_eval['exploitability']:.6f}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("normal", "exact_g"), required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--source-checkpoint", type=Path)
    p.add_argument("--training-minutes", type=float, required=True,
                   help="Additional measured training minutes; excludes monitoring overhead")
    p.add_argument("--monitor-minutes", type=float, default=15)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--seed", type=int, default=31)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    if args.training_minutes <= 0 or args.monitor_minutes <= 0 or not 1 <= args.threads <= 16:
        p.error("training-minutes and monitor-minutes must be positive; threads must be 1..16")
    if args.mode == "exact_g" and args.source_checkpoint is None and not args.resume:
        p.error("exact_g requires --source-checkpoint")
    if torch.cuda.is_available() and os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        p.error("This experiment is CPU-only; set CUDA_VISIBLE_DEVICES=''")
    torch.set_num_threads(args.threads)
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "checkpoints").mkdir(exist_ok=True)
    manifest_path, state_path = out / "manifest.json", out / "state.json"
    if args.resume:
        if not manifest_path.exists() or not state_path.exists():
            p.error("Resume requires manifest.json and state.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if manifest["mode"] != args.mode or manifest["monitor_minutes"] != args.monitor_minutes:
            p.error("Resume options differ from the original run")
        trainer = DeepCFRPlusTrainer.load_checkpoint(out / "checkpoints" / state["checkpoint"],
                                                      device="cpu")
        if trainer.iteration != state["iteration"]:
            raise RuntimeError("Checkpoint iteration does not match state")
        elapsed = float(state["measured_training_s"])
        origin = float(manifest["origin_training_min"])
        next_monitor = float(state["next_monitor_min"])
    else:
        if manifest_path.exists():
            p.error("Output directory already contains a run; use --resume")
        trainer = (DeepCFRPlusTrainer.load_checkpoint(args.source_checkpoint, device="cpu")
                   if args.mode == "exact_g" else
                   make_trainer("aggregate_then_clip", args.seed, "none", 500_000, "normalized"))
        if trainer.spec != SPEC or trainer.regret_target_mode != "aggregate_then_clip" or (
                trainer.regret_accumulation_mode != "normalized"):
            raise ValueError("Checkpoint does not match normalized conditional 18-claim recipe")
        elapsed = 0.0
        origin = 330.0 if args.mode == "exact_g" else 0.0
        next_monitor = origin + args.monitor_minutes
        atomic_json(manifest_path, {"mode": args.mode, "spec": SPEC.to_json(),
                    "seed": trainer.seed, "source_checkpoint": str(args.source_checkpoint)
                    if args.source_checkpoint else None,
                    "source_iteration": trainer.iteration, "origin_training_min": origin,
                    "monitor_minutes": args.monitor_minutes, "roots_per_player": 4096,
                    "threads": args.threads, "target_additional_minutes": args.training_minutes,
                    "created_utc": datetime.now(timezone.utc).isoformat()})
    if args.mode == "exact_g":
        trainer._train_regret = MethodType(replace_with_exact_g, trainer)
    if args.resume and state["status"] == "monitor_pending":
        label = Path(state["checkpoint"]).stem
        done = (out / "monitors.jsonl")
        completed = any(json.loads(line).get("label") == label
                        for line in done.read_text(encoding="utf-8").splitlines()) if done.exists() else False
        if not completed:
            monitor_point(trainer, out, label=label, training_s=origin * 60 + elapsed,
                          mode=args.mode, threads=args.threads)
        state["status"] = "running"
        atomic_json(state_path, state)
    target_s = args.training_minutes * 60
    while elapsed < target_s:
        start = time.perf_counter()
        record = trainer.run_iteration(traversals_per_player=4096)
        iteration_s = time.perf_counter() - start
        elapsed += iteration_s
        total_min = origin + elapsed / 60
        append(out / "training.jsonl", {
            "utc": datetime.now(timezone.utc).isoformat(), "mode": args.mode,
            "training_min": total_min, "additional_training_min": elapsed / 60,
            "iteration": trainer.iteration, "iteration_s": iteration_s,
            "regret_loss": record["regret_loss"], "strategy_loss": record["strategy_loss"],
            "new_regret_records": record["new_regret_records"],
            "new_strategy_records": record["new_strategy_records"],
            "regret_buffer_sizes": record["regret_buffer_sizes"],
            "strategy_buffer_sizes": record["strategy_buffer_sizes"],
            "timing": record["timing"], "action_sampling": record["action_sampling"]})
        if total_min >= next_monitor or elapsed >= target_s:
            label = f"{int(round(next_monitor)):04d}m" if total_min >= next_monitor else f"{int(round(total_min)):04d}m"
            checkpoint = out / "checkpoints" / f"{label}.pt"
            tmp = checkpoint.with_suffix(".tmp")
            trainer.save_checkpoint(tmp)
            os.replace(tmp, checkpoint)
            pending_state = {
                "mode": args.mode, "iteration": trainer.iteration,
                "measured_training_s": elapsed, "checkpoint": checkpoint.name,
                "next_monitor_min": (next_monitor + args.monitor_minutes
                                     if total_min >= next_monitor else next_monitor),
                "status": "monitor_pending", "updated_utc": datetime.now(timezone.utc).isoformat()}
            atomic_json(state_path, pending_state)
            monitor_point(trainer, out, label=label, training_s=total_min * 60,
                          mode=args.mode, threads=args.threads)
            pending_state["status"] = "running"
            atomic_json(state_path, pending_state)
            # Keep two complete checkpoints. The historical source checkpoint is never touched.
            checkpoints = sorted((out / "checkpoints").glob("*.pt"))
            for older in checkpoints[:-2]:
                older.unlink()
            next_monitor = float(pending_state["next_monitor_min"])
        if trainer.iteration % 100 == 0:
            print(f"[{args.mode}] train={total_min:.1f}m iter={trainer.iteration} "
                  f"iter_s={iteration_s:.2f}", flush=True)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["status"] = "complete"
    atomic_json(state_path, state)
    print(f"complete: {out} iter={trainer.iteration} additional_train={elapsed/60:.1f}m", flush=True)


if __name__ == "__main__":
    main()
