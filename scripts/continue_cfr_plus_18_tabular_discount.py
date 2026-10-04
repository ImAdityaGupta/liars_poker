#!/usr/bin/env python3
"""Continue a discount arm with five-minute policy snapshots for external BR evaluation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from liars_poker.algo.cfr_discount_tabular import TabularDiscountTrainer
from liars_poker.serialization import save_policy
from scripts.run_cfr_plus_18_tabular_discount import (
    ARMS, MONITOR_MINUTES, append_jsonl, atomic_json, save_progress, utc,
)


def snapshot(trainer: TabularDiscountTrainer, run_dir: Path, label: str,
             measured_s: float, next_monitor_s: float, next_snapshot_s: float) -> None:
    """Checkpoint the model before making its policy visible to evaluator workers."""
    parent = run_dir / "policy_snapshots" / label
    parent.mkdir(parents=True, exist_ok=True)
    staged = parent / "average_policy.tmp"
    final = parent / "average_policy"
    if staged.exists():
        shutil.rmtree(staged)
    if final.exists():
        raise RuntimeError(f"A committed snapshot already exists: {final}")
    save_policy(trainer.average_policy(), str(staged))
    save_progress(trainer, run_dir, measured_s, next_monitor_s,
                  next_snapshot_s=next_snapshot_s, pending_snapshot=label)
    os.replace(staged, final)
    ready = {"snapshot": label, "iteration": trainer.iteration,
             "measured_training_min": measured_s / 60,
             "policy_dir": str(final), "utc": utc()}
    atomic_json(parent / "READY.json", ready)
    print(f"[snapshot] {run_dir.name} {label} iter={trainer.iteration}", flush=True)


def finish_pending(trainer: TabularDiscountTrainer, run_dir: Path,
                   state: dict) -> None:
    label = state.get("pending_snapshot")
    if label is None:
        return
    parent = run_dir / "policy_snapshots" / label
    final = parent / "average_policy"
    staged = parent / "average_policy.tmp"
    if not final.exists():
        if not staged.exists():
            save_policy(trainer.average_policy(), str(staged))
        os.replace(staged, final)
    ready_path = parent / "READY.json"
    if not ready_path.exists():
        atomic_json(ready_path, {
            "snapshot": label, "iteration": trainer.iteration,
            "measured_training_min": state["measured_training_s"] / 60,
            "policy_dir": str(final), "utc": utc(),
        })


def run(root: Path, arm: str, target_min: float, threads: int) -> None:
    run_dir = root / arm
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if arm not in ARMS or manifest["arm"] != arm or manifest["traversals_per_player"] != 4096:
        raise ValueError(f"Unexpected manifest for {arm}")
    checkpoint = run_dir / "latest_checkpoint.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    torch.set_num_threads(threads)
    trainer = TabularDiscountTrainer.load_fork_checkpoint(checkpoint)
    state = trainer._experiment_progress
    if state is None or state.get("pending_evaluation") is not None:
        raise ValueError(f"Resolve the old evaluator before continuing {arm}")
    if trainer.iteration != state["iteration"] or trainer.update_rule != ARMS[arm][0]:
        raise ValueError(f"Checkpoint mismatch for {arm}")
    measured_s = float(state["measured_training_s"])
    next_monitor_s = float(state["next_monitor_s"])
    next_snapshot_s = float(state.get(
        "next_snapshot_s", (int(measured_s // 300) + 1) * 300,
    ))
    finish_pending(trainer, run_dir, state)
    # The checkpoint is the committed cursor. Discard any uncommitted staged
    # snapshot from a crash; a READY file cannot precede its checkpoint.
    next_label = f"{int(round(next_snapshot_s / 60)):04d}m"
    orphan = run_dir / "policy_snapshots" / next_label / "average_policy.tmp"
    if orphan.exists():
        shutil.rmtree(orphan)

    print(f"[resume] {arm} iter={trainer.iteration} train={measured_s/60:.2f}m "+
          f"next_snapshot={next_snapshot_s/60:.0f}m", flush=True)
    stop = [False]
    def request_stop(signum, _frame):
        stop[0] = True
        print(f"[signal] {signum}; checkpoint after current iteration", flush=True)
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    target_s = target_min * 60
    while measured_s < target_s and not stop[0] and not (root / "PAUSE").exists():
        start = time.perf_counter()
        row = trainer.run_iteration(traversals_per_player=4096)
        iteration_s = time.perf_counter() - start
        measured_s += iteration_s
        append_jsonl(run_dir / "training.jsonl", {
            "utc": utc(), "iteration": trainer.iteration,
            "measured_training_min": measured_s / 60,
            "iteration_s": iteration_s, "timing": row["timing"],
            "regret_records": row["new_regret_records"],
        })
        if trainer.iteration % 100 == 0:
            print(f"[train] {arm} {measured_s/60:.1f}m iter={trainer.iteration} "+
                  f"iteration_s={iteration_s:.2f}", flush=True)
        if measured_s >= next_snapshot_s:
            label = f"{int(round(next_snapshot_s / 60)):04d}m"
            while measured_s >= next_snapshot_s:
                next_snapshot_s += 300
            while measured_s >= next_monitor_s:
                next_monitor_s += MONITOR_MINUTES * 60
            snapshot(trainer, run_dir, label, measured_s,
                     next_monitor_s, next_snapshot_s)

    save_progress(trainer, run_dir, measured_s, next_monitor_s,
                  next_snapshot_s=next_snapshot_s)
    status = "paused" if stop[0] or (root / "PAUSE").exists() else "target_reached"
    atomic_json(run_dir / "summary.json", {
        "status": status, "iteration": trainer.iteration,
        "measured_training_min": measured_s / 60, "updated_utc": utc(),
    })
    print(f"[done] {arm} {status} train={measured_s/60:.1f}m", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--minutes-per-arm", type=float, default=540)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if args.minutes_per_arm <= 0 or args.threads <= 0:
        parser.error("Minutes and threads must be positive")
    run(args.output_root.resolve(), args.arm, args.minutes_per_arm, args.threads)


if __name__ == "__main__":
    main()
