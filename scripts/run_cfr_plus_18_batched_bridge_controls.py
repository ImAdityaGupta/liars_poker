#!/usr/bin/env python3
"""Two from-scratch batched bridge controls: exact K4096 and neural K1024."""

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

from liars_poker.algo.cfr_discount_exact_average import ExactAverageTabularDiscountTrainer
from liars_poker.algo.cfr_discount_tabular import TabularDiscountTrainer
from liars_poker.serialization import save_policy
from scripts.run_cfr_plus_18_tabular_discount import (
    append_jsonl, atomic_json, make_trainer, save_progress, utc,
)


CONTROLS = {
    "exact4096": {"roots": 4096, "average_kind": "exact"},
    "neural1024": {"roots": 1024, "average_kind": "neural"},
}
SNAPSHOT_MINUTES = 15


def policy_for(trainer: TabularDiscountTrainer, average_kind: str):
    return (trainer.exact_average_policy() if average_kind == "exact"
            else trainer.average_policy())


def ready_snapshot(trainer: TabularDiscountTrainer, run_dir: Path,
                   average_kind: str, label: str, measured_s: float,
                   next_snapshot_s: float) -> None:
    parent = run_dir / "policy_snapshots" / label
    parent.mkdir(parents=True, exist_ok=True)
    staged = parent / "average_policy.tmp"
    final = parent / "average_policy"
    if staged.exists():
        shutil.rmtree(staged)
    if final.exists():
        raise RuntimeError(f"Refusing to replace committed policy: {final}")
    save_policy(policy_for(trainer, average_kind), str(staged))
    save_progress(trainer, run_dir, measured_s, next_snapshot_s,
                  next_snapshot_s=next_snapshot_s, pending_snapshot=label)
    os.replace(staged, final)
    atomic_json(parent / "READY.json", {
        "snapshot": label, "iteration": trainer.iteration,
        "measured_training_min": measured_s / 60,
        "policy_dir": str(final), "average_kind": average_kind, "utc": utc(),
    })
    print(f"[snapshot] {run_dir.name} {label} iter={trainer.iteration}", flush=True)


def finish_pending(trainer: TabularDiscountTrainer, run_dir: Path,
                   average_kind: str, state: dict) -> None:
    label = state.get("pending_snapshot")
    if label is None:
        return
    parent = run_dir / "policy_snapshots" / label
    staged = parent / "average_policy.tmp"
    final = parent / "average_policy"
    if not final.exists():
        if not staged.exists():
            save_policy(policy_for(trainer, average_kind), str(staged))
        os.replace(staged, final)
    ready = parent / "READY.json"
    if not ready.exists():
        atomic_json(ready, {
            "snapshot": label, "iteration": trainer.iteration,
            "measured_training_min": state["measured_training_s"] / 60,
            "policy_dir": str(final), "average_kind": average_kind, "utc": utc(),
        })


def run(root: Path, control: str, target_minutes: float, threads: int) -> None:
    config = CONTROLS[control]
    average_kind = config["average_kind"]
    trainer_type = (ExactAverageTabularDiscountTrainer if average_kind == "exact"
                    else TabularDiscountTrainer)
    run_dir = root / control
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = run_dir / "latest_checkpoint.pt"
    manifest_path = run_dir / "manifest.json"
    expected = {
        "control": control, "seed": 17, "roots_per_player": config["roots"],
        "average_kind": average_kind, "regret_rule": "cfr_plus",
        "regret_units": "cumulative conditional mean; clip after update",
        "traversal_backend": "gpu_native on CPU",
        "strategy_weighting": "linear", "snapshot_minutes": SNAPSHOT_MINUTES,
    }
    torch.set_num_threads(threads)
    if checkpoint.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if any(manifest.get(k) != v for k, v in expected.items()):
            raise ValueError("Manifest mismatch; refusing to resume")
        trainer = trainer_type.load_fork_checkpoint(checkpoint)
        state = trainer._experiment_progress
        if state is None or state["iteration"] != trainer.iteration:
            raise ValueError("Checkpoint progress mismatch")
        measured_s = float(state["measured_training_s"])
        next_snapshot_s = float(state["next_snapshot_s"])
        finish_pending(trainer, run_dir, average_kind, state)
        label = f"{int(round(next_snapshot_s / 60)):04d}m"
        orphan = run_dir / "policy_snapshots" / label / "average_policy.tmp"
        if orphan.exists():
            shutil.rmtree(orphan)
        print(f"[resume] {control} iter={trainer.iteration} "+
              f"train={measured_s/60:.2f}m", flush=True)
    else:
        if manifest_path.exists() or (run_dir / "state.json").exists():
            raise ValueError(f"Existing incomplete run: {run_dir}")
        trainer = make_trainer("A_cfr_plus_linear", trainer_type=trainer_type)
        measured_s = 0.0
        next_snapshot_s = SNAPSHOT_MINUTES * 60
        atomic_json(manifest_path, {**expected, "created_utc": utc()})
        save_progress(trainer, run_dir, 0.0, next_snapshot_s,
                      next_snapshot_s=next_snapshot_s)
    if trainer.update_rule != "cfr_plus" or trainer.strategy_weighting != "linear":
        raise ValueError("Unexpected trainer settings")

    stop = [False]
    def request_stop(signum, _frame):
        stop[0] = True
        print(f"[signal] {signum}; checkpoint after current iteration", flush=True)
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    target_s = target_minutes * 60
    while measured_s < target_s and not stop[0] and not (root / "PAUSE").exists():
        start = time.perf_counter()
        average_start = start
        if average_kind == "exact":
            trainer.accumulate_exact_average()
        average_s = time.perf_counter() - average_start
        row = trainer.run_iteration(traversals_per_player=config["roots"])
        iteration_s = time.perf_counter() - start
        measured_s += iteration_s
        append_jsonl(run_dir / "training.jsonl", {
            "utc": utc(), "control": control, "iteration": trainer.iteration,
            "measured_training_min": measured_s / 60,
            "iteration_s": iteration_s, "average_s": average_s,
            "timing": row["timing"], "regret_records": row["new_regret_records"],
        })
        if trainer.iteration % 50 == 0:
            print(f"[train] {control} train={measured_s/60:.1f}m "+
                  f"iter={trainer.iteration} iter_s={iteration_s:.2f} "+
                  f"avg_s={average_s:.2f}", flush=True)
        if measured_s >= next_snapshot_s:
            label = f"{int(round(next_snapshot_s / 60)):04d}m"
            while measured_s >= next_snapshot_s:
                next_snapshot_s += SNAPSHOT_MINUTES * 60
            ready_snapshot(trainer, run_dir, average_kind, label,
                           measured_s, next_snapshot_s)

    save_progress(trainer, run_dir, measured_s, next_snapshot_s,
                  next_snapshot_s=next_snapshot_s)
    status = "paused" if stop[0] or (root / "PAUSE").exists() else "target_reached"
    atomic_json(run_dir / "summary.json", {
        "status": status, "control": control, "iteration": trainer.iteration,
        "measured_training_min": measured_s / 60, "updated_utc": utc(),
    })
    print(f"[done] {control} {status} train={measured_s/60:.1f}m", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--control", choices=CONTROLS, required=True)
    parser.add_argument("--minutes", type=float, default=540)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if args.minutes <= 0 or args.threads <= 0:
        parser.error("Minutes and threads must be positive")
    run(args.output_root.resolve(), args.control, args.minutes, args.threads)


if __name__ == "__main__":
    main()
