#!/usr/bin/env python3
"""Run one exact-average sampled tabular discount arm on the 18-claim game."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
try:
    import fcntl
except ImportError:  # pragma: no cover - Windows is only used for local checks
    fcntl = None

from liars_poker.algo.cfr_discount_exact_average import ExactAverageTabularDiscountTrainer
from liars_poker.serialization import save_policy
from scripts.run_cfr_plus_18_tabular_discount import (
    ARMS, CHECKPOINT_MINUTES, MONITOR_MINUTES, SEED, TRAVERSALS,
    append_jsonl, atomic_json, make_trainer, read_evals, save_progress, utc,
)

# A is the already-running exact4096 bridge control. These are the five new arms.
NEW_ARMS = tuple(name for name in ARMS if name != "A_cfr_plus_linear")
WEIGHT_POWER = {"uniform": 0, "linear": 1, "quadratic": 2}


@contextmanager
def checkpoint_lock(root: Path):
    lock_path = root / ".checkpoint_write.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as handle:
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def save_checkpoint_safe(trainer, run_dir: Path, measured_s: float,
                         next_monitor_s: float,
                         pending: str | None = None) -> None:
    with checkpoint_lock(run_dir.parent):
        save_progress(trainer, run_dir, measured_s, next_monitor_s, pending)


def save_snapshot(trainer: ExactAverageTabularDiscountTrainer, run_dir: Path,
                  label: str, measured_s: float) -> Path:
    parent = run_dir / "policy_snapshots" / label
    parent.mkdir(parents=True, exist_ok=True)
    staged = parent / "average_policy.tmp"
    final = parent / "average_policy"
    if not final.exists() and staged.exists():
        import shutil
        shutil.rmtree(staged)
    if not final.exists():
        save_policy(trainer.exact_average_policy(), str(staged))
        os.replace(staged, final)
    ready_path = parent / "READY.json"
    if not ready_path.exists():
        atomic_json(ready_path, {
            "snapshot": label, "iteration": trainer.iteration,
            "measured_training_min": measured_s / 60,
            "policy_dir": str(final), "policy_kind": "exact_average",
            "utc": utc(),
        })
    return final


def launch_evaluation(run_dir: Path, label: str, policy_dir: Path) -> None:
    import subprocess

    log_dir = run_dir / "evaluation_workers"
    log_dir.mkdir(parents=True, exist_ok=True)
    with (log_dir / f"{label}.log").open("ab") as log:
        env = dict(os.environ)
        env.update({"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
        subprocess.Popen(
            [sys.executable, "-u", str(ROOT / "scripts" /
             "evaluate_cfr_plus_18_discount_snapshot.py"),
             "--run-dir", str(run_dir), "--policy-dir", str(policy_dir),
             "--snapshot", label],
            cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True,
        )


def launch_missing_evaluations(run_dir: Path) -> None:
    completed = {row.get("snapshot") for row in read_evals(run_dir / "evaluations.jsonl")}
    for ready_path in sorted((run_dir / "policy_snapshots").glob("*/READY.json")):
        ready = json.loads(ready_path.read_text(encoding="utf-8"))
        label = ready["snapshot"]
        if label not in completed:
            launch_evaluation(run_dir, label, Path(ready["policy_dir"]))


def run_arm(root: Path, arm: str, minutes: float, threads: int) -> None:
    rule, weighting = ARMS[arm]
    run_dir = root / arm
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.json"
    checkpoint = run_dir / "latest_checkpoint.pt"
    power = WEIGHT_POWER[weighting]
    expected = {
        "arm": arm, "update_rule": rule, "strategy_weighting": weighting,
        "average_weight_power": power, "seed": SEED,
        "traversals_per_player": TRAVERSALS,
        "monitor_minutes": MONITOR_MINUTES,
        "checkpoint_minutes": CHECKPOINT_MINUTES,
        "regret_units": "cumulative conditional advantage; no /t",
        "sampled_reach": "one conditional-mean update if visited; none if unvisited",
        "average_policy": "exact own-reach-weighted tabular average",
        "neural_regret": False, "neural_strategy": False,
    }
    torch.set_num_threads(threads)

    if checkpoint.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if any(manifest.get(k) != v for k, v in expected.items()):
            raise ValueError(f"Manifest mismatch for {arm}; refusing to resume")
        trainer = ExactAverageTabularDiscountTrainer.load_fork_checkpoint(checkpoint)
        state = trainer._experiment_progress
        if state is None or state["iteration"] != trainer.iteration:
            raise ValueError(f"Checkpoint progress mismatch for {arm}")
        if trainer.average_weight_power != power:
            raise ValueError(f"Checkpoint average weight mismatch for {arm}")
        measured_s = float(state["measured_training_s"])
        next_monitor_s = float(state["next_monitor_s"])
        pending = state.get("pending_evaluation")
        if pending:
            completed = any(row.get("snapshot") == pending
                            for row in read_evals(run_dir / "evaluations.jsonl"))
            if not completed:
                policy_dir = save_snapshot(trainer, run_dir, pending, measured_s)
                launch_evaluation(run_dir, pending, policy_dir)
            save_checkpoint_safe(trainer, run_dir, measured_s, next_monitor_s)
        print(f"[resume] {arm} iter={trainer.iteration} "
              f"train={measured_s/60:.1f}m", flush=True)
    else:
        if manifest_path.exists():
            raise ValueError(f"Manifest exists without checkpoint: {run_dir}")
        trainer = make_trainer(
            arm, trainer_type=ExactAverageTabularDiscountTrainer,
            use_neural_regret=False, use_neural_strategy=False,
            average_weight_power=power,
        )
        measured_s = 0.0
        next_monitor_s = MONITOR_MINUTES * 60
        atomic_json(manifest_path, {**expected, "created_utc": utc()})
        save_checkpoint_safe(trainer, run_dir, measured_s, next_monitor_s)
        print(f"[start] {arm} exact average, weight=t^{power}", flush=True)

    # Recover any evaluation job interrupted with the parent, or a snapshot
    # committed immediately before a pause/restart.
    launch_missing_evaluations(run_dir)

    target_s = minutes * 60
    stop = [False]

    def request_stop(signum, _frame):
        stop[0] = True
        print(f"[signal] {signum}; checkpoint after current iteration", flush=True)

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    while (measured_s < target_s and not stop[0]
           and not (root / "PAUSE").exists()):
        started = time.perf_counter()
        trainer.accumulate_exact_average()
        average_s = time.perf_counter() - started
        row = trainer.run_iteration(traversals_per_player=TRAVERSALS)
        iteration_s = time.perf_counter() - started
        measured_s += iteration_s
        append_jsonl(run_dir / "training.jsonl", {
            "utc": utc(), "iteration": trainer.iteration,
            "measured_training_min": measured_s / 60,
            "iteration_s": iteration_s, "average_s": average_s,
            "timing": row["timing"],
            "regret_records": row["new_regret_records"],
        })
        if trainer.iteration % 100 == 0:
            print(f"[train] {arm} {measured_s/60:.1f}m "
                  f"iter={trainer.iteration} iter_s={iteration_s:.2f} "
                  f"average_s={average_s:.2f}", flush=True)
        if measured_s >= next_monitor_s:
            label = f"{int(round(next_monitor_s / 60)):04d}m"
            next_monitor_s += MONITOR_MINUTES * 60
            save_checkpoint_safe(trainer, run_dir, measured_s, next_monitor_s, label)
            policy_dir = save_snapshot(trainer, run_dir, label, measured_s)
            launch_evaluation(run_dir, label, policy_dir)
            save_checkpoint_safe(trainer, run_dir, measured_s, next_monitor_s)

    evaluations = read_evals(run_dir / "evaluations.jsonl")
    if not evaluations or evaluations[-1].get("iteration") != trainer.iteration:
        label = f"{int(round(measured_s / 60)):04d}m_final"
        save_checkpoint_safe(trainer, run_dir, measured_s, next_monitor_s, label)
        policy_dir = save_snapshot(trainer, run_dir, label, measured_s)
        launch_evaluation(run_dir, label, policy_dir)
    save_checkpoint_safe(trainer, run_dir, measured_s, next_monitor_s)
    status = "paused" if stop[0] or (root / "PAUSE").exists() else "target_reached"
    atomic_json(run_dir / "summary.json", {
        "status": status, "iteration": trainer.iteration,
        "measured_training_min": measured_s / 60, "updated_utc": utc(),
    })
    print(f"[done] {arm} {status} train={measured_s/60:.1f}m "
          f"iter={trainer.iteration}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--arm", choices=[*NEW_ARMS, "all"], required=True)
    parser.add_argument("--minutes", type=float, default=540)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if args.minutes <= 0 or args.threads <= 0:
        parser.error("Minutes and threads must be positive")
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    for arm in (NEW_ARMS if args.arm == "all" else (args.arm,)):
        if (root / "PAUSE").exists():
            break
        run_arm(root, arm, args.minutes, args.threads)


if __name__ == "__main__":
    main()
