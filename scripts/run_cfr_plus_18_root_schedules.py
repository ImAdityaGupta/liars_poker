#!/usr/bin/env python3
"""Matched-total-root schedules with tabular regrets and an exact average."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
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
try:
    import fcntl
except ImportError:  # pragma: no cover - Windows is only used for local checks
    fcntl = None

from liars_poker.algo.cfr_discount_exact_average import ExactAverageTabularDiscountTrainer
from liars_poker.serialization import save_policy
from scripts.run_cfr_plus_18_tabular_discount import (
    append_jsonl, atomic_json, make_trainer, save_progress, utc,
)

DEFAULT_HOURS = 9.0
EVALUATION_INTERVAL_S = 15 * 60
CONSTANT_ROOTS = {
    "k0256": 256,
    "k0512": 512,
    "k1024": 1024,
    "k2048": 2048,
    "k8192": 8192,
    "k16384": 16384,
}
DYNAMIC_ARMS = ("ramp_up", "ramp_down", "step_late", "step_early")
ARMS = (*CONSTANT_ROOTS, *DYNAMIC_ARMS)
ROOT_MIN, ROOT_MAX = 512, 7680
RAMP_EXTENSION_END_ROOTS = 32768


@contextmanager
def checkpoint_lock(root: Path):
    """Serialize atomic checkpoint rewrites across arms to cap disk peaks."""
    lock_path = root / ".checkpoint_write.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock_file:
        if fcntl is not None:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def save_progress_locked(trainer, directory: Path, measured_s: float,
                         next_evaluation_s: float,
                         *, pending_snapshot: str | None = None) -> None:
    with checkpoint_lock(directory.parent):
        save_progress(
            trainer, directory, measured_s, next_evaluation_s,
            next_snapshot_s=next_evaluation_s,
            pending_snapshot=pending_snapshot,
        )


def roots_at_elapsed_fraction(arm: str, fraction: float) -> int:
    """Choose K by elapsed training-time fraction, so arms have no iteration cap."""
    if arm not in ARMS or not 0.0 <= fraction <= 1.0:
        raise ValueError("Unknown arm or invalid elapsed-time fraction")
    if arm in CONSTANT_ROOTS:
        return CONSTANT_ROOTS[arm]
    if arm in {"ramp_up", "ramp_down"}:
        progress = fraction if arm == "ramp_up" else 1.0 - fraction
        value = ROOT_MIN + (ROOT_MAX - ROOT_MIN) * progress
    else:
        high_first = arm == "step_early"
        first, second = ((ROOT_MAX, ROOT_MIN) if high_first
                         else (ROOT_MIN, ROOT_MAX))
        value = first if fraction < 0.5 else second
    return max(ROOT_MIN, int(round(value / 64)) * 64)


def roots_at_training_time(arm: str, measured_s: float, target_s: float,
                           continuation_from_s: float | None) -> int:
    """Continue ramp_up from its original endpoint when extending a completed run."""
    if arm == "ramp_up" and continuation_from_s is not None:
        if measured_s <= continuation_from_s:
            return roots_at_elapsed_fraction(arm, measured_s / continuation_from_s)
        extension_s = max(1.0, target_s - continuation_from_s)
        progress = min(1.0, (measured_s - continuation_from_s) / extension_s)
        value = ROOT_MAX + (RAMP_EXTENSION_END_ROOTS - ROOT_MAX) * progress
        return max(ROOT_MAX, int(round(value / 64)) * 64)
    return roots_at_elapsed_fraction(arm, min(1.0, measured_s / target_s))


def snapshot(trainer: ExactAverageTabularDiscountTrainer, run: Path,
             measured_s: float, cumulative_roots: int, label: str,
             next_evaluation_s: float) -> None:
    directory = run / "policy_snapshots" / label
    directory.mkdir(parents=True, exist_ok=True)
    staged = directory / "average_policy.tmp"
    final = directory / "average_policy"
    if final.exists():
        raise RuntimeError(f"Refusing to overwrite {final}")
    if staged.exists():
        shutil.rmtree(staged)
    save_policy(trainer.exact_average_policy(), str(staged))
    save_progress_locked(
        trainer, run, measured_s, next_evaluation_s, pending_snapshot=label
    )
    os.replace(staged, final)
    atomic_json(directory / "READY.json", {
        "snapshot": label, "iteration": trainer.iteration,
        "measured_training_min": measured_s / 60,
        "cumulative_roots_per_player": cumulative_roots,
        "policy_dir": str(final), "utc": utc(),
    })


def run(args: argparse.Namespace) -> None:
    torch.set_num_threads(args.threads)
    root = args.output_root.resolve()
    directory = root / args.arm
    directory.mkdir(parents=True, exist_ok=True)
    ckpt = directory / "latest_checkpoint.pt"
    manifest = directory / "manifest.json"
    target_s = args.hours * 60 * 60
    expected = {"arm": args.arm, "seed": 17, "target_hours": args.hours,
                "average_kind": "exact", "evaluation_interval_min": 15,
                "regret_representation": "tabular",
                "use_regret_network": False, "use_strategy_network": False,
                "root_schedule": args.arm,
                "schedule_clock": "measured_training_time"}
    training_log = directory / "training.jsonl"
    cumulative = 0
    if training_log.exists():
        for line in training_log.read_text(encoding="utf-8").splitlines():
            try:
                cumulative += int(json.loads(line).get("roots_per_player", 0))
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
    if ckpt.exists():
        actual = json.loads(manifest.read_text(encoding="utf-8"))
        mismatch = [k for k, v in expected.items()
                    if k != "target_hours" and actual.get(k) != v]
        if mismatch:
            raise ValueError("Manifest mismatch; refusing to resume")
        old_hours = float(actual.get("target_hours", 0))
        continuation_from_hours = args.extend_from_hours
        if old_hours != args.hours:
            if continuation_from_hours is None or old_hours != continuation_from_hours or args.hours <= old_hours:
                raise ValueError("Target-hours change requires --extend-from-hours matching the saved run")
            history = actual.setdefault("continuations", [])
            history.append({"from_hours": old_hours, "to_hours": args.hours,
                            "utc": utc(), "ramp_up_end_roots":
                            RAMP_EXTENSION_END_ROOTS if args.arm == "ramp_up" else None})
            actual["target_hours"] = args.hours
            atomic_json(manifest, actual)
        elif continuation_from_hours is None:
            continuation_from_hours = actual.get("continuations", [{}])[-1].get("from_hours") if actual.get("continuations") else None
        continuation_from_s = (float(continuation_from_hours) * 3600
                               if continuation_from_hours is not None else None)
        trainer = ExactAverageTabularDiscountTrainer.load_fork_checkpoint(ckpt)
        state = trainer._experiment_progress
        if state is None or state["iteration"] != trainer.iteration:
            raise ValueError("Checkpoint progress mismatch")
        measured_s = float(state["measured_training_s"])
        next_evaluation_s = float(state.get(
            "next_snapshot_s",
            (int(measured_s // EVALUATION_INTERVAL_S) + 1) * EVALUATION_INTERVAL_S,
        ))
        label = state.get("pending_snapshot")
        if label and not (directory / "policy_snapshots" / label / "READY.json").exists():
            pending = directory / "policy_snapshots" / label
            staged, final = pending / "average_policy.tmp", pending / "average_policy"
            if not final.exists():
                if not staged.exists():
                    save_policy(trainer.exact_average_policy(), str(staged))
                os.replace(staged, final)
            atomic_json(pending / "READY.json", {
                "snapshot": label, "iteration": trainer.iteration,
                "measured_training_min": measured_s / 60,
                "cumulative_roots_per_player": cumulative,
                "policy_dir": str(final), "utc": utc(),
            })
    else:
        continuation_from_s = None
        if manifest.exists():
            raise RuntimeError("Manifest exists without checkpoint")
        trainer = make_trainer(
            "A_cfr_plus_linear",
            trainer_type=ExactAverageTabularDiscountTrainer,
            use_neural_regret=False,
            use_neural_strategy=False,
        )
        measured_s = 0.0
        next_evaluation_s = float(EVALUATION_INTERVAL_S)
        atomic_json(manifest, {**expected, "created_utc": utc()})
        save_progress_locked(trainer, directory, 0.0, next_evaluation_s)
    stop = [False]
    def on_stop(_signum, _frame):
        stop[0] = True
    signal.signal(signal.SIGINT, on_stop)
    signal.signal(signal.SIGTERM, on_stop)
    # Reconstruct the root counter from the durable training log on resume.
    smoke_limit = args.stop_after_iterations
    while measured_s < target_s and not stop[0] and not (root / "PAUSE").exists():
        roots = roots_at_training_time(
            args.arm, measured_s, target_s, continuation_from_s
        )
        started = time.perf_counter()
        trainer.accumulate_exact_average()
        average_s = time.perf_counter() - started
        result = trainer.run_iteration(traversals_per_player=roots)
        elapsed = time.perf_counter() - started
        measured_s += elapsed
        cumulative += roots
        append_jsonl(directory / "training.jsonl", {
            "utc": utc(), "arm": args.arm, "iteration": trainer.iteration,
            "roots_per_player": roots, "cumulative_roots_per_player": cumulative,
            "measured_training_min": measured_s / 60,
            "iteration_s": elapsed, "average_s": average_s,
            "regret_records": result["new_regret_records"],
            "visit_stats": getattr(trainer, "last_visit_stats", [{}, {}]),
            "timing": result["timing"],
        })
        if measured_s >= next_evaluation_s or measured_s >= target_s:
            label = (f"{int(round(next_evaluation_s / 60)):04d}m"
                     if measured_s >= next_evaluation_s else
                     f"{int(round(measured_s / 60)):04d}m_final")
            following_evaluation_s = (
                (int(measured_s // EVALUATION_INTERVAL_S) + 1)
                * EVALUATION_INTERVAL_S
            )
            snapshot(
                trainer, directory, measured_s, cumulative, label,
                following_evaluation_s,
            )
            next_evaluation_s = following_evaluation_s
            print(f"[{args.arm}] iter={trainer.iteration} roots={cumulative} "
                  f"train={measured_s/60:.1f}m", flush=True)
        if smoke_limit is not None and trainer.iteration >= smoke_limit:
            break
    save_progress_locked(trainer, directory, measured_s, next_evaluation_s)
    status = ("paused" if stop[0] or (root / "PAUSE").exists() else
              "target_reached" if measured_s >= target_s else
              "stopped_for_smoke" if smoke_limit is not None and trainer.iteration >= smoke_limit else
              "paused" if stop[0] or (root / "PAUSE").exists() else "paused")
    atomic_json(directory / "summary.json", {
        "status": status,
        "iteration": trainer.iteration, "measured_training_min": measured_s / 60,
        "target_training_hours": args.hours,
        "cumulative_roots_per_player": cumulative, "updated_utc": utc(),
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--hours", type=float, default=DEFAULT_HOURS,
                        help="Measured CFR training hours per arm (default: 9)")
    parser.add_argument("--extend-from-hours", type=float,
                        help="Explicitly extend an existing run from its saved total-hours target")
    parser.add_argument("--stop-after-iterations", type=int,
                        help="End after this iteration, leaving a resumable checkpoint")
    args = parser.parse_args()
    if args.threads <= 0 or args.hours <= 0:
        parser.error("threads and hours must be positive")
    if args.stop_after_iterations is not None and args.stop_after_iterations <= 0:
        parser.error("stop-after-iterations must be positive")
    run(args)
