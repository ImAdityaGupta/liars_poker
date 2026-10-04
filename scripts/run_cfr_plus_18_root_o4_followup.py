#!/usr/bin/env python3
"""18-claim tabular-regret root schedules with neural averaging and GPU O4 refits."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from liars_poker.algo.cfr_discount_tabular import TabularDiscountTrainer
from liars_poker.algo.cfr_discount_exact_average import ExactAverageTabularDiscountTrainer
from liars_poker.serialization import save_policy
from scripts.run_cfr_plus_18_neural_o4_cpu import (
    append_jsonl, atomic_json, fit_one, freeze_input, publish_online, utc,
)
from scripts.run_cfr_plus_18_root_schedules import checkpoint_lock
from scripts.run_cfr_plus_18_tabular_discount import SPEC

ARMS = {
    "k1024": (1024, 2_000_000),
    "k4096": (4096, 2_000_000),
    "k16384": (16384, 2_000_000),
    "k32768": (32768, 2_000_000),
    "ramp": (None, 2_000_000),
    "ramp8m": (None, 8_000_000),
    "ramp_exact": (None, 2_000_000),
}
SNAPSHOT_S = 15 * 60
RAMP_S = 10 * 3600  # Original ramp reaches K=32,768 at minute 600.


def roots_for(arm: str, measured_s: float, target_s: float) -> int:
    fixed = ARMS[arm][0]
    if fixed is not None:
        return fixed
    if measured_s <= RAMP_S or target_s <= RAMP_S:
        fraction = min(1.0, measured_s / RAMP_S)
        value = 512 + (32768 - 512) * fraction
    else:
        fraction = min(1.0, (measured_s - RAMP_S) / (target_s - RAMP_S))
        value = 32768 + (65536 - 32768) * fraction
    return int(round(value / 64)) * 64


def save_checkpoint(trainer: TabularDiscountTrainer, run: Path, progress: dict) -> None:
    with checkpoint_lock(run.parent):
        ckpt = run / "latest_checkpoint.pt"
        old_size = ckpt.stat().st_size if ckpt.exists() else 0
        if shutil.disk_usage(run).free < old_size + 1024**3:
            raise OSError(f"Insufficient disk for atomic checkpoint: {run}")
        state = trainer.checkpoint_dict()
        state["experiment_progress"] = dict(progress)
        staged = run / "latest_checkpoint.pt.tmp"
        try:
            torch.save(state, staged)
            os.replace(staged, ckpt)
        except BaseException:
            staged.unlink(missing_ok=True)
            raise
        atomic_json(run / "state.json", progress)


def shm_input(run: Path, label: str) -> Path:
    return Path("/dev/shm/cfr_plus_18_root_o4") / run.parent.name / run.name / f"{label}.pt"


def ensure_fit_input(trainer: TabularDiscountTrainer, run: Path, progress: dict) -> None:
    label = progress["pending_snapshot"]
    directory = run / "policy_snapshots" / label
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "READY.json").exists():
        return
    final = directory / "FIT_INPUT.pt"
    backing = shm_input(run, label)
    if final.is_symlink() and final.exists():
        return
    if final.is_symlink():
        final.unlink()  # a VM restart cleared /dev/shm; regenerate from checkpoint
    backing.parent.mkdir(parents=True, exist_ok=True)
    staged = backing.with_suffix(".tmp")
    freeze_input(trainer, staged, progress)
    os.replace(staged, backing)
    final.symlink_to(backing)


def publish_exact(trainer: ExactAverageTabularDiscountTrainer,
                  run: Path, progress: dict) -> None:
    directory = run / "policy_snapshots" / progress["pending_snapshot"]
    ready = directory / "EXACT_READY.json"
    if ready.exists():
        return
    final = directory / "exact_policy"
    staged = directory / "exact_policy.tmp"
    if staged.exists():
        shutil.rmtree(staged)
    if not final.exists():
        save_policy(trainer.exact_average_policy(), str(staged))
        os.replace(staged, final)
    atomic_json(ready, {
        "snapshot": directory.name, "policy_kind": "exact",
        "policy_dir": str(final), "iteration": trainer.iteration,
        "measured_training_min": progress["measured_training_s"] / 60,
        "utc": utc(),
    })


def finish_pending(trainer: TabularDiscountTrainer, run: Path, progress: dict) -> None:
    if not progress.get("pending_snapshot"):
        return
    ensure_fit_input(trainer, run, progress)
    directory = run / "policy_snapshots" / progress["pending_snapshot"]
    publish_online(trainer, directory, progress)
    if isinstance(trainer, ExactAverageTabularDiscountTrainer):
        publish_exact(trainer, run, progress)


def run_train(args: argparse.Namespace) -> None:
    torch.set_num_threads(args.threads)
    root = args.output_root.resolve()
    run = root / args.arm
    run.mkdir(parents=True, exist_ok=True)
    ckpt = run / "latest_checkpoint.pt"
    manifest = run / "manifest.json"
    target_s = args.hours * 3600
    expected = {"arm": args.arm, "seed": 17, "target_hours": args.hours,
                "regret_representation": "tabular", "average_kind": "online_plus_o4",
                "reservoir_capacity": ARMS[args.arm][1], "schedule": "linear_measured_time",
                "snapshot_minutes": 15, "fit_steps_per_player": 5000}
    if ckpt.exists():
        actual = json.loads(manifest.read_text(encoding="utf-8"))
        old_target = actual.get("target_hours")
        extending = (args.extend_from_hours is not None
                     and old_target == args.extend_from_hours
                     and args.hours > args.extend_from_hours)
        if (any(actual.get(k) != v for k, v in expected.items()
                if k != "target_hours")
                or (old_target != args.hours and not extending)):
            raise ValueError(f"Manifest mismatch for {args.arm}; refusing to resume")
        cls = (ExactAverageTabularDiscountTrainer if args.arm == "ramp_exact"
               else TabularDiscountTrainer)
        trainer = cls.load_fork_checkpoint(ckpt)
        progress = trainer._experiment_progress
        if progress is None or progress["iteration"] != trainer.iteration:
            raise ValueError("Checkpoint and progress disagree")
        if extending:
            if progress["measured_training_s"] < args.extend_from_hours * 3600:
                raise ValueError("Original training target has not been reached")
            atomic_json(manifest, {**actual, "target_hours": args.hours,
                                   "extended_utc": utc()})
        finish_pending(trainer, run, progress)
        print(f"[resume] {args.arm} iter={trainer.iteration} "
              f"train={progress['measured_training_s']/60:.1f}m", flush=True)
    else:
        if manifest.exists():
            raise RuntimeError(f"Manifest exists without checkpoint: {run}")
        cls = (ExactAverageTabularDiscountTrainer if args.arm == "ramp_exact"
               else TabularDiscountTrainer)
        trainer = cls(
            SPEC, device="cpu", seed=17, update_rule="cfr_plus",
            regret_hidden_sizes=(), strategy_hidden_sizes=(256, 256),
            learning_rate=1e-3, batch_size=1024,
            regret_buffer_capacity=4_000_000,
            strategy_buffer_capacity=ARMS[args.arm][1],
            regret_train_steps=0, strategy_train_steps=6,
            use_regret_network=False, use_strategy_network=True,
            strategy_weighting="linear",
            regret_target_mode="aggregate_then_clip",
            regret_increment_reach_mode="none",
            regret_accumulation_mode="cumulative",
            traversal_backend="gpu_native", traversal_batch_size=512,
            device_replay=True, fused_optimizer=False,
            validation_fraction=0.0,
        )
        trainer.activate_regret_table()
        progress = {"iteration": 0, "measured_training_s": 0.0,
                    "next_snapshot_s": float(SNAPSHOT_S), "pending_snapshot": None,
                    "cumulative_roots_per_player": 0}
        atomic_json(manifest, {**expected, "created_utc": utc()})
        save_checkpoint(trainer, run, progress)

    stop = [False]
    def request_stop(_signum, _frame):
        stop[0] = True
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    while progress["measured_training_s"] < target_s and not stop[0] and not (root / "PAUSE").exists():
        # Each trainer has at most one pending frozen input. Its fitting time
        # never enters the measured training budget.
        while (sum(1 for _ in (run / "policy_snapshots").glob("*/FIT_INPUT.pt"))
               and not stop[0] and not (root / "PAUSE").exists()):
            if progress.get("pending_snapshot"):
                pending = run / "policy_snapshots" / progress["pending_snapshot"] / "FIT_INPUT.pt"
                if pending.is_symlink() and not pending.exists():
                    ensure_fit_input(trainer, run, progress)
            time.sleep(5)
        if stop[0] or (root / "PAUSE").exists():
            break
        k = roots_for(args.arm, progress["measured_training_s"], target_s)
        start = time.perf_counter()
        if isinstance(trainer, ExactAverageTabularDiscountTrainer):
            trainer.accumulate_exact_average()
        row = trainer.run_iteration(traversals_per_player=k)
        elapsed = time.perf_counter() - start
        progress["measured_training_s"] += elapsed
        progress["iteration"] = trainer.iteration
        progress["cumulative_roots_per_player"] += k
        append_jsonl(run / "training.jsonl", {
            "utc": utc(), "arm": args.arm, "iteration": trainer.iteration,
            "measured_training_min": progress["measured_training_s"] / 60,
            "iteration_s": elapsed, "roots_per_player": k,
            "cumulative_roots_per_player": progress["cumulative_roots_per_player"],
            "regret_records": row["new_regret_records"],
            "strategy_records": row["new_strategy_records"],
            "strategy_buffer_sizes": row["strategy_buffer_sizes"],
            "strategy_records_seen": [b.seen for b in trainer.strategy_buffers],
            "timing": row["timing"],
        })
        if progress["measured_training_s"] >= progress["next_snapshot_s"]:
            label = f"{int(progress['next_snapshot_s']/60):04d}m"
            progress["pending_snapshot"] = label
            progress["next_snapshot_s"] += SNAPSHOT_S
            # Freeze before the checkpoint; a restart can regenerate the
            # shared-memory file from that checkpoint if necessary.
            directory = run / "policy_snapshots" / label
            directory.mkdir(parents=True, exist_ok=True)
            save_checkpoint(trainer, run, progress)
            finish_pending(trainer, run, progress)
            print(f"[snapshot] {args.arm} {label} iter={trainer.iteration} "
                  f"k={k} train={progress['measured_training_s']/60:.1f}m", flush=True)
    # Keep the last pending label in the checkpoint until its refit is ready;
    # after a VM restart /dev/shm can then be regenerated from this checkpoint.
    if progress.get("pending_snapshot"):
        final_ready = run / "policy_snapshots" / progress["pending_snapshot"] / "READY.json"
        if final_ready.exists():
            progress["pending_snapshot"] = None
    save_checkpoint(trainer, run, progress)
    status = "complete" if progress["measured_training_s"] >= target_s else "paused"
    atomic_json(run / "summary.json", {
        "status": status, "iteration": trainer.iteration,
        "measured_training_min": progress["measured_training_s"] / 60,
        "cumulative_roots_per_player": progress["cumulative_roots_per_player"],
        "updated_utc": utc(),
    })
    print(f"[done] {args.arm} {status}", flush=True)


def run_fit(args: argparse.Namespace) -> None:
    root = args.output_root.resolve()
    while not (root / "STOP_FITTER").exists():
        pending = sorted(path for arm in ARMS for path in
                         (root / arm / "policy_snapshots").glob("*/FIT_INPUT.pt"))
        if not pending:
            time.sleep(5)
            continue
        for path in pending:
            if not path.exists():
                continue  # broken /dev/shm link; trainer regenerates on resume
            backing = path.resolve()
            fit_one(path, 5000, 16384, args.threads, "cuda")
            if (path.parent / "READY.json").exists():
                backing.unlink(missing_ok=True)


def run_eval(args: argparse.Namespace) -> None:
    torch.set_num_threads(1)
    root = args.output_root.resolve()
    evaluator = ROOT / "scripts/evaluate_cfr_plus_18_fit_snapshot.py"
    while not (root / "STOP_EVALUATOR").exists():
        found = False
        for arm in ARMS:
            run = root / arm
            out = run / "evaluations.jsonl"
            done = set()
            if out.exists():
                for line in out.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        row = json.loads(line)
                        done.add((row["snapshot"], row["policy_kind"]))
            for filename in ("ONLINE_READY.json", "READY.json", "EXACT_READY.json"):
                for ready_path in sorted((run / "policy_snapshots").glob(f"*/{filename}")):
                    ready = json.loads(ready_path.read_text(encoding="utf-8"))
                    key = (ready["snapshot"], ready["policy_kind"])
                    if key in done:
                        continue
                    env = os.environ.copy()
                    env.update({"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1",
                                "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
                    try:
                        result = subprocess.run(
                            [sys.executable, str(evaluator), ready["policy_dir"]],
                            cwd=ROOT, env=env, capture_output=True, text=True,
                            timeout=900, check=True,
                        )
                        score = json.loads(result.stdout.splitlines()[-1])
                    except Exception as exc:
                        print(f"[eval failed] {arm} {key}: {exc}", flush=True)
                        continue
                    append_jsonl(out, {
                        "arm": arm, "snapshot": ready["snapshot"],
                        "policy_kind": ready["policy_kind"],
                        "iteration": ready["iteration"],
                        "measured_training_min": ready["measured_training_min"],
                        "policy_dir": ready["policy_dir"],
                        "p_first": score["p_first"], "p_second": score["p_second"],
                        "exploitability": score["exploitability"],
                        "evaluation_s": score["evaluation_s"], "utc": utc(),
                    })
                    found = True
                    print(f"[eval] {arm} {key} x={score['exploitability']:.6g}", flush=True)
        if found:
            render_plots(root)
        if not found:
            time.sleep(10)


def render_plots(root: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for arm in ARMS:
        path = root / arm / "evaluations.jsonl"
        train_path = root / arm / "training.jsonl"
        if not path.exists() or not train_path.exists():
            continue
        evaluations = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        training = [json.loads(line) for line in train_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        roots_by_iteration = {row["iteration"]: row["cumulative_roots_per_player"] for row in training}
        for kind, style, alpha in (("o4", "-", 1.0), ("online", ":", 0.55), ("exact", "--", 0.8)):
            rows = [row for row in evaluations if row["policy_kind"] == kind]
            if not rows:
                continue
            rows.sort(key=lambda row: row["iteration"])
            xvals = ([row["measured_training_min"] for row in rows],
                     [row["iteration"] for row in rows],
                     [roots_by_iteration.get(row["iteration"], 0) / 1e6 for row in rows])
            for ax, x in zip(axes, xvals):
                ax.plot(x, [row["exploitability"] for row in rows], style,
                        marker="o", markersize=2, alpha=alpha, label=f"{arm} {kind}")
    for ax, xlabel in zip(axes, ("Measured training minutes", "Iteration", "Cumulative roots per player (M)")):
        ax.set(xlabel=xlabel, ylabel="Exact exploitability")
        ax.set_yscale("log")
        ax.grid(alpha=0.25)
    axes[2].legend(fontsize=7, ncol=2, loc="best")
    fig.tight_layout()
    staged = root / "comparison.png.tmp"
    fig.savefig(staged, format="png", dpi=130)
    os.replace(staged, root / "comparison.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "fit", "eval"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--hours", type=float, default=10.0)
    parser.add_argument("--extend-from-hours", type=float)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if args.command == "train":
        if args.arm is None:
            parser.error("train requires --arm")
        run_train(args)
    elif args.command == "fit":
        run_fit(args)
    else:
        run_eval(args)


if __name__ == "__main__":
    main()
