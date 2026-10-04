#!/usr/bin/env python3
"""Long CPU comparison of the two neural CFR+ regret-target constructions.

The experiment compares the production target (clip every sampled record) with
the experimental target (average raw updates at each infoset, then clip once).
Each arm runs for a fixed wall-clock budget and writes policies, training rows,
and a rolling checkpoint.  It is deliberately self-contained so it can run
unattended from a terminal.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from liars_poker.algo.deep_cfr_plus import DeepCFRPlusTrainer
from liars_poker.core import GameSpec
from liars_poker.serialization import save_policy


SPEC = GameSpec(
    ranks=4,
    suits=4,
    hand_size=2,
    claim_kinds=("RankHigh", "Pair", "TwoPair", "Trips"),
    suit_symmetry=True,
)

MODES = (
    "clip_each_record", "aggregate_then_clip", "clip_on_read",
    "aggregate_then_clip_on_read",
)


def positive_weight_for_mode(mode: str) -> float:
    return 0.0 if mode in {"clip_on_read", "aggregate_then_clip_on_read"} else 0.5


def json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(type(value).__name__)


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, default=json_default) + "\n")
        handle.flush()


def atomic_json(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, default=json_default), encoding="utf-8")
    tmp.replace(path)


def atomic_checkpoint(trainer: DeepCFRPlusTrainer, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    trainer.save_checkpoint(tmp)
    tmp.replace(path)


def make_trainer(mode: str, seed: int, reach_mode: str = "none",
                 regret_buffer_capacity: int = 500_000,
                 accumulation_mode: str = "normalized",
                 regret_positive_weight: float | None = None) -> DeepCFRPlusTrainer:
    positive_weight = (positive_weight_for_mode(mode)
                       if regret_positive_weight is None
                       else float(regret_positive_weight))
    return DeepCFRPlusTrainer(
        SPEC,
        device="cpu",
        seed=seed,
        # A CPU-sized network keeps the four matched arms feasible overnight.
        # The target-order question is the variable being tested here.
        regret_hidden_sizes=(512, 512),
        strategy_hidden_sizes=(256, 256),
        regret_buffer_capacity=regret_buffer_capacity,
        strategy_buffer_capacity=2_000_000,
        learning_rate=1e-3,
        batch_size=1024,
        regret_train_steps=24,
        strategy_train_steps=6,
        regret_positive_weight=positive_weight,
        regret_target_mode=mode,
        regret_increment_reach_mode=reach_mode,
        regret_accumulation_mode=accumulation_mode,
        strategy_weighting="linear",
        traversal_backend="gpu_native",
        traversal_batch_size=512,
        device_replay=False,
        fused_optimizer=False,
        validation_fraction=0.0,
    )


def load_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def run_arm(root: Path, mode: str, seed: int, arm_hours: float,
            snapshot_minutes: float, checkpoint_minutes: float,
            traversals: int, resume: bool, reach_mode: str = "none",
            regret_buffer_capacity: int = 500_000,
            accumulation_mode: str = "normalized",
            regret_positive_weight: float | None = None) -> None:
    arm = root / f"{mode}__seed_{seed}"
    arm.mkdir(parents=True, exist_ok=True)
    training_path = arm / "training.jsonl"
    manifest_path = arm / "manifest.json"
    state_path = arm / "state.json"
    checkpoint_path = arm / "latest_checkpoint.pt"

    trainer = None
    measured_s = 0.0
    next_snapshot_s = snapshot_minutes * 60.0
    next_checkpoint_s = checkpoint_minutes * 60.0
    if resume and checkpoint_path.exists() and state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("mode") != mode or int(state.get("seed", -1)) != seed:
            raise ValueError(f"Resume state does not match {arm.name}")
        measured_s = float(state["measured_training_s"])
        if measured_s >= arm_hours * 3600.0:
            print(f"[already complete] {arm.name}: {measured_s / 60:.1f}m", flush=True)
            return
        trainer = DeepCFRPlusTrainer.load_checkpoint(checkpoint_path, device="cpu")
        expected_positive_weight = (positive_weight_for_mode(mode)
                                    if regret_positive_weight is None
                                    else float(regret_positive_weight))
        if (trainer.seed != seed or trainer.regret_target_mode != mode
                or trainer.regret_positive_weight != expected_positive_weight
                or trainer.regret_increment_reach_mode != reach_mode
                or trainer.regret_accumulation_mode != accumulation_mode
                or trainer.regret_buffers[0].capacity != regret_buffer_capacity
                or trainer.iteration != int(state["iteration"])):
            raise ValueError(f"Checkpoint and resume state disagree for {arm.name}")
        next_snapshot_s = float(state.get(
            "next_snapshot_s",
            (math.floor(measured_s / (snapshot_minutes * 60.0)) + 1)
            * snapshot_minutes * 60.0,
        ))
        next_checkpoint_s = float(state.get(
            "next_checkpoint_s",
            (math.floor(measured_s / (checkpoint_minutes * 60.0)) + 1)
            * checkpoint_minutes * 60.0,
        ))
        print(f"[resume] {arm.name}: {measured_s / 60:.1f}m iter={trainer.iteration}", flush=True)
    else:
        trainer = make_trainer(
            mode, seed, reach_mode, regret_buffer_capacity, accumulation_mode,
            regret_positive_weight,
        )
        atomic_json(manifest_path, {
            "run_type": "cfr_plus_18_target_order_cpu",
            "spec": SPEC.to_json(),
            "mode": mode,
            "regret_increment_reach_mode": reach_mode,
            "regret_accumulation_mode": accumulation_mode,
            "seed": seed,
            "arm_hours": arm_hours,
            "traversals_per_player": traversals,
            "trainer": {
                "regret_hidden_sizes": [512, 512],
                "strategy_hidden_sizes": [256, 256],
                "regret_buffer_capacity": regret_buffer_capacity,
                "batch_size": 1024,
                "regret_train_steps": 24,
                "strategy_train_steps": 6,
                "learning_rate": 1e-3,
                "traversal_batch_size": 512,
                "regret_target_mode": mode,
                "regret_positive_weight": trainer.regret_positive_weight,
                "regret_increment_reach_mode": reach_mode,
                "regret_accumulation_mode": accumulation_mode,
            },
        })
        # Always make a resumable initial state, so interruption before the
        # first timed checkpoint loses no more than startup overhead.
        atomic_checkpoint(trainer, checkpoint_path)
        atomic_json(state_path, {
            "status": "running", "mode": mode, "seed": seed,
            "iteration": trainer.iteration, "measured_training_s": 0.0,
            "next_snapshot_s": next_snapshot_s,
            "next_checkpoint_s": next_checkpoint_s,
        })

    start = time.perf_counter()
    target_s = arm_hours * 3600.0
    last_report_min = -1
    print(f"\n=== {arm.name}: target {arm_hours:.2f}h ===", flush=True)
    while measured_s < target_s:
        iteration_start = time.perf_counter()
        record = trainer.run_iteration(traversals_per_player=traversals)
        iteration_s = time.perf_counter() - iteration_start
        if not (np.isfinite(record.get("regret_loss", [])).all()
                and np.isfinite(record.get("strategy_loss", [])).all()):
            raise FloatingPointError(
                f"Non-finite fitting loss at iteration {trainer.iteration}; "
                "preserving the previous checkpoint"
            )
        measured_s += iteration_s
        timing = record.get("timing", {})
        row = {
            "utc": datetime.now(timezone.utc).isoformat(),
            "mode": mode,
            "regret_increment_reach_mode": reach_mode,
            "regret_accumulation_mode": accumulation_mode,
            "seed": seed,
            "iteration": trainer.iteration,
            "measured_training_s": measured_s,
            "measured_training_min": measured_s / 60.0,
            "iteration_s": iteration_s,
            "traversals_per_player": traversals,
            "mean_regret_loss": float(np.mean(record.get("regret_loss", [np.nan]))),
            "mean_strategy_loss": float(np.mean(record.get("strategy_loss", [np.nan]))),
            "regret_records": record.get("new_regret_records"),
            "strategy_records": record.get("new_strategy_records"),
            "action_sampling": record.get("action_sampling"),
            "timing": timing,
        }
        append_jsonl(training_path, row)

        while measured_s >= next_snapshot_s:
            label = f"{int(next_snapshot_s // 60):04d}m"
            snapshot = arm / "snapshots" / label
            save_policy(trainer.average_policy(), str(snapshot / "average_policy"))
            save_policy(trainer.current_policy_dense(), str(snapshot / "current_policy"))
            append_jsonl(arm / "events.jsonl", {
                "event": "policy_snapshot",
                "label": label,
                "iteration": trainer.iteration,
                "measured_training_s": measured_s,
            })
            next_snapshot_s += snapshot_minutes * 60.0

        if measured_s >= next_checkpoint_s:
            checkpoint_start = time.perf_counter()
            atomic_checkpoint(trainer, checkpoint_path)
            atomic_json(state_path, {
                "status": "running",
                "mode": mode,
                "seed": seed,
                "iteration": trainer.iteration,
                "measured_training_s": measured_s,
                "next_snapshot_s": next_snapshot_s,
                "next_checkpoint_s": next_checkpoint_s + checkpoint_minutes * 60.0,
                "checkpoint_s": time.perf_counter() - checkpoint_start,
            })
            next_checkpoint_s += checkpoint_minutes * 60.0

        current_min = int(measured_s // 60)
        if current_min != last_report_min and current_min % 10 == 0:
            print(
                f"[{mode} seed={seed}] train={measured_s / 60:.1f}m "
                f"iter={trainer.iteration} iter_s={iteration_s:.2f} "
                f"trav={timing.get('traversal_s', float('nan')):.2f}s "
                f"regfit={timing.get('regret_training_s', float('nan')):.2f}s "
                f"strfit={timing.get('strategy_training_s', float('nan')):.2f}s",
                flush=True,
            )
            last_report_min = current_min

    atomic_checkpoint(trainer, checkpoint_path)
    final = arm / "final_policy"
    save_policy(trainer.average_policy(), str(final / "average_policy"))
    save_policy(trainer.current_policy_dense(), str(final / "current_policy"))
    summary = {
        "status": "complete",
        "mode": mode,
        "regret_increment_reach_mode": reach_mode,
        "regret_accumulation_mode": accumulation_mode,
        "seed": seed,
        "iteration": trainer.iteration,
        "measured_training_s": measured_s,
        "measured_training_min": measured_s / 60.0,
        "final_policy": str(final),
        "checkpoint": str(checkpoint_path),
    }
    atomic_json(state_path, summary)
    append_jsonl(root / "arm_summaries.jsonl", summary)
    print(f"[complete] {arm.name}: {measured_s / 60:.1f}m iter={trainer.iteration}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours-per-arm", type=float, default=1.0)
    parser.add_argument("--traversals", type=int, default=1024)
    parser.add_argument("--snapshot-minutes", type=float, default=15.0)
    parser.add_argument("--checkpoint-minutes", type=float, default=15.0)
    parser.add_argument("--seeds", default="17,23")
    parser.add_argument("--modes", default=",".join(MODES))
    parser.add_argument("--reach-mode", choices=("none", "visit_fraction", "visit_count"), default="none")
    parser.add_argument("--regret-buffer-capacity", type=int, default=500_000)
    parser.add_argument("--regret-accumulation-mode", choices=("normalized", "cumulative"),
                        default="normalized")
    parser.add_argument("--regret-positive-weight", type=float, default=None,
                        help="Override the mode's default positive-target MSE weighting")
    parser.add_argument("--torch-threads", type=int, default=0)
    parser.add_argument("--evaluate-after", action="store_true")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if torch.cuda.is_available():
        raise SystemExit("This script is intentionally CPU-only; use the GPU runner for CUDA.")
    if args.hours_per_arm <= 0 or args.traversals <= 0 or args.regret_buffer_capacity <= 0:
        raise SystemExit("hours-per-arm, traversals, and regret-buffer-capacity must be positive")
    if args.torch_threads < 0:
        raise SystemExit("torch-threads must be nonnegative")
    if args.regret_positive_weight is not None and args.regret_positive_weight < 0:
        raise SystemExit("regret-positive-weight must be nonnegative")
    if args.torch_threads:
        torch.set_num_threads(args.torch_threads)
    selected_modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    if not selected_modes or len(set(selected_modes)) != len(selected_modes) or any(
        mode not in MODES for mode in selected_modes
    ):
        raise SystemExit(f"modes must be a nonempty subset of {MODES}")
    if args.reach_mode in {"visit_fraction", "visit_count"} and selected_modes != ["aggregate_then_clip"]:
        raise SystemExit("visit-based updates require only aggregate_then_clip mode")
    if args.reach_mode == "visit_count" and args.regret_accumulation_mode != "cumulative":
        raise SystemExit("visit_count requires cumulative regret accumulation")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    root = args.output_root or (REPO_ROOT / "artifacts" / "cfr_plus_18_target_order_cpu" / run_id)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    if args.resume:
        if not manifest_path.exists():
            raise SystemExit(f"Cannot resume: missing {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (manifest.get("spec") != SPEC.to_json()
                or int(manifest.get("traversals_per_player", -1)) != args.traversals
                or float(manifest.get("snapshot_minutes", -1)) != args.snapshot_minutes
                or float(manifest.get("checkpoint_minutes", -1)) != args.checkpoint_minutes
                or manifest.get("regret_increment_reach_mode", "none") != args.reach_mode
                or manifest.get("regret_accumulation_mode", "normalized")
                != args.regret_accumulation_mode):
            raise SystemExit("Resume arguments differ from the original run manifest")
        if not set(selected_modes).issubset(manifest.get("modes", [])):
            raise SystemExit("Selected modes are absent from the original run manifest")
        if args.hours_per_arm > float(manifest["hours_per_arm"]):
            append_jsonl(root / "resume_events.jsonl", {
                "utc": datetime.now(timezone.utc).isoformat(),
                "target_hours_per_arm": args.hours_per_arm,
                "seeds": [int(x) for x in args.seeds.split(",")],
                "modes": selected_modes,
                "torch_threads": args.torch_threads,
                "traversals_per_player": args.traversals,
            })
    else:
        atomic_json(manifest_path, {
            "run_type": "cfr_plus_18_target_order_cpu",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "spec": SPEC.to_json(),
            "modes": selected_modes,
            "regret_increment_reach_mode": args.reach_mode,
            "regret_accumulation_mode": args.regret_accumulation_mode,
            "seeds": [int(x) for x in args.seeds.split(",")],
            "hours_per_arm": args.hours_per_arm,
            "traversals_per_player": args.traversals,
            "snapshot_minutes": args.snapshot_minutes,
            "checkpoint_minutes": args.checkpoint_minutes,
        })
    print(f"run_root: {root}", flush=True)
    print("resume with: --output-root", root, "--resume", flush=True)
    for mode in selected_modes:
        for seed in (int(x) for x in args.seeds.split(",")):
            run_arm(
                root, mode, seed, args.hours_per_arm,
                args.snapshot_minutes, args.checkpoint_minutes,
                args.traversals, args.resume, args.reach_mode,
                args.regret_buffer_capacity,
                args.regret_accumulation_mode,
                args.regret_positive_weight,
            )
    print("all arms complete", flush=True)
    if args.evaluate_after:
        print("evaluating saved average policies exactly", flush=True)
        subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts" / "evaluate_cfr_plus_18_target_order_cpu.py"), str(root)],
            check=True,
        )


if __name__ == "__main__":
    main()
