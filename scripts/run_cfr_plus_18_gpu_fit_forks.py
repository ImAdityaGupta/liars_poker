#!/usr/bin/env python3
"""Resume sequential GPU regret-fit forks from one frozen 18-claim checkpoint."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from liars_poker.algo.deep_cfr_plus import DeepCFRPlusTrainer
from liars_poker.core import GameSpec
from liars_poker.policies.neural_regret import NeuralRegretMatchingPolicy
from liars_poker.serialization import save_policy

SPEC = GameSpec(ranks=4, suits=4, hand_size=2,
                claim_kinds=("RankHigh", "Pair", "TwoPair", "Trips"),
                suit_symmetry=True)


def write_json(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def append_jsonl(path: Path, value: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value) + "\n")


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_snapshot(trainer: DeepCFRPlusTrainer, arm_dir: Path, label: str,
                  branch_s: float, source_s: float) -> None:
    parent = arm_dir / "snapshots"
    parent.mkdir(exist_ok=True)
    final = parent / label
    if not final.exists():
        temporary = parent / f".{label}.{os.getpid()}.tmp"
        temporary.mkdir(exist_ok=False)
        save_policy(trainer.average_policy(), str(temporary / "average_policy"))
        current = NeuralRegretMatchingPolicy.from_models(
            trainer.spec, trainer.regret_nets,
            hidden_sizes=trainer.regret_hidden_sizes, device="cpu",
        )
        save_policy(current, str(temporary / "current_policy"))
        write_json(temporary / "snapshot.json", {
            "label": label, "iteration": trainer.iteration,
            "branch_training_s": branch_s,
            "total_training_s": source_s + branch_s,
        })
        os.replace(temporary, final)
    metadata = json.loads((final / "snapshot.json").read_text(encoding="utf-8"))
    if metadata["iteration"] != trainer.iteration:
        raise RuntimeError(f"Existing snapshot has different iteration: {final}")
    events = arm_dir / "events.jsonl"
    if not events.exists() or not any(
        json.loads(line).get("label") == label
        for line in events.read_text(encoding="utf-8").splitlines() if line.strip()
    ):
        append_jsonl(events, {"event": "policy_snapshot", **metadata})


def save_checkpoint(trainer: DeepCFRPlusTrainer, arm_dir: Path,
                    state: dict, label: str, interval_s: float) -> None:
    if shutil.disk_usage(arm_dir).free < 3 * 1024**3:
        raise OSError("Less than 3 GiB free before checkpoint; previous checkpoint is intact")
    directory = arm_dir / "checkpoints"
    directory.mkdir(exist_ok=True)
    path = directory / f"{label}.pt"
    temporary = path.with_suffix(".pt.tmp")
    trainer.save_checkpoint(temporary)
    os.replace(temporary, path)
    state["checkpoint"] = str(path.resolve())
    state["iteration"] = trainer.iteration
    state["next_checkpoint_s"] += interval_s
    write_json(arm_dir / "state.json", state)
    checkpoints = sorted(directory.glob("*.pt"))
    for old in checkpoints[:-2]:
        old.unlink()


def run_arm(root: Path, arm: int, source: Path, source_s: float,
            source_iteration: int, hours: float, interval_min: float) -> None:
    directory = root / f"s{arm}"
    directory.mkdir(exist_ok=True)
    state_file = directory / "state.json"
    interval_s = interval_min * 60
    target_s = hours * 3600
    if state_file.exists():
        state = json.loads(state_file.read_text(encoding="utf-8"))
        if state["status"] == "complete":
            print(f"S{arm} already complete", flush=True)
            return
        trainer = DeepCFRPlusTrainer.load_checkpoint(state["checkpoint"], device="cuda")
        if trainer.iteration != state["iteration"]:
            raise ValueError(f"S{arm} checkpoint/state iteration mismatch")
        print(f"resume S{arm}: branch={state['branch_training_s']/60:.1f}m "
              f"iteration={trainer.iteration}", flush=True)
    else:
        trainer = DeepCFRPlusTrainer.load_checkpoint(source, device="cuda")
        if trainer.iteration != source_iteration:
            raise ValueError("Frozen source checkpoint/state iteration mismatch")
        state = {
            "status": "running", "checkpoint": str(source.resolve()),
            "iteration": trainer.iteration, "branch_training_s": 0.0,
            "next_checkpoint_s": interval_s, "next_snapshot_s": interval_s,
        }
        write_json(state_file, state)
        print(f"start S{arm}: source iteration={trainer.iteration}", flush=True)
    if (trainer.regret_target_mode != "aggregate_then_clip"
            or trainer.regret_accumulation_mode != "cumulative"
            or trainer.regret_increment_reach_mode != "none"
            or trainer.traversal_backend != "gpu_native"
            or trainer.spec != SPEC):
        raise ValueError("Source must be the cumulative conditional 18-claim recipe")
    trainer.regret_train_steps = arm
    # A crash can happen after the checkpoint commit but before its snapshot.
    while state["branch_training_s"] >= state["next_snapshot_s"]:
        label = f"{int(round(state['next_snapshot_s']/60)):04d}m"
        save_snapshot(trainer, directory, label, state["branch_training_s"], source_s)
        state["next_snapshot_s"] += interval_s
        write_json(state_file, state)
    while state["branch_training_s"] < target_s:
        start = time.perf_counter()
        record = trainer.run_iteration(traversals_per_player=4096)
        torch.cuda.synchronize()
        iteration_s = time.perf_counter() - start
        if not all(np.isfinite(record[key]).all()
                   for key in ("regret_loss", "strategy_loss")):
            raise FloatingPointError(f"Non-finite S{arm} loss at iteration {trainer.iteration}")
        state["branch_training_s"] += iteration_s
        state["iteration"] = trainer.iteration
        append_jsonl(directory / "training.jsonl", {
            "utc": datetime.now(timezone.utc).isoformat(),
            "iteration": trainer.iteration, "fit_steps": arm,
            "branch_training_s": state["branch_training_s"],
            "total_training_s": source_s + state["branch_training_s"],
            "iteration_s": iteration_s,
            "regret_loss": record["regret_loss"],
            "strategy_loss": record["strategy_loss"],
            "regret_records": record["new_regret_records"],
            "strategy_records": record["new_strategy_records"],
            "timing": record["timing"],
            "gpu_allocated_gib": torch.cuda.memory_allocated() / 1024**3,
            "gpu_reserved_gib": torch.cuda.memory_reserved() / 1024**3,
        })
        while state["branch_training_s"] >= state["next_checkpoint_s"]:
            label = f"{int(round(state['next_checkpoint_s']/60)):04d}m"
            save_checkpoint(trainer, directory, state, label, interval_s)
        while state["branch_training_s"] >= state["next_snapshot_s"]:
            label = f"{int(round(state['next_snapshot_s']/60)):04d}m"
            save_snapshot(trainer, directory, label, state["branch_training_s"], source_s)
            state["next_snapshot_s"] += interval_s
            write_json(state_file, state)
        if trainer.iteration % 500 == 0:
            print(f"S{arm}: branch={state['branch_training_s']/60:.1f}m "
                  f"iter={trainer.iteration} iter_s={iteration_s:.3f} "
                  f"gpu_reserved={torch.cuda.memory_reserved()/1024**3:.2f}GiB",
                  flush=True)
    state["status"] = "complete"
    write_json(state_file, state)
    print(f"complete S{arm}: branch={state['branch_training_s']/60:.1f}m "
          f"iter={trainer.iteration}", flush=True)
    del trainer
    torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--source-state", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hours-per-arm", type=float, default=3)
    parser.add_argument("--arms", default="96,384")
    parser.add_argument("--interval-minutes", type=float, default=15)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    if args.hours_per_arm <= 0 or args.interval_minutes <= 0:
        parser.error("hours and interval must be positive")
    arms = [int(part) for part in args.arms.split(",")]
    if arms != [96, 384]:
        parser.error("This experiment is fixed to sequential arms 96,384")
    torch.set_num_threads(4)
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    source = args.source_checkpoint.resolve()
    source_state = json.loads(args.source_state.read_text(encoding="utf-8"))
    source_s = float(source_state["measured_training_s"])
    source_iteration = int(source_state["iteration"])
    manifest_file = root / "manifest.json"
    if manifest_file.exists():
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        if (manifest["source_sha256"] != hash_file(source)
                or manifest["source_iteration"] != source_iteration
                or manifest["arms"] != arms
                or manifest["hours_per_arm"] != args.hours_per_arm
                or manifest["interval_minutes"] != args.interval_minutes):
            raise ValueError("Arguments or frozen source differ from existing run")
    else:
        write_json(manifest_file, {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_checkpoint": str(source), "source_sha256": hash_file(source),
            "source_iteration": source_iteration,
            "source_training_s": source_s, "arms": arms,
            "hours_per_arm": args.hours_per_arm,
            "interval_minutes": args.interval_minutes,
            "traversals_per_player": 4096,
            "device": torch.cuda.get_device_name(),
        })
    for arm in arms:
        run_arm(root, arm, source, source_s, source_iteration,
                args.hours_per_arm, args.interval_minutes)
    print("all GPU fitting forks complete", flush=True)


if __name__ == "__main__":
    main()
