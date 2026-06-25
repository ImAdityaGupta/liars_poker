#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from liars_poker.algo.deep_cfr_plus import DeepCFRPlusTrainer
from liars_poker.core import GameSpec
from liars_poker.serialization import save_policy


SPEC = GameSpec(
    ranks=6,
    suits=4,
    hand_size=4,
    claim_kinds=("RankHigh", "Pair", "TwoPair", "Trips", "FullHouse", "Quads"),
    suit_symmetry=True,
)

TOTAL_HOURS = 6.0
POLICY_SNAPSHOT_EVERY_MINUTES = 5.0
CHECKPOINT_EVERY_MINUTES = 60.0
PRINT_EVERY_MINUTES = 5.0
VALIDATE_EVERY_MINUTES = 15.0
VALIDATION_RECORDS = 4096

# Best current guess for a long 69-claim run:
# - start with many cheap CFR+ iterations while the policy is moving quickly;
# - drop the optimizer scale once current-strategy oscillation starts to matter;
# - increase traversal data as the regret target gets more delicate;
# - reset Adam state at LR drops, but keep networks and replay buffers.
PHASES = [
    {
        "name": "warmup_lr_1e-3",
        "start_minute": 0.0,
        "learning_rate": 1e-3,
        "traversals_per_player": 4096,
        "regret_train_steps": 48,
        "strategy_train_steps": 6,
        "reset_optimizers": False,
    },
    {
        "name": "mid_lr_3e-4_trav8192",
        "start_minute": 60.0,
        "learning_rate": 3e-4,
        "traversals_per_player": 8192,
        "regret_train_steps": 48,
        "strategy_train_steps": 6,
        "reset_optimizers": True,
    },
    {
        "name": "late_lr_1e-4_trav16384",
        "start_minute": 180.0,
        "learning_rate": 1e-4,
        "traversals_per_player": 16384,
        "regret_train_steps": 48,
        "strategy_train_steps": 6,
        "reset_optimizers": True,
    },
    {
        "name": "polish_lr_3e-5_trav16384",
        "start_minute": 300.0,
        "learning_rate": 3e-5,
        "traversals_per_player": 16384,
        "regret_train_steps": 48,
        "strategy_train_steps": 6,
        "reset_optimizers": True,
    },
]

TRAINER_KWARGS: dict[str, Any] = {
    "regret_hidden_sizes": (2048, 2048),
    "strategy_hidden_sizes": (512, 512),
    "seed": 7,
    # Regret memory is per traverser update and is cleared before each update.
    # Later phases use up to 16k traversals/player, so this must be much
    # larger than the older 500k and larger than one 4k root batch.
    "regret_buffer_capacity": 16_000_000,
    "strategy_buffer_capacity": 2_000_000,
    "learning_rate": 1e-3,
    "batch_size": 4096,
    "regret_train_steps": 48,
    "strategy_train_steps": 6,
    "strategy_weighting": "linear",
    "regret_positive_weight": 0.5,
    "validation_fraction": 0.001,
    "validation_buffer_capacity": 20_000,
    "traversal_backend": "gpu_native",
    "traversal_batch_size": 4096,
    "traverser_action_sample_schedule": (24,),
    "traverser_action_sample_count": None,
    "traverser_action_sample_fraction": None,
    "traverser_action_baseline": "call",
    "traverser_action_sample_mode": "hash",
    "traversal_streaming": True,
    "traversal_live_row_budget": None,
    "traverser_action_chunk_size": 1_048_576,
    "traversal_record_flush_size": 262_144,
    "device_replay": True,
    "fused_optimizer": True,
    "amp_dtype": None,
    "compile_models": False,
}


def json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, tuple):
        return list(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, default=json_default, sort_keys=True) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".new")
    tmp.write_text(
        json.dumps(payload, indent=2, default=json_default, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def last_jsonl(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    last: dict[str, Any] | None = None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                last = json.loads(line)
    return last


def atomic_checkpoint(trainer: DeepCFRPlusTrainer, path: Path) -> float:
    start = time.perf_counter()
    tmp = path.with_suffix(path.suffix + ".new")
    trainer.save_checkpoint(tmp)
    os.replace(tmp, path)
    return time.perf_counter() - start


def copy_checkpoint_snapshot(
    latest_checkpoint: Path,
    run_dir: Path,
    measured_training_s: float,
    metadata: dict[str, Any] | None = None,
) -> tuple[Path, float]:
    target_minute = int(round(measured_training_s / 60.0))
    snapshot_path = (
        run_dir
        / "checkpoints"
        / f"checkpoint_{target_minute:04d}m.pt"
    )
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    tmp = snapshot_path.with_suffix(snapshot_path.suffix + ".new")
    shutil.copy2(latest_checkpoint, tmp)
    os.replace(tmp, snapshot_path)
    if metadata is not None:
        write_json(
            snapshot_path.with_suffix(".json"),
            {
                **metadata,
                "checkpoint_snapshot_path": str(snapshot_path),
            },
        )
    return snapshot_path, time.perf_counter() - start


def optimizer_reset(trainer: DeepCFRPlusTrainer) -> None:
    for optimizer in [*trainer.regret_optimizers, *trainer.strategy_optimizers]:
        optimizer.state.clear()


def set_optimizer_lr(trainer: DeepCFRPlusTrainer, learning_rate: float) -> None:
    trainer.learning_rate = float(learning_rate)
    for optimizer in [*trainer.regret_optimizers, *trainer.strategy_optimizers]:
        for group in optimizer.param_groups:
            group["lr"] = float(learning_rate)


def phase_for(measured_training_s: float) -> tuple[int, dict[str, Any]]:
    measured_min = measured_training_s / 60.0
    phase_index = 0
    for idx, phase in enumerate(PHASES):
        if measured_min >= float(phase["start_minute"]):
            phase_index = idx
    return phase_index, PHASES[phase_index]


def apply_phase(
    trainer: DeepCFRPlusTrainer,
    phase_index: int,
    phase: dict[str, Any],
    completed_phase_resets: set[int],
) -> bool:
    trainer.regret_train_steps = int(phase["regret_train_steps"])
    trainer.strategy_train_steps = int(phase["strategy_train_steps"])
    set_optimizer_lr(trainer, float(phase["learning_rate"]))
    if phase.get("reset_optimizers") and phase_index not in completed_phase_resets:
        optimizer_reset(trainer)
        completed_phase_resets.add(phase_index)
        return True
    return False


def next_after(current_s: float, period_minutes: float) -> float:
    period_s = 60.0 * float(period_minutes)
    return (math.floor(current_s / period_s) + 1) * period_s


def save_average_snapshot(
    trainer: DeepCFRPlusTrainer,
    run_dir: Path,
    measured_training_s: float,
    *,
    label: str | None = None,
) -> dict[str, Any]:
    measured_min = measured_training_s / 60.0
    snapshot_label = label or f"{int(round(measured_min)):04d}m"
    snapshot_dir = run_dir / "policy_snapshots" / snapshot_label
    start = time.perf_counter()
    save_policy(trainer.average_policy(), str(snapshot_dir))
    snapshot_s = time.perf_counter() - start
    metadata = {
        "snapshot_label": snapshot_label,
        "measured_training_s": measured_training_s,
        "measured_training_min": measured_min,
        "iteration": trainer.iteration,
        "path": str(snapshot_dir),
        "snapshot_s": snapshot_s,
    }
    write_json(snapshot_dir / "snapshot.json", metadata)
    return metadata


def mean(values: Any) -> float:
    if values is None:
        return float("nan")
    if isinstance(values, (int, float)):
        return float(values)
    values = list(values)
    return float(np.mean(values)) if values else float("nan")


def summarize_validation(metrics: dict[str, Any] | None) -> dict[str, Any]:
    if not metrics:
        return {}

    out: dict[str, Any] = {}
    for section in ("regret", "strategy"):
        entries = metrics.get(section, []) or []
        for key in ("mse", "strategy_tv", "support_accuracy", "cross_entropy"):
            values = [
                float(entry[key])
                for entry in entries
                if isinstance(entry, dict) and key in entry and entry[key] is not None
            ]
            if values:
                out[f"{section}_{key}"] = float(np.mean(values))
    return out


def make_run_dir(args: argparse.Namespace) -> Path:
    if args.run_dir is not None:
        return Path(args.run_dir).expanduser().resolve()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return (
        REPO_ROOT
        / "artifacts"
        / "cfr_plus_69_claim_overnight"
        / f"{SPEC.to_short_str()}___{run_id}"
    )


def create_trainer(device: torch.device, seed: int) -> DeepCFRPlusTrainer:
    kwargs = dict(TRAINER_KWARGS)
    kwargs["device"] = device
    kwargs["seed"] = seed
    return DeepCFRPlusTrainer(SPEC, **kwargs)


def tensor_bytes(obj: Any) -> int:
    if not torch.is_tensor(obj):
        return 0
    return int(obj.numel() * obj.element_size())


def device_buffer_gib(trainer: DeepCFRPlusTrainer) -> float:
    total = 0
    for buffer in [
        *trainer.regret_buffers,
        *trainer.regret_validation_buffers,
        *trainer.strategy_buffers,
        *trainer.strategy_validation_buffers,
    ]:
        for name in ("features", "targets", "legal_masks", "weights"):
            total += tensor_bytes(getattr(buffer, name, None))
    return total / 2**30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the 69-claim neural CFR+ overnight training job."
    )
    parser.add_argument("--hours", type=float, default=TOTAL_HOURS)
    parser.add_argument("--run-dir", type=str, default=None)
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Existing run directory to resume from. Uses latest_checkpoint.pt.",
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=str,
        default=None,
        help=(
            "Specific checkpoint .pt file to load. If omitted with --resume, "
            "loads <run-dir>/latest_checkpoint.pt."
        ),
    )
    parser.add_argument("--seed", type=int, default=int(TRAINER_KWARGS["seed"]))
    parser.add_argument("--policy-every-minutes", type=float, default=POLICY_SNAPSHOT_EVERY_MINUTES)
    parser.add_argument("--checkpoint-every-minutes", type=float, default=CHECKPOINT_EVERY_MINUTES)
    parser.add_argument("--print-every-minutes", type=float, default=PRINT_EVERY_MINUTES)
    parser.add_argument("--validate-every-minutes", type=float, default=VALIDATE_EVERY_MINUTES)
    parser.add_argument("--validation-records", type=int, default=VALIDATION_RECORDS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this overnight run.")

    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    free, total = torch.cuda.mem_get_info()
    print("repo:", REPO_ROOT, flush=True)
    print("torch:", torch.__version__, "cuda:", torch.version.cuda, flush=True)
    print("gpu:", torch.cuda.get_device_name(0), flush=True)
    print(
        "free / total GPU GiB:",
        round(free / 2**30, 2),
        "/",
        round(total / 2**30, 2),
        flush=True,
    )

    resume_dir = Path(args.resume).expanduser().resolve() if args.resume else None
    resume_checkpoint = (
        Path(args.resume_checkpoint).expanduser().resolve()
        if args.resume_checkpoint
        else None
    )
    if resume_checkpoint is not None and resume_dir is None:
        if resume_checkpoint.parent.name == "checkpoints":
            resume_dir = resume_checkpoint.parent.parent
        else:
            resume_dir = resume_checkpoint.parent
    run_dir = resume_dir if resume_dir is not None else make_run_dir(args)
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "latest_checkpoint.pt"
    state_path = run_dir / "run_state.json"
    manifest_path = run_dir / "manifest.json"
    training_path = run_dir / "training.jsonl"
    event_path = run_dir / "events.jsonl"
    summary_path = run_dir / "summary.json"

    completed_phase_resets: set[int] = set()
    measured_training_s = 0.0

    if resume_dir is not None:
        checkpoint_to_load = resume_checkpoint or checkpoint_path
        if not checkpoint_to_load.exists():
            raise FileNotFoundError(f"Cannot resume: missing {checkpoint_to_load}")
        print("resuming:", run_dir, flush=True)
        print("loading checkpoint:", checkpoint_to_load, flush=True)
        trainer = DeepCFRPlusTrainer.load_checkpoint(checkpoint_to_load, device=device)
        sidecar_path = (
            resume_checkpoint.with_suffix(".json")
            if resume_checkpoint is not None
            else None
        )
        if sidecar_path is not None and sidecar_path.exists():
            state = read_json(sidecar_path)
            measured_training_s = float(state.get("measured_training_s", 0.0))
            completed_phase_resets = {
                int(idx) for idx in state.get("completed_phase_resets", [])
            }
        elif state_path.exists():
            state = read_json(state_path)
            measured_training_s = float(state.get("measured_training_s", 0.0))
            completed_phase_resets = {
                int(idx) for idx in state.get("completed_phase_resets", [])
            }
        else:
            last = last_jsonl(training_path)
            if last is not None:
                measured_training_s = float(last.get("measured_training_s", 0.0))
        # Command-line seed should not overwrite a resumed RNG state.
    else:
        print("new run:", run_dir, flush=True)
        trainer = create_trainer(device, args.seed)
        manifest = {
            "run_type": "cfr_plus_69_claim_overnight",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "spec": SPEC.to_json(),
            "total_hours": args.hours,
            "policy_snapshot_every_minutes": args.policy_every_minutes,
            "checkpoint_every_minutes": args.checkpoint_every_minutes,
            "print_every_minutes": args.print_every_minutes,
            "validate_every_minutes": args.validate_every_minutes,
            "validation_records": args.validation_records,
            "phases": PHASES,
            "trainer_kwargs": {
                **TRAINER_KWARGS,
                "device": str(device),
                "seed": args.seed,
            },
        }
        write_json(manifest_path, manifest)

    print(
        "device replay buffers GiB:",
        round(device_buffer_gib(trainer), 2),
        flush=True,
    )

    budget_s = 60.0 * 60.0 * float(args.hours)
    next_policy_s = next_after(measured_training_s, args.policy_every_minutes)
    next_checkpoint_s = next_after(measured_training_s, args.checkpoint_every_minutes)
    next_print_s = next_after(measured_training_s, args.print_every_minutes)
    next_validate_s = next_after(measured_training_s, args.validate_every_minutes)

    append_jsonl(
        event_path,
        {
            "event": "start_or_resume",
            "utc": datetime.now(timezone.utc).isoformat(),
            "measured_training_s": measured_training_s,
            "iteration": trainer.iteration,
        },
    )

    try:
        while measured_training_s < budget_s:
            phase_index, phase = phase_for(measured_training_s)
            did_reset = apply_phase(
                trainer,
                phase_index,
                phase,
                completed_phase_resets,
            )
            if did_reset:
                event = {
                    "event": "phase_optimizer_reset",
                    "utc": datetime.now(timezone.utc).isoformat(),
                    "phase_index": phase_index,
                    "phase": phase,
                    "measured_training_s": measured_training_s,
                    "measured_training_min": measured_training_s / 60.0,
                    "iteration": trainer.iteration,
                }
                append_jsonl(event_path, event)
                print(
                    f"[phase reset] train={measured_training_s / 60.0:.1f}m "
                    f"iter={trainer.iteration} phase={phase['name']} "
                    f"lr={phase['learning_rate']:.1e}",
                    flush=True,
                )

            start = time.perf_counter()
            record = trainer.run_iteration(
                traversals_per_player=int(phase["traversals_per_player"])
            )
            iteration_s = time.perf_counter() - start
            measured_training_s += iteration_s

            validation = None
            if measured_training_s >= next_validate_s:
                validation = trainer.validation_metrics(
                    max_records=int(args.validation_records)
                )
                while next_validate_s <= measured_training_s:
                    next_validate_s += 60.0 * float(args.validate_every_minutes)

            record.update(
                {
                    "utc": datetime.now(timezone.utc).isoformat(),
                    "measured_training_s": measured_training_s,
                    "measured_training_min": measured_training_s / 60.0,
                    "iteration_s": iteration_s,
                    "phase_index": phase_index,
                    "phase_name": phase["name"],
                    "learning_rate": float(phase["learning_rate"]),
                    "traversals_per_player": int(phase["traversals_per_player"]),
                    "regret_train_steps": int(phase["regret_train_steps"]),
                    "strategy_train_steps": int(phase["strategy_train_steps"]),
                    "mean_regret_loss": mean(record.get("regret_loss")),
                    "mean_strategy_loss": mean(record.get("strategy_loss")),
                    "validation": validation,
                    **{
                        f"validation_{key}": value
                        for key, value in summarize_validation(validation).items()
                    },
                }
            )
            append_jsonl(training_path, record)

            regret_seen = record.get("regret_records_seen", [])
            regret_sizes = record.get("regret_buffer_sizes", [])
            clipped = [
                int(seen) > int(size)
                for seen, size in zip(regret_seen, regret_sizes)
            ]
            if any(clipped):
                append_jsonl(
                    event_path,
                    {
                        "event": "regret_buffer_capacity_clipped",
                        "utc": datetime.now(timezone.utc).isoformat(),
                        "iteration": trainer.iteration,
                        "measured_training_s": measured_training_s,
                        "regret_records_seen": regret_seen,
                        "regret_buffer_sizes": regret_sizes,
                    },
                )

            if measured_training_s >= next_policy_s:
                while next_policy_s <= measured_training_s:
                    label = f"{int(round(next_policy_s / 60.0)):04d}m"
                    snapshot = save_average_snapshot(
                        trainer,
                        run_dir,
                        measured_training_s,
                        label=label,
                    )
                    append_jsonl(event_path, {"event": "policy_snapshot", **snapshot})
                    print(
                        f"[snapshot] target={label} actual={measured_training_s / 60.0:.1f}m "
                        f"iter={trainer.iteration} time={snapshot['snapshot_s']:.2f}s",
                        flush=True,
                    )
                    next_policy_s += 60.0 * float(args.policy_every_minutes)

            if measured_training_s >= next_checkpoint_s:
                ckpt_s = atomic_checkpoint(trainer, checkpoint_path)
                checkpoint_metadata = {
                    "updated_utc": datetime.now(timezone.utc).isoformat(),
                    "checkpoint_target_min": next_checkpoint_s / 60.0,
                    "measured_training_s": measured_training_s,
                    "measured_training_min": measured_training_s / 60.0,
                    "iteration": trainer.iteration,
                    "completed_phase_resets": sorted(completed_phase_resets),
                    "latest_checkpoint_path": str(checkpoint_path),
                }
                checkpoint_snapshot_path, checkpoint_copy_s = copy_checkpoint_snapshot(
                    checkpoint_path,
                    run_dir,
                    next_checkpoint_s,
                    checkpoint_metadata,
                )
                ckpt_gib = checkpoint_path.stat().st_size / 2**30
                state = {
                    **checkpoint_metadata,
                    "checkpoint_path": str(checkpoint_path),
                    "checkpoint_snapshot_path": str(checkpoint_snapshot_path),
                    "checkpoint_GiB": ckpt_gib,
                }
                write_json(state_path, state)
                append_jsonl(
                    event_path,
                    {
                        "event": "checkpoint",
                        **state,
                        "checkpoint_s": ckpt_s,
                        "checkpoint_copy_s": checkpoint_copy_s,
                    },
                )
                print(
                    f"[checkpoint] train={measured_training_s / 60.0:.1f}m "
                    f"iter={trainer.iteration} size={ckpt_gib:.2f}GiB "
                    f"time={ckpt_s:.1f}s copy={checkpoint_copy_s:.1f}s "
                    f"snapshot={checkpoint_snapshot_path.name}",
                    flush=True,
                )
                while next_checkpoint_s <= measured_training_s:
                    next_checkpoint_s += 60.0 * float(args.checkpoint_every_minutes)

            if measured_training_s >= next_print_s:
                timing = record["timing"]
                action_sampling = record.get("action_sampling", {})
                print(
                    f"train={measured_training_s / 60.0:7.1f}m "
                    f"iter={trainer.iteration:7d} "
                    f"phase={phase['name']} "
                    f"lr={float(phase['learning_rate']):.1e} "
                    f"it_s={iteration_s:.2f} "
                    f"trav={timing['traversal_s']:.2f}s "
                    f"regfit={timing['regret_training_s']:.2f}s "
                    f"strfit={timing['strategy_training_s']:.2f}s "
                    f"edges={float(action_sampling.get('claim_edge_fraction', float('nan'))):.3f} "
                    f"ess={float(action_sampling.get('regret_weight_ess_fraction', float('nan'))):.3f} "
                    f"maxw={float(action_sampling.get('max_regret_weight', 0.0)):.1f}",
                    flush=True,
                )
                while next_print_s <= measured_training_s:
                    next_print_s += 60.0 * float(args.print_every_minutes)

    except KeyboardInterrupt:
        append_jsonl(
            event_path,
            {
                "event": "keyboard_interrupt",
                "utc": datetime.now(timezone.utc).isoformat(),
                "measured_training_s": measured_training_s,
                "iteration": trainer.iteration,
            },
        )
        print("Interrupted; saving final checkpoint/policy before exit.", flush=True)
    finally:
        final_policy_dir = run_dir / "final_policy"
        save_start = time.perf_counter()
        save_policy(trainer.average_policy(), str(final_policy_dir))
        final_policy_s = time.perf_counter() - save_start
        ckpt_s = atomic_checkpoint(trainer, checkpoint_path)
        ckpt_gib = checkpoint_path.stat().st_size / 2**30
        state = {
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "measured_training_s": measured_training_s,
            "measured_training_min": measured_training_s / 60.0,
            "iteration": trainer.iteration,
            "completed_phase_resets": sorted(completed_phase_resets),
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_GiB": ckpt_gib,
            "final_policy_dir": str(final_policy_dir),
        }
        write_json(state_path, state)
        summary = {
            "status": "complete" if measured_training_s >= budget_s else "interrupted",
            "run_dir": str(run_dir),
            "spec": SPEC.to_json(),
            "iteration": trainer.iteration,
            "measured_training_s": measured_training_s,
            "measured_training_min": measured_training_s / 60.0,
            "budget_hours": args.hours,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_GiB": ckpt_gib,
            "checkpoint_s": ckpt_s,
            "final_policy_dir": str(final_policy_dir),
            "final_policy_s": final_policy_s,
            "completed_phase_resets": sorted(completed_phase_resets),
        }
        write_json(summary_path, summary)
        append_jsonl(event_path, {"event": "finalize", **summary})
        print("summary:", json.dumps(summary, indent=2, default=json_default), flush=True)


if __name__ == "__main__":
    main()
