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
from liars_poker.policies.neural_regret import NeuralRegretMatchingPolicy
from liars_poker.serialization import save_policy
from liars_poker.training.br_runs import run_best_responder


SPEC = GameSpec(
    ranks=6,
    suits=4,
    hand_size=4,
    claim_kinds=("RankHigh", "Pair", "TwoPair", "Trips", "FullHouse", "Quads"),
    suit_symmetry=True,
)

MONITOR_EVERY_MINUTES = 15.0
LONG_BR_EVERY_MINUTES = 30.0
CHECKPOINT_EVERY_MINUTES = 60.0
PRINT_EVERY_MINUTES = 5.0
VALIDATE_EVERY_MINUTES = 30.0
VALIDATION_RECORDS = 4096
CONTROLLER_TOLERANCE = 0.01
MIN_PHASE_MINUTES = 60.0
PATIENCE_MONITORS = 4

BR_DECISION_MINUTES = 2.0
BR_LONG_MINUTES = 10.0
BR_EVALUATE_EVERY_MINUTES = 1.0
BR_EPISODES_PER_ROLE = 4096
BR_ROLLOUT_BATCH_SIZE = 1024
BR_EVAL_EPISODES_PER_ROLE = 200_000
BR_EVAL_ROLLOUT_BATCH_SIZE = 8192
BR_SEED = 17

PHASES: list[dict[str, Any]] = [
    {
        "name": "p0_lr_1e-3_trav4096_cap24",
        "learning_rate": 1e-3,
        "traversals_per_player": 4096,
        "regret_train_steps": 24,
        "strategy_train_steps": 6,
        "sample_schedule": (24,),
        "reset_optimizers": False,
    },
    {
        "name": "p1_lr_3e-4_trav8192_cap24",
        "learning_rate": 3e-4,
        "traversals_per_player": 8192,
        "regret_train_steps": 24,
        "strategy_train_steps": 6,
        "sample_schedule": (24,),
        "reset_optimizers": True,
    },
    {
        "name": "p2_lr_1e-4_trav16384_cap24",
        "learning_rate": 1e-4,
        "traversals_per_player": 16384,
        "regret_train_steps": 48,
        "strategy_train_steps": 6,
        "sample_schedule": (24,),
        "reset_optimizers": True,
    },
    {
        "name": "p3_lr_3e-5_trav16384_cap24",
        "learning_rate": 3e-5,
        "traversals_per_player": 16384,
        "regret_train_steps": 48,
        "strategy_train_steps": 6,
        "sample_schedule": (24,),
        "reset_optimizers": True,
    },
    {
        "name": "p4_lr_3e-5_trav16384_cap32",
        "learning_rate": 3e-5,
        "traversals_per_player": 16384,
        "regret_train_steps": 48,
        "strategy_train_steps": 6,
        "sample_schedule": (32,),
        "reset_optimizers": True,
    },
    {
        "name": "p5_lr_3e-5_trav16384_cap32_strategy12",
        "learning_rate": 3e-5,
        "traversals_per_player": 16384,
        "regret_train_steps": 48,
        "strategy_train_steps": 12,
        "sample_schedule": (32,),
        "reset_optimizers": True,
    },
]

TRAINER_KWARGS: dict[str, Any] = {
    "regret_hidden_sizes": (2048, 2048),
    "strategy_hidden_sizes": (512, 512),
    "seed": 7,
    "regret_buffer_capacity": 16_000_000,
    "strategy_buffer_capacity": 2_000_000,
    "learning_rate": 1e-3,
    "batch_size": 4096,
    "regret_train_steps": 24,
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
    "traverser_action_baseline": "none",
    "traverser_action_sample_mode": "random",
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
    if isinstance(obj, torch.device):
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
    return str(obj)


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


def mean(values: Any) -> float:
    if values is None:
        return float("nan")
    if isinstance(values, (int, float)):
        return float(values)
    values = list(values)
    return float(np.mean(values)) if values else float("nan")


def next_after(current_s: float, period_minutes: float) -> float:
    period_s = 60.0 * float(period_minutes)
    return (math.floor(current_s / period_s) + 1) * period_s


def optimizer_reset(trainer: DeepCFRPlusTrainer) -> None:
    for optimizer in [*trainer.regret_optimizers, *trainer.strategy_optimizers]:
        optimizer.state.clear()


def set_optimizer_lr(trainer: DeepCFRPlusTrainer, learning_rate: float) -> None:
    trainer.learning_rate = float(learning_rate)
    for optimizer in [*trainer.regret_optimizers, *trainer.strategy_optimizers]:
        for group in optimizer.param_groups:
            group["lr"] = float(learning_rate)


def apply_phase(
    trainer: DeepCFRPlusTrainer,
    phase: dict[str, Any],
    *,
    reset: bool,
) -> None:
    set_optimizer_lr(trainer, float(phase["learning_rate"]))
    trainer.regret_train_steps = int(phase["regret_train_steps"])
    trainer.strategy_train_steps = int(phase["strategy_train_steps"])
    trainer.traverser_action_sample_schedule = tuple(
        int(count) for count in phase["sample_schedule"]
    )
    trainer.traverser_action_sample_count = None
    trainer.traverser_action_sample_fraction = None
    if trainer._gpu_traverser is not None:
        trainer._gpu_traverser.traverser_action_sample_schedule = (
            trainer.traverser_action_sample_schedule
        )
        trainer._gpu_traverser.traverser_action_sample_count = None
        trainer._gpu_traverser.traverser_action_sample_fraction = None
    if reset:
        optimizer_reset(trainer)


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
    metadata: dict[str, Any],
) -> tuple[Path, float]:
    target_minute = int(round(measured_training_s / 60.0))
    snapshot_path = run_dir / "checkpoints" / f"checkpoint_{target_minute:04d}m.pt"
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    tmp = snapshot_path.with_suffix(snapshot_path.suffix + ".new")
    shutil.copy2(latest_checkpoint, tmp)
    os.replace(tmp, snapshot_path)
    write_json(snapshot_path.with_suffix(".json"), metadata)
    return snapshot_path, time.perf_counter() - start


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
                if isinstance(entry, dict)
                and key in entry
                and entry[key] is not None
            ]
            if values:
                out[f"{section}_{key}"] = float(np.mean(values))
    return out


def save_policy_pair(
    trainer: DeepCFRPlusTrainer,
    run_dir: Path,
    label: str,
    measured_training_s: float,
    phase_index: int,
) -> dict[str, Any]:
    snapshot_root = run_dir / "policy_snapshots" / label
    average_dir = snapshot_root / "average_policy"
    current_dir = snapshot_root / "current_policy"

    start = time.perf_counter()
    save_policy(trainer.average_policy(), str(average_dir))
    average_s = time.perf_counter() - start

    start = time.perf_counter()
    current_policy = NeuralRegretMatchingPolicy.from_models(
        trainer.spec,
        trainer.regret_nets,
        hidden_sizes=trainer.regret_hidden_sizes,
        device="cpu",
    )
    save_policy(current_policy, str(current_dir))
    current_s = time.perf_counter() - start

    metadata = {
        "snapshot_label": label,
        "measured_training_s": measured_training_s,
        "measured_training_min": measured_training_s / 60.0,
        "iteration": trainer.iteration,
        "phase_index": int(phase_index),
        "phase_name": PHASES[int(phase_index)]["name"],
        "average_policy_dir": str(average_dir),
        "current_policy_dir": str(current_dir),
        "current_policy_kind": current_policy.POLICY_KIND,
        "average_policy_save_s": average_s,
        "current_policy_save_s": current_s,
    }
    write_json(snapshot_root / "snapshot.json", metadata)
    return metadata


def br_trainer_kwargs(device: torch.device) -> dict[str, Any]:
    return {
        "state_hidden_sizes": (512, 512),
        "action_hidden_sizes": (128, 128),
        "embedding_dim": 256,
        "device": device,
        "replay_capacity": 1_000_000,
        "batch_size": 4096,
        "learning_rate": 1e-3,
        "train_steps": 100,
        "warmup_transitions": 20_000,
        "epsilon_start": 1.0,
        "epsilon_end": 0.05,
        "epsilon_decay_decisions": 500_000,
        "rollouts_per_action": 1,
        "fused_optimizer": device.type == "cuda",
        "seed": BR_SEED,
    }


def decision_eval(records: list[dict[str, Any]]) -> dict[str, Any]:
    target_s = 60.0 * BR_DECISION_MINUTES
    after = [
        record for record in records if float(record["measured_training_s"]) >= target_s
    ]
    return min(
        after or records,
        key=lambda record: abs(float(record["measured_training_s"]) - target_s),
    )


def run_monitor_br(
    *,
    policy_dir: Path,
    run_dir: Path,
    label: str,
    policy_kind: str,
    long_br: bool,
    device: torch.device,
) -> dict[str, Any]:
    minutes = BR_LONG_MINUTES if long_br else BR_DECISION_MINUTES
    out_dir = run_dir / "monitor_brs" / label / policy_kind
    result = run_best_responder(
        policy_dir,
        method="action_conditioned_fitted_return",
        minutes=float(minutes),
        trainer_kwargs=br_trainer_kwargs(device),
        episodes_per_role=BR_EPISODES_PER_ROLE,
        rollout_batch_size=BR_ROLLOUT_BATCH_SIZE,
        evaluate_every_minutes=BR_EVALUATE_EVERY_MINUTES,
        eval_episodes_per_role=BR_EVAL_EPISODES_PER_ROLE,
        run_dir=out_dir,
        debug=True,
    )
    selected = decision_eval(result.evaluation_records)
    best_p_first = max(float(row["p_first"]) for row in result.evaluation_records)
    best_p_second = max(float(row["p_second"]) for row in result.evaluation_records)
    best_p_first_lcb = max(
        float(row["p_first_lcb"]) for row in result.evaluation_records
    )
    best_p_second_lcb = max(
        float(row["p_second_lcb"]) for row in result.evaluation_records
    )
    return {
        "policy_kind": policy_kind,
        "policy_dir": str(policy_dir),
        "br_run_dir": str(out_dir),
        "br_total_target_min": float(minutes),
        "br_measured_training_min": float(result.measured_training_s) / 60.0,
        "decision_responder_min": float(selected["measured_training_s"]) / 60.0,
        "decision_p_first": float(selected["p_first"]),
        "decision_p_second": float(selected["p_second"]),
        "decision_estimate": float(selected["exploitability_estimate"]),
        "decision_lower_bound": float(selected["exploitability_lower_bound"]),
        "best_p_first": best_p_first,
        "best_p_second": best_p_second,
        "best_discovered_estimate": best_p_first + best_p_second - 1.0,
        "best_p_first_lcb": best_p_first_lcb,
        "best_p_second_lcb": best_p_second_lcb,
        "best_lower_bound": best_p_first_lcb + best_p_second_lcb - 1.0,
    }


def write_monitor_plots(run_dir: Path) -> None:
    monitor_path = run_dir / "monitor_summary.jsonl"
    if not monitor_path.exists():
        return
    rows = [
        json.loads(line)
        for line in monitor_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"[plot skipped] matplotlib import failed: {exc}", flush=True)
        return

    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2))
    for policy_kind in ("average", "current"):
        subset = [row for row in rows if row["policy_kind"] == policy_kind]
        subset.sort(key=lambda row: float(row["measured_training_min"]))
        if not subset:
            continue
        xs = [float(row["measured_training_min"]) for row in subset]
        ys = [float(row["decision_estimate"]) for row in subset]
        axes[0].plot(xs, ys, marker="o", label=f"{policy_kind} 2m BR")
    axes[0].set(
        title="Controller BR estimate",
        xlabel="CFR+ training minutes",
        ylabel="2-minute BR exploitability estimate",
    )
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    long_rows = [row for row in rows if float(row["br_total_target_min"]) >= 10.0]
    for row in long_rows[-8:]:
        eval_path = Path(row["br_run_dir"]) / "evaluations.jsonl"
        if not eval_path.exists():
            continue
        evals = [
            json.loads(line)
            for line in eval_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not evals:
            continue
        xs = [float(item["measured_training_s"]) / 60.0 for item in evals]
        ys = [float(item["exploitability_estimate"]) for item in evals]
        axes[1].plot(
            xs,
            ys,
            marker="o",
            label=f"{row['snapshot_label']} {row['policy_kind']}",
        )
    axes[1].set(
        title="Recent 10-minute responder curves",
        xlabel="Responder training minutes",
        ylabel="Discovered exploitability",
    )
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(plots_dir / "monitor_summary.png", dpi=160)
    plt.close(fig)


def make_run_dir(args: argparse.Namespace) -> Path:
    if args.resume is not None:
        return Path(args.resume).expanduser().resolve()
    if args.run_dir is not None:
        return Path(args.run_dir).expanduser().resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return (
        REPO_ROOT
        / "artifacts"
        / "cfr_plus_69_claim_adaptive"
        / f"{SPEC.to_short_str()}___{stamp}"
    ).resolve()


def initial_controller_state() -> dict[str, Any]:
    return {
        "phase_index": 0,
        "phase_started_s": 0.0,
        "completed_phase_resets": [],
        "best_average_estimate": None,
        "best_current_estimate": None,
        "monitor_history": [],
    }


def load_controller_state(path: Path) -> dict[str, Any]:
    if path.exists():
        state = read_json(path)
        state.setdefault("completed_phase_resets", [])
        state.setdefault("monitor_history", [])
        return state
    return initial_controller_state()


def maybe_trigger_phase(
    state: dict[str, Any],
    *,
    measured_training_s: float,
    events_path: Path,
) -> bool:
    phase_index = int(state["phase_index"])
    if phase_index >= len(PHASES) - 1:
        return False
    phase_age_s = measured_training_s - float(state["phase_started_s"])
    if phase_age_s < 60.0 * MIN_PHASE_MINUTES:
        return False

    history = list(state.get("monitor_history", []))
    if len(history) < PATIENCE_MONITORS:
        return False
    recent = history[-PATIENCE_MONITORS:]
    best_average = state.get("best_average_estimate")
    best_current = state.get("best_current_estimate")
    if best_average is None or best_current is None:
        return False

    tol = CONTROLLER_TOLERANCE
    recent_avg_not_bestish = all(
        float(row["average_decision_estimate"]) > float(best_average) + tol
        for row in recent
    )
    latest_current_not_bestish = (
        float(history[-1]["current_decision_estimate"]) > float(best_current) + tol
    )
    if not (recent_avg_not_bestish and latest_current_not_bestish):
        return False

    old_phase = PHASES[phase_index]
    state["phase_index"] = phase_index + 1
    state["phase_started_s"] = measured_training_s
    event = {
        "event": "controller_phase_advance",
        "utc": datetime.now(timezone.utc).isoformat(),
        "measured_training_s": measured_training_s,
        "measured_training_min": measured_training_s / 60.0,
        "old_phase_index": phase_index,
        "old_phase": old_phase,
        "new_phase_index": phase_index + 1,
        "new_phase": PHASES[phase_index + 1],
        "best_average_estimate": best_average,
        "best_current_estimate": best_current,
        "recent_average_estimates": [
            float(row["average_decision_estimate"]) for row in recent
        ],
        "latest_current_estimate": float(history[-1]["current_decision_estimate"]),
        "tolerance": tol,
    }
    append_jsonl(events_path, event)
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Adaptive long-run 69-claim neural CFR+ with periodic BR monitors."
    )
    parser.add_argument("--run-dir", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--max-hours", type=float, default=None)
    parser.add_argument("--seed", type=int, default=int(TRAINER_KWARGS["seed"]))
    parser.add_argument("--br-device", type=str, default="cuda")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help=(
            "Run the real 69-claim training setup with compressed monitor, "
            "checkpoint, and approximate-BR settings. This is for verifying "
            "the full control flow before a long tmux run."
        ),
    )
    return parser.parse_args()


def apply_smoke_test_overrides(args: argparse.Namespace) -> None:
    global MONITOR_EVERY_MINUTES
    global LONG_BR_EVERY_MINUTES
    global CHECKPOINT_EVERY_MINUTES
    global PRINT_EVERY_MINUTES
    global VALIDATE_EVERY_MINUTES
    global VALIDATION_RECORDS
    global MIN_PHASE_MINUTES
    global PATIENCE_MONITORS
    global BR_DECISION_MINUTES
    global BR_LONG_MINUTES
    global BR_EVALUATE_EVERY_MINUTES
    global BR_EPISODES_PER_ROLE
    global BR_ROLLOUT_BATCH_SIZE
    global BR_EVAL_EPISODES_PER_ROLE
    global BR_EVAL_ROLLOUT_BATCH_SIZE

    MONITOR_EVERY_MINUTES = 2.0
    LONG_BR_EVERY_MINUTES = 4.0
    CHECKPOINT_EVERY_MINUTES = 3.0
    PRINT_EVERY_MINUTES = 1.0
    VALIDATE_EVERY_MINUTES = 2.0
    VALIDATION_RECORDS = 512
    MIN_PHASE_MINUTES = 4.0
    PATIENCE_MONITORS = 2

    BR_DECISION_MINUTES = 0.25
    BR_LONG_MINUTES = 0.5
    BR_EVALUATE_EVERY_MINUTES = 0.25
    BR_EPISODES_PER_ROLE = 256
    BR_ROLLOUT_BATCH_SIZE = 256
    BR_EVAL_EPISODES_PER_ROLE = 4_096
    BR_EVAL_ROLLOUT_BATCH_SIZE = 1_024

    if args.max_hours is None:
        args.max_hours = 0.12


def main() -> None:
    args = parse_args()
    if args.smoke_test:
        apply_smoke_test_overrides(args)
    run_dir = make_run_dir(args)
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = run_dir / "manifest.json"
    training_path = run_dir / "training.jsonl"
    events_path = run_dir / "events.jsonl"
    monitor_path = run_dir / "monitor_summary.jsonl"
    state_path = run_dir / "controller_state.json"
    summary_path = run_dir / "summary.json"
    checkpoint_path = run_dir / "latest_checkpoint.pt"
    pause_path = run_dir / "PAUSE"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    br_device = torch.device(args.br_device if torch.cuda.is_available() else "cpu")

    if args.resume is not None:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Cannot resume without {checkpoint_path}")
        trainer = DeepCFRPlusTrainer.load_checkpoint(checkpoint_path, device=device)
        state = load_controller_state(state_path)
        measured_training_s = float(state.get("measured_training_s", 0.0))
        print(f"resumed {run_dir} at {measured_training_s / 60.0:.1f}m", flush=True)
    else:
        kwargs = dict(TRAINER_KWARGS)
        kwargs["device"] = device
        kwargs["seed"] = int(args.seed)
        trainer = DeepCFRPlusTrainer(SPEC, **kwargs)
        state = initial_controller_state()
        measured_training_s = 0.0
        manifest = {
            "run_type": "cfr_plus_69_claim_adaptive",
            "spec": SPEC.to_json(),
            "trainer_kwargs": kwargs,
            "phases": PHASES,
            "controller": {
                "monitor_every_minutes": MONITOR_EVERY_MINUTES,
                "long_br_every_minutes": LONG_BR_EVERY_MINUTES,
                "checkpoint_every_minutes": CHECKPOINT_EVERY_MINUTES,
                "tolerance": CONTROLLER_TOLERANCE,
                "min_phase_minutes": MIN_PHASE_MINUTES,
                "patience_monitors": PATIENCE_MONITORS,
                "smoke_test": bool(args.smoke_test),
            },
            "br": {
                "decision_minutes": BR_DECISION_MINUTES,
                "long_minutes": BR_LONG_MINUTES,
                "evaluate_every_minutes": BR_EVALUATE_EVERY_MINUTES,
                "episodes_per_role": BR_EPISODES_PER_ROLE,
                "eval_episodes_per_role": BR_EVAL_EPISODES_PER_ROLE,
                "seed": BR_SEED,
                "device": str(br_device),
            },
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }
        write_json(manifest_path, manifest)

    phase_index = int(state["phase_index"])
    apply_phase(trainer, PHASES[phase_index], reset=False)
    next_monitor_s = next_after(measured_training_s, MONITOR_EVERY_MINUTES)
    next_checkpoint_s = next_after(measured_training_s, CHECKPOINT_EVERY_MINUTES)
    next_print_s = next_after(measured_training_s, PRINT_EVERY_MINUTES)
    next_validate_s = next_after(measured_training_s, VALIDATE_EVERY_MINUTES)
    max_training_s = (
        float("inf")
        if args.max_hours is None
        else measured_training_s + 3600.0 * float(args.max_hours)
    )

    append_jsonl(
        events_path,
        {
            "event": "start_or_resume",
            "utc": datetime.now(timezone.utc).isoformat(),
            "run_dir": str(run_dir),
            "measured_training_s": measured_training_s,
            "iteration": trainer.iteration,
            "phase_index": phase_index,
        },
    )
    print("run_dir:", run_dir, flush=True)
    print("pause with:", f"touch {pause_path}", flush=True)

    status = "running"
    try:
        while measured_training_s < max_training_s:
            phase_index = int(state["phase_index"])
            phase = PHASES[phase_index]
            apply_phase(trainer, phase, reset=False)

            start = time.perf_counter()
            record = trainer.run_iteration(
                traversals_per_player=int(phase["traversals_per_player"])
            )
            iteration_s = time.perf_counter() - start
            measured_training_s += iteration_s
            state["measured_training_s"] = measured_training_s
            state["iteration"] = trainer.iteration

            validation = None
            if measured_training_s >= next_validate_s:
                validation = trainer.validation_metrics(max_records=VALIDATION_RECORDS)
                while next_validate_s <= measured_training_s:
                    next_validate_s += 60.0 * VALIDATE_EVERY_MINUTES

            action_sampling = record.get("action_sampling", {}) or {}
            timing = record.get("timing", {}) or {}
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
                    "sample_schedule": tuple(int(x) for x in phase["sample_schedule"]),
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

            if measured_training_s >= next_monitor_s:
                while next_monitor_s <= measured_training_s:
                    target_s = next_monitor_s
                    label = f"{int(round(target_s / 60.0)):04d}m"
                    is_long = (
                        int(round(target_s / 60.0))
                        % int(round(LONG_BR_EVERY_MINUTES))
                        == 0
                    )
                    snapshot = save_policy_pair(
                        trainer,
                        run_dir,
                        label,
                        measured_training_s,
                        phase_index,
                    )
                    append_jsonl(events_path, {"event": "policy_snapshot", **snapshot})

                    print(
                        f"[monitor] {label} long={is_long} "
                        f"train={measured_training_s / 60.0:.1f}m "
                        f"iter={trainer.iteration}",
                        flush=True,
                    )
                    average_br = run_monitor_br(
                        policy_dir=Path(snapshot["average_policy_dir"]),
                        run_dir=run_dir,
                        label=label,
                        policy_kind="average",
                        long_br=is_long,
                        device=br_device,
                    )
                    current_br = run_monitor_br(
                        policy_dir=Path(snapshot["current_policy_dir"]),
                        run_dir=run_dir,
                        label=label,
                        policy_kind="current",
                        long_br=is_long,
                        device=br_device,
                    )

                    for row in (average_br, current_br):
                        monitor_row = {
                            "utc": datetime.now(timezone.utc).isoformat(),
                            "snapshot_label": label,
                            "measured_training_s": measured_training_s,
                            "measured_training_min": measured_training_s / 60.0,
                            "iteration": trainer.iteration,
                            "phase_index": phase_index,
                            "phase_name": phase["name"],
                            **row,
                        }
                        append_jsonl(monitor_path, monitor_row)

                    avg_est = float(average_br["decision_estimate"])
                    cur_est = float(current_br["decision_estimate"])
                    best_avg = state.get("best_average_estimate")
                    best_cur = state.get("best_current_estimate")
                    if best_avg is None or avg_est < float(best_avg):
                        state["best_average_estimate"] = avg_est
                    if best_cur is None or cur_est < float(best_cur):
                        state["best_current_estimate"] = cur_est
                    state["monitor_history"].append(
                        {
                            "snapshot_label": label,
                            "measured_training_s": measured_training_s,
                            "measured_training_min": measured_training_s / 60.0,
                            "iteration": trainer.iteration,
                            "phase_index": phase_index,
                            "average_decision_estimate": avg_est,
                            "current_decision_estimate": cur_est,
                            "average_br_total_target_min": average_br[
                                "br_total_target_min"
                            ],
                            "current_br_total_target_min": current_br[
                                "br_total_target_min"
                            ],
                        }
                    )
                    did_advance = maybe_trigger_phase(
                        state,
                        measured_training_s=measured_training_s,
                        events_path=events_path,
                    )
                    if did_advance:
                        new_phase_index = int(state["phase_index"])
                        new_phase = PHASES[new_phase_index]
                        apply_phase(
                            trainer,
                            new_phase,
                            reset=bool(new_phase["reset_optimizers"]),
                        )
                        state["completed_phase_resets"] = sorted(
                            set(state.get("completed_phase_resets", []))
                            | ({new_phase_index} if new_phase["reset_optimizers"] else set())
                        )
                        print(
                            f"[phase advance] -> {new_phase_index} {new_phase['name']}",
                            flush=True,
                        )
                    write_json(state_path, state)
                    write_monitor_plots(run_dir)
                    next_monitor_s += 60.0 * MONITOR_EVERY_MINUTES

            if measured_training_s >= next_checkpoint_s:
                while next_checkpoint_s <= measured_training_s:
                    metadata = {
                        "updated_utc": datetime.now(timezone.utc).isoformat(),
                        "checkpoint_target_min": next_checkpoint_s / 60.0,
                        "measured_training_s": measured_training_s,
                        "measured_training_min": measured_training_s / 60.0,
                        "iteration": trainer.iteration,
                        "phase_index": int(state["phase_index"]),
                        "phase": PHASES[int(state["phase_index"])],
                    }
                    ckpt_s = atomic_checkpoint(trainer, checkpoint_path)
                    ckpt_path, copy_s = copy_checkpoint_snapshot(
                        checkpoint_path,
                        run_dir,
                        next_checkpoint_s,
                        metadata,
                    )
                    append_jsonl(
                        events_path,
                        {
                            "event": "checkpoint",
                            **metadata,
                            "checkpoint_path": str(checkpoint_path),
                            "checkpoint_snapshot_path": str(ckpt_path),
                            "checkpoint_s": ckpt_s,
                            "checkpoint_copy_s": copy_s,
                            "checkpoint_GiB": checkpoint_path.stat().st_size / 2**30,
                        },
                    )
                    write_json(state_path, state)
                    print(
                        f"[checkpoint] target={next_checkpoint_s/60:.0f}m "
                        f"actual={measured_training_s/60:.1f}m "
                        f"iter={trainer.iteration} path={ckpt_path.name}",
                        flush=True,
                    )
                    next_checkpoint_s += 60.0 * CHECKPOINT_EVERY_MINUTES

            if measured_training_s >= next_print_s:
                print(
                    f"train={measured_training_s / 60.0:7.1f}m "
                    f"iter={trainer.iteration:7d} "
                    f"phase={phase['name']} "
                    f"lr={float(phase['learning_rate']):.1e} "
                    f"it_s={iteration_s:.2f} "
                    f"trav={float(timing.get('traversal_s', float('nan'))):.2f}s "
                    f"regfit={float(timing.get('regret_training_s', float('nan'))):.2f}s "
                    f"strfit={float(timing.get('strategy_training_s', float('nan'))):.2f}s "
                    f"edges={float(action_sampling.get('claim_edge_fraction', float('nan'))):.3f} "
                    f"ess={float(action_sampling.get('regret_weight_ess_fraction', float('nan'))):.3f}",
                    flush=True,
                )
                while next_print_s <= measured_training_s:
                    next_print_s += 60.0 * PRINT_EVERY_MINUTES

            if pause_path.exists():
                status = "paused"
                print(f"[pause] detected {pause_path}", flush=True)
                break

        if status == "running" and measured_training_s >= max_training_s:
            status = "max_hours_reached"

    except KeyboardInterrupt:
        status = "interrupted"
        append_jsonl(
            events_path,
            {
                "event": "keyboard_interrupt",
                "utc": datetime.now(timezone.utc).isoformat(),
                "measured_training_s": measured_training_s,
                "iteration": trainer.iteration,
            },
        )
        print("Interrupted; saving resumable state before exit.", flush=True)

    finally:
        final_label = f"{int(round(measured_training_s / 60.0)):04d}m_final"
        try:
            snapshot = save_policy_pair(
                trainer,
                run_dir,
                final_label,
                measured_training_s,
                int(state["phase_index"]),
            )
            append_jsonl(events_path, {"event": "final_policy_snapshot", **snapshot})
        except Exception as exc:
            append_jsonl(
                events_path,
                {
                    "event": "final_policy_snapshot_failed",
                    "error": repr(exc),
                    "utc": datetime.now(timezone.utc).isoformat(),
                },
            )

        ckpt_s = atomic_checkpoint(trainer, checkpoint_path)
        state["measured_training_s"] = measured_training_s
        state["iteration"] = trainer.iteration
        write_json(state_path, state)
        summary = {
            "status": status,
            "run_dir": str(run_dir),
            "spec": SPEC.to_json(),
            "iteration": trainer.iteration,
            "measured_training_s": measured_training_s,
            "measured_training_min": measured_training_s / 60.0,
            "phase_index": int(state["phase_index"]),
            "phase": PHASES[int(state["phase_index"])],
            "best_average_estimate": state.get("best_average_estimate"),
            "best_current_estimate": state.get("best_current_estimate"),
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_GiB": checkpoint_path.stat().st_size / 2**30,
            "checkpoint_s": ckpt_s,
            "pause_file": str(pause_path),
            "updated_utc": datetime.now(timezone.utc).isoformat(),
        }
        write_json(summary_path, summary)
        append_jsonl(events_path, {"event": "finalize", **summary})
        write_monitor_plots(run_dir)
        print("summary:", json.dumps(summary, indent=2, default=json_default), flush=True)


if __name__ == "__main__":
    main()
