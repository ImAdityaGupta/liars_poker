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

ROUND_MINUTES = 60.0
MONITOR_EVERY_MINUTES = 15.0
LONG_BR_EVERY_MINUTES = 30.0
CHECKPOINT_EVERY_MINUTES = 60.0
PRINT_EVERY_MINUTES = 5.0
VALIDATE_EVERY_MINUTES = 30.0
VALIDATION_RECORDS = 4096

BR_DECISION_MINUTES = 2.0
BR_LONG_MINUTES = 10.0
BR_EVALUATE_EVERY_MINUTES = 1.0
BR_EPISODES_PER_ROLE = 4096
BR_ROLLOUT_BATCH_SIZE = 1024
BR_EVAL_EPISODES_PER_ROLE = 200_000
BR_SEED = 17

BRANCHES: list[dict[str, Any]] = [
    {
        "name": "cap24_only",
        "label": "cap24 only",
        "learning_rate": 1e-3,
        "traversals_per_player": 1024,
        "regret_train_steps": 24,
        "strategy_train_steps": 6,
        "sample_schedule": (24,),
        "reset_optimizers_at_start": False,
    },
    {
        "name": "lr_drop_only",
        "label": "LR drop only",
        "learning_rate": 3e-4,
        "traversals_per_player": 1024,
        "regret_train_steps": 24,
        "strategy_train_steps": 6,
        "sample_schedule": (16,),
        "reset_optimizers_at_start": True,
    },
    {
        "name": "combined_cap24_lr_drop_trav2048",
        "label": "cap24 + LR drop + 2048 traversals",
        "learning_rate": 3e-4,
        "traversals_per_player": 2048,
        "regret_train_steps": 24,
        "strategy_train_steps": 6,
        "sample_schedule": (24,),
        "reset_optimizers_at_start": True,
    },
]


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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


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


def apply_branch(
    trainer: DeepCFRPlusTrainer,
    branch: dict[str, Any],
    *,
    reset: bool,
) -> None:
    set_optimizer_lr(trainer, float(branch["learning_rate"]))
    trainer.regret_train_steps = int(branch["regret_train_steps"])
    trainer.strategy_train_steps = int(branch["strategy_train_steps"])
    trainer.traverser_action_sample_schedule = tuple(
        int(count) for count in branch["sample_schedule"]
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
    branch_dir: Path,
    branch: dict[str, Any],
    label: str,
    branch_training_s: float,
    source_training_s: float,
) -> dict[str, Any]:
    snapshot_root = branch_dir / "policy_snapshots" / label
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
        "branch": branch["name"],
        "branch_label": branch["label"],
        "branch_training_s": branch_training_s,
        "branch_training_min": branch_training_s / 60.0,
        "source_training_s": source_training_s,
        "source_training_min": source_training_s / 60.0,
        "total_equivalent_training_s": source_training_s + branch_training_s,
        "total_equivalent_training_min": (source_training_s + branch_training_s) / 60.0,
        "iteration": trainer.iteration,
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
    branch_dir: Path,
    snapshot_label: str,
    policy_kind: str,
    long_br: bool,
    device: torch.device,
) -> dict[str, Any]:
    minutes = BR_LONG_MINUTES if long_br else BR_DECISION_MINUTES
    out_dir = branch_dir / "monitor_brs" / snapshot_label / policy_kind
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


def load_branch_state(branch_dir: Path) -> dict[str, Any]:
    state_path = branch_dir / "branch_state.json"
    if state_path.exists():
        state = read_json(state_path)
    else:
        state = {
            "branch_training_s": 0.0,
            "rounds_completed": 0,
            "initialized": False,
            "next_monitor_s": 60.0 * MONITOR_EVERY_MINUTES,
            "next_checkpoint_s": 60.0 * CHECKPOINT_EVERY_MINUTES,
            "next_print_s": 60.0 * PRINT_EVERY_MINUTES,
            "next_validate_s": 60.0 * VALIDATE_EVERY_MINUTES,
        }
    state.setdefault("next_monitor_s", next_after(float(state["branch_training_s"]), MONITOR_EVERY_MINUTES))
    state.setdefault("next_checkpoint_s", next_after(float(state["branch_training_s"]), CHECKPOINT_EVERY_MINUTES))
    state.setdefault("next_print_s", next_after(float(state["branch_training_s"]), PRINT_EVERY_MINUTES))
    state.setdefault("next_validate_s", next_after(float(state["branch_training_s"]), VALIDATE_EVERY_MINUTES))
    return state


def load_branch_trainer(
    *,
    branch: dict[str, Any],
    branch_dir: Path,
    source_checkpoint: Path,
    device: torch.device,
    events_path: Path,
) -> tuple[DeepCFRPlusTrainer, dict[str, Any], bool]:
    checkpoint_path = branch_dir / "latest_checkpoint.pt"
    state = load_branch_state(branch_dir)
    first_initialization = not bool(state.get("initialized", False))
    if checkpoint_path.exists() and not first_initialization:
        trainer = DeepCFRPlusTrainer.load_checkpoint(checkpoint_path, device=device)
        apply_branch(trainer, branch, reset=False)
        return trainer, state, False

    trainer = DeepCFRPlusTrainer.load_checkpoint(source_checkpoint, device=device)
    reset = bool(branch["reset_optimizers_at_start"])
    apply_branch(trainer, branch, reset=reset)
    state["initialized"] = True
    state["source_iteration"] = int(trainer.iteration)
    append_jsonl(
        events_path,
        {
            "event": "branch_initialized",
            "utc": datetime.now(timezone.utc).isoformat(),
            "branch": branch["name"],
            "branch_label": branch["label"],
            "source_checkpoint": str(source_checkpoint),
            "source_iteration": int(trainer.iteration),
            "optimizer_reset": reset,
        },
    )
    return trainer, state, True


def write_branch_state(branch_dir: Path, state: dict[str, Any]) -> None:
    write_json(branch_dir / "branch_state.json", state)


def resolve_source_checkpoint(args: argparse.Namespace) -> tuple[Path, float]:
    if args.source_checkpoint is not None:
        path = Path(args.source_checkpoint).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        return path, 60.0 * float(args.source_minute)

    if args.source_run is not None:
        run_dir = Path(args.source_run).expanduser().resolve()
    else:
        root = REPO_ROOT / "artifacts" / "cfr_plus_69_claim_adaptive"
        runs = sorted(
            [path for path in root.glob(f"{SPEC.to_short_str()}___*") if path.is_dir()],
            key=lambda path: path.stat().st_mtime,
        )
        if not runs:
            raise FileNotFoundError(
                "No adaptive 69-claim runs found; pass --source-run or --source-checkpoint."
            )
        run_dir = runs[-1]

    target_min = int(round(float(args.source_minute)))
    candidates = [
        run_dir / "checkpoints" / f"checkpoint_{target_min:04d}m.pt",
        run_dir / "checkpoints" / f"checkpoint_{target_min:03d}m.pt",
        run_dir / "latest_checkpoint.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate, 60.0 * float(args.source_minute)

    checkpoint_dir = run_dir / "checkpoints"
    available = sorted(checkpoint_dir.glob("checkpoint_*m.pt"))
    names = ", ".join(path.name for path in available[:20])
    raise FileNotFoundError(
        f"No source checkpoint found near {args.source_minute}m in {run_dir}. "
        f"Available checkpoint files include: {names}"
    )


def make_run_dir(args: argparse.Namespace) -> Path:
    if args.resume is not None:
        return Path(args.resume).expanduser().resolve()
    if args.run_dir is not None:
        return Path(args.run_dir).expanduser().resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return (
        REPO_ROOT
        / "artifacts"
        / "cfr_plus_69_claim_branch_forks"
        / f"{SPEC.to_short_str()}___{stamp}"
    ).resolve()


def write_plots(run_dir: Path) -> None:
    rows: list[dict[str, Any]] = []
    for branch_dir in sorted((run_dir / "branches").glob("*")):
        rows.extend(read_jsonl(branch_dir / "monitor_summary.jsonl"))
    if not rows:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"[plot skipped] matplotlib import failed: {exc}", flush=True)
        return

    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    for branch in BRANCHES:
        for policy_kind, linestyle in (("average", "-"), ("current", "--")):
            subset = [
                row
                for row in rows
                if row["branch"] == branch["name"]
                and row["policy_kind"] == policy_kind
            ]
            subset.sort(key=lambda row: float(row["branch_training_min"]))
            if not subset:
                continue
            label = f"{branch['label']} {policy_kind}"
            xs_branch = [float(row["branch_training_min"]) for row in subset]
            xs_total = [float(row["total_equivalent_training_min"]) for row in subset]
            ys = [float(row["decision_estimate"]) for row in subset]
            axes[0, 0].plot(xs_branch, ys, marker="o", linestyle=linestyle, label=label)
            axes[0, 1].plot(xs_total, ys, marker="o", linestyle=linestyle, label=label)

    axes[0, 0].set(
        title="Branch monitor BR estimate",
        xlabel="Branch training minutes",
        ylabel="2-minute BR exploitability estimate",
    )
    axes[0, 1].set(
        title="Equivalent total training time",
        xlabel="Source + branch training minutes",
        ylabel="2-minute BR exploitability estimate",
    )

    long_rows = [row for row in rows if float(row["br_total_target_min"]) >= 10.0]
    for row in long_rows[-12:]:
        eval_path = Path(row["br_run_dir"]) / "evaluations.jsonl"
        evals = read_jsonl(eval_path)
        if not evals:
            continue
        xs = [float(item["measured_training_s"]) / 60.0 for item in evals]
        ys = [float(item["exploitability_estimate"]) for item in evals]
        axes[1, 0].plot(
            xs,
            ys,
            marker="o",
            label=f"{row['branch_label']} {row['snapshot_label']} {row['policy_kind']}",
        )

    axes[1, 0].set(
        title="Recent 10-minute responder curves",
        xlabel="Responder training minutes",
        ylabel="Discovered exploitability",
    )

    for branch in BRANCHES:
        training_rows = read_jsonl(
            run_dir / "branches" / branch["name"] / "training.jsonl"
        )
        if not training_rows:
            continue
        xs = [float(row["branch_training_min"]) for row in training_rows]
        ys = [float(row["iteration_s"]) for row in training_rows]
        axes[1, 1].plot(xs, ys, alpha=0.75, label=branch["label"])
    axes[1, 1].set(
        title="Iteration cost",
        xlabel="Branch training minutes",
        ylabel="Seconds",
    )

    for ax in axes.ravel():
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(plots_dir / "branch_fork_summary.png", dpi=160)
    plt.close(fig)


def train_branch_round(
    *,
    run_dir: Path,
    branch: dict[str, Any],
    source_checkpoint: Path,
    source_training_s: float,
    device: torch.device,
    br_device: torch.device,
    pause_path: Path,
) -> bool:
    branch_dir = run_dir / "branches" / branch["name"]
    branch_dir.mkdir(parents=True, exist_ok=True)
    events_path = branch_dir / "events.jsonl"
    training_path = branch_dir / "training.jsonl"
    monitor_path = branch_dir / "monitor_summary.jsonl"
    checkpoint_path = branch_dir / "latest_checkpoint.pt"

    trainer, state, initialized_now = load_branch_trainer(
        branch=branch,
        branch_dir=branch_dir,
        source_checkpoint=source_checkpoint,
        device=device,
        events_path=events_path,
    )
    branch_training_s = float(state["branch_training_s"])
    round_start_s = branch_training_s
    round_target_s = round_start_s + 60.0 * ROUND_MINUTES

    append_jsonl(
        events_path,
        {
            "event": "round_start",
            "utc": datetime.now(timezone.utc).isoformat(),
            "branch": branch["name"],
            "branch_label": branch["label"],
            "branch_training_s": branch_training_s,
            "branch_training_min": branch_training_s / 60.0,
            "round_target_s": round_target_s,
            "round_target_min": round_target_s / 60.0,
            "initialized_now": initialized_now,
        },
    )
    print(
        f"\n=== {branch['label']} === "
        f"branch={branch_training_s/60.0:.1f}m -> {round_target_s/60.0:.1f}m",
        flush=True,
    )

    status = "round_complete"
    try:
        while branch_training_s < round_target_s:
            if pause_path.exists():
                status = "paused"
                print(f"[pause] detected {pause_path}", flush=True)
                break

            apply_branch(trainer, branch, reset=False)
            start = time.perf_counter()
            record = trainer.run_iteration(
                traversals_per_player=int(branch["traversals_per_player"])
            )
            iteration_s = time.perf_counter() - start
            branch_training_s += iteration_s
            total_s = source_training_s + branch_training_s

            validation = None
            if branch_training_s >= float(state["next_validate_s"]):
                validation = trainer.validation_metrics(max_records=VALIDATION_RECORDS)
                while float(state["next_validate_s"]) <= branch_training_s:
                    state["next_validate_s"] = float(state["next_validate_s"]) + 60.0 * VALIDATE_EVERY_MINUTES

            action_sampling = record.get("action_sampling", {}) or {}
            record.update(
                {
                    "utc": datetime.now(timezone.utc).isoformat(),
                    "branch": branch["name"],
                    "branch_label": branch["label"],
                    "branch_training_s": branch_training_s,
                    "branch_training_min": branch_training_s / 60.0,
                    "source_training_s": source_training_s,
                    "source_training_min": source_training_s / 60.0,
                    "total_equivalent_training_s": total_s,
                    "total_equivalent_training_min": total_s / 60.0,
                    "iteration_s": iteration_s,
                    "learning_rate": float(branch["learning_rate"]),
                    "traversals_per_player": int(branch["traversals_per_player"]),
                    "regret_train_steps": int(branch["regret_train_steps"]),
                    "strategy_train_steps": int(branch["strategy_train_steps"]),
                    "sample_schedule": tuple(int(x) for x in branch["sample_schedule"]),
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

            if branch_training_s >= float(state["next_monitor_s"]):
                while float(state["next_monitor_s"]) <= branch_training_s:
                    target_s = float(state["next_monitor_s"])
                    label = f"branch_{int(round(target_s / 60.0)):04d}m"
                    is_long = (
                        int(round(target_s / 60.0))
                        % int(round(LONG_BR_EVERY_MINUTES))
                        == 0
                    )
                    snapshot = save_policy_pair(
                        trainer,
                        branch_dir,
                        branch,
                        label,
                        branch_training_s,
                        source_training_s,
                    )
                    append_jsonl(events_path, {"event": "policy_snapshot", **snapshot})
                    print(
                        f"[monitor] {branch['label']} {label} long={is_long} "
                        f"branch={branch_training_s/60.0:.1f}m iter={trainer.iteration}",
                        flush=True,
                    )

                    average_br = run_monitor_br(
                        policy_dir=Path(snapshot["average_policy_dir"]),
                        branch_dir=branch_dir,
                        snapshot_label=label,
                        policy_kind="average",
                        long_br=is_long,
                        device=br_device,
                    )
                    current_br = run_monitor_br(
                        policy_dir=Path(snapshot["current_policy_dir"]),
                        branch_dir=branch_dir,
                        snapshot_label=label,
                        policy_kind="current",
                        long_br=is_long,
                        device=br_device,
                    )
                    for row in (average_br, current_br):
                        monitor_row = {
                            "utc": datetime.now(timezone.utc).isoformat(),
                            "snapshot_label": label,
                            "branch": branch["name"],
                            "branch_label": branch["label"],
                            "branch_training_s": branch_training_s,
                            "branch_training_min": branch_training_s / 60.0,
                            "source_training_s": source_training_s,
                            "source_training_min": source_training_s / 60.0,
                            "total_equivalent_training_s": source_training_s
                            + branch_training_s,
                            "total_equivalent_training_min": (
                                source_training_s + branch_training_s
                            )
                            / 60.0,
                            "iteration": trainer.iteration,
                            **row,
                        }
                        append_jsonl(monitor_path, monitor_row)

                    monitor_ckpt_s = atomic_checkpoint(trainer, checkpoint_path)
                    append_jsonl(
                        events_path,
                        {
                            "event": "monitor_checkpoint",
                            "utc": datetime.now(timezone.utc).isoformat(),
                            "snapshot_label": label,
                            "branch_training_s": branch_training_s,
                            "branch_training_min": branch_training_s / 60.0,
                            "iteration": trainer.iteration,
                            "checkpoint_path": str(checkpoint_path),
                            "checkpoint_s": monitor_ckpt_s,
                            "checkpoint_GiB": checkpoint_path.stat().st_size / 2**30,
                        },
                    )
                    write_plots(run_dir)
                    state["next_monitor_s"] = float(state["next_monitor_s"]) + 60.0 * MONITOR_EVERY_MINUTES

            if branch_training_s >= float(state["next_checkpoint_s"]):
                while float(state["next_checkpoint_s"]) <= branch_training_s:
                    target_s = float(state["next_checkpoint_s"])
                    ckpt_s = atomic_checkpoint(trainer, checkpoint_path)
                    archive = (
                        branch_dir
                        / "checkpoints"
                        / f"checkpoint_branch_{int(round(target_s / 60.0)):04d}m.pt"
                    )
                    archive.parent.mkdir(parents=True, exist_ok=True)
                    tmp_archive = archive.with_suffix(archive.suffix + ".new")
                    copy_start = time.perf_counter()
                    shutil.copy2(checkpoint_path, tmp_archive)
                    os.replace(tmp_archive, archive)
                    copy_s = time.perf_counter() - copy_start
                    append_jsonl(
                        events_path,
                        {
                            "event": "checkpoint",
                            "utc": datetime.now(timezone.utc).isoformat(),
                            "branch_training_s": branch_training_s,
                            "branch_training_min": branch_training_s / 60.0,
                            "checkpoint_target_min": target_s / 60.0,
                            "iteration": trainer.iteration,
                            "checkpoint_path": str(checkpoint_path),
                            "checkpoint_archive_path": str(archive),
                            "checkpoint_s": ckpt_s,
                            "checkpoint_copy_s": copy_s,
                            "checkpoint_GiB": checkpoint_path.stat().st_size / 2**30,
                        },
                    )
                    state["next_checkpoint_s"] = target_s + 60.0 * CHECKPOINT_EVERY_MINUTES

            if branch_training_s >= float(state["next_print_s"]):
                timing = record.get("timing", {}) or {}
                print(
                    f"{branch['label']}: "
                    f"branch={branch_training_s / 60.0:7.1f}m "
                    f"iter={trainer.iteration:7d} "
                    f"it_s={iteration_s:.2f} "
                    f"trav={float(timing.get('traversal_s', float('nan'))):.2f}s "
                    f"regfit={float(timing.get('regret_training_s', float('nan'))):.2f}s "
                    f"strfit={float(timing.get('strategy_training_s', float('nan'))):.2f}s "
                    f"edges={float(action_sampling.get('claim_edge_fraction', float('nan'))):.3f} "
                    f"ess={float(action_sampling.get('regret_weight_ess_fraction', float('nan'))):.3f}",
                    flush=True,
                )
                while float(state["next_print_s"]) <= branch_training_s:
                    state["next_print_s"] = float(state["next_print_s"]) + 60.0 * PRINT_EVERY_MINUTES

            state["branch_training_s"] = branch_training_s
            state["iteration"] = int(trainer.iteration)
            write_branch_state(branch_dir, state)

    except KeyboardInterrupt:
        status = "interrupted"
        raise
    finally:
        state["branch_training_s"] = branch_training_s
        state["iteration"] = int(trainer.iteration)
        if status == "round_complete":
            state["rounds_completed"] = int(state.get("rounds_completed", 0)) + 1
        ckpt_s = atomic_checkpoint(trainer, checkpoint_path)
        write_branch_state(branch_dir, state)
        append_jsonl(
            events_path,
            {
                "event": "round_end",
                "utc": datetime.now(timezone.utc).isoformat(),
                "status": status,
                "branch": branch["name"],
                "branch_label": branch["label"],
                "branch_training_s": branch_training_s,
                "branch_training_min": branch_training_s / 60.0,
                "iteration": trainer.iteration,
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_s": ckpt_s,
                "checkpoint_GiB": checkpoint_path.stat().st_size / 2**30,
            },
        )
        del trainer
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return status != "paused"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rotate 69-claim CFR+ branch continuations from a common checkpoint. "
            "The script runs indefinitely until PAUSE is touched or --max-rounds is reached."
        )
    )
    parser.add_argument("--source-run", type=str, default=None)
    parser.add_argument("--source-checkpoint", type=str, default=None)
    parser.add_argument("--source-minute", type=float, default=120.0)
    parser.add_argument("--run-dir", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--max-rounds", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--br-device", type=str, default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() and args.device == "cuda":
        raise RuntimeError("CUDA is not available.")
    device = torch.device(args.device)
    br_device = torch.device(args.br_device)
    torch.set_float32_matmul_precision("high")

    run_dir = make_run_dir(args)
    run_dir.mkdir(parents=True, exist_ok=True)
    pause_path = run_dir / "PAUSE"
    events_path = run_dir / "events.jsonl"
    summary_path = run_dir / "summary.json"
    manifest_path = run_dir / "manifest.json"

    if args.resume is not None and manifest_path.exists():
        manifest = read_json(manifest_path)
        source_checkpoint = Path(manifest["source_checkpoint"])
        source_training_s = float(manifest["source_training_s"])
    else:
        source_checkpoint, source_training_s = resolve_source_checkpoint(args)
        manifest = {
            "run_type": "cfr_plus_69_claim_branch_forks",
            "spec": SPEC.to_json(),
            "source_checkpoint": str(source_checkpoint),
            "source_training_s": source_training_s,
            "source_training_min": source_training_s / 60.0,
            "branches": BRANCHES,
            "round_minutes": ROUND_MINUTES,
            "monitor_every_minutes": MONITOR_EVERY_MINUTES,
            "long_br_every_minutes": LONG_BR_EVERY_MINUTES,
            "checkpoint_every_minutes": CHECKPOINT_EVERY_MINUTES,
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

    append_jsonl(
        events_path,
        {
            "event": "start_or_resume",
            "utc": datetime.now(timezone.utc).isoformat(),
            "run_dir": str(run_dir),
            "source_checkpoint": str(source_checkpoint),
            "source_training_min": source_training_s / 60.0,
        },
    )
    print("run_dir:", run_dir, flush=True)
    print("source_checkpoint:", source_checkpoint, flush=True)
    print("source_training_min:", source_training_s / 60.0, flush=True)
    print("pause with:", f"touch {pause_path}", flush=True)

    rounds_completed = 0
    status = "running"
    try:
        while args.max_rounds is None or rounds_completed < int(args.max_rounds):
            for branch in BRANCHES:
                if pause_path.exists():
                    status = "paused"
                    break
                keep_going = train_branch_round(
                    run_dir=run_dir,
                    branch=branch,
                    source_checkpoint=source_checkpoint,
                    source_training_s=source_training_s,
                    device=device,
                    br_device=br_device,
                    pause_path=pause_path,
                )
                write_plots(run_dir)
                if not keep_going:
                    status = "paused"
                    break
            if status == "paused":
                break
            rounds_completed += 1
            append_jsonl(
                events_path,
                {
                    "event": "rotation_round_complete",
                    "utc": datetime.now(timezone.utc).isoformat(),
                    "rounds_completed": rounds_completed,
                },
            )

        if status == "running" and args.max_rounds is not None:
            status = "max_rounds_reached"
    except KeyboardInterrupt:
        status = "interrupted"
        append_jsonl(
            events_path,
            {"event": "keyboard_interrupt", "utc": datetime.now(timezone.utc).isoformat()},
        )
        print("Interrupted. Branch state/checkpoints are saved after each branch round/monitor.", flush=True)
    finally:
        branch_states = {}
        for branch in BRANCHES:
            state_path = run_dir / "branches" / branch["name"] / "branch_state.json"
            if state_path.exists():
                branch_states[branch["name"]] = read_json(state_path)
        summary = {
            "status": status,
            "run_dir": str(run_dir),
            "source_checkpoint": str(source_checkpoint),
            "source_training_s": source_training_s,
            "source_training_min": source_training_s / 60.0,
            "rounds_completed": rounds_completed,
            "branches": BRANCHES,
            "branch_states": branch_states,
            "pause_file": str(pause_path),
            "updated_utc": datetime.now(timezone.utc).isoformat(),
        }
        write_json(summary_path, summary)
        append_jsonl(events_path, {"event": "finalize", **summary})
        write_plots(run_dir)
        print("summary:", json.dumps(summary, indent=2, default=json_default), flush=True)


if __name__ == "__main__":
    main()
