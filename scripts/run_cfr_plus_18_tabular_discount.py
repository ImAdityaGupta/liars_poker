#!/usr/bin/env python3
"""Run one or all six sampled tabular regret rules on the 18-claim game.

Each arm starts from seed 17 and a zero regret table. Measured training time
excludes policy evaluation and checkpoint writing. Nothing runs on import.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
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

from liars_poker.algo.br_exact_dense_to_dense import best_response_dense
from liars_poker.algo.cfr_discount_tabular import TabularDiscountTrainer
from liars_poker.core import GameSpec
from liars_poker.policies.neural import compile_neural_to_dense
from liars_poker.serialization import save_policy


SPEC = GameSpec(ranks=4, suits=4, hand_size=2,
                claim_kinds=("RankHigh", "Pair", "TwoPair", "Trips"),
                suit_symmetry=True)
ARMS = {
    "V_cfr_uniform": ("cfr", "uniform"),
    "A_cfr_plus_linear": ("cfr_plus", "linear"),
    "B_cfr_plus_quadratic": ("cfr_plus", "quadratic"),
    "C_dcfr_plus_quadratic": ("dcfr_plus", "quadratic"),
    "D_dcfr_exact_quadratic": ("dcfr_exact", "quadratic"),
    "E_dcfr_visited_quadratic": ("dcfr_visited", "quadratic"),
}
SEED = 17
TRAVERSALS = 4096
MONITOR_MINUTES = 15
CHECKPOINT_MINUTES = 15


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def append_jsonl(path: Path, value: dict) -> None:
    with path.open("a", encoding="utf-8") as out:
        out.write(json.dumps(value) + "\n")
        out.flush()


def make_trainer(arm: str, *,
                 trainer_type: type[TabularDiscountTrainer] = TabularDiscountTrainer,
                 use_neural_regret: bool = True,
                 use_neural_strategy: bool = True,
                 average_weight_power: int | None = None,
                 ) -> TabularDiscountTrainer:
    rule, weighting = ARMS[arm]
    trainer_kwargs = {}
    if average_weight_power is not None:
        trainer_kwargs["average_weight_power"] = average_weight_power
    trainer = trainer_type(
        SPEC, device="cpu", seed=SEED, update_rule=rule,
        regret_hidden_sizes=((512, 512) if use_neural_regret else ()),
        strategy_hidden_sizes=((256, 256) if use_neural_strategy else ()),
        learning_rate=1e-3, batch_size=1024,
        regret_buffer_capacity=4_000_000,
        strategy_buffer_capacity=(2_000_000 if use_neural_strategy else 0),
        regret_train_steps=0,
        strategy_train_steps=(6 if use_neural_strategy else 0),
        use_regret_network=use_neural_regret,
        use_strategy_network=use_neural_strategy,
        strategy_weighting=weighting,
        regret_target_mode="aggregate_then_clip",
        regret_increment_reach_mode="none",
        regret_accumulation_mode="cumulative",
        traversal_backend="gpu_native", traversal_batch_size=512,
        device_replay=True, fused_optimizer=False,
        validation_fraction=0.0,
        **trainer_kwargs,
    )
    trainer.activate_regret_table()
    return trainer


def save_progress(trainer: TabularDiscountTrainer, run_dir: Path,
                  measured_s: float, next_monitor_s: float,
                  pending_evaluation: str | None = None, *,
                  next_snapshot_s: float | None = None,
                  pending_snapshot: str | None = None) -> None:
    checkpoint = run_dir / "latest_checkpoint.pt"
    old_size = checkpoint.stat().st_size if checkpoint.exists() else 0
    if shutil.disk_usage(run_dir).free < old_size + 2 * 1024**3:
        raise OSError(f"Not enough free disk for an atomic checkpoint in {run_dir}")
    tmp = run_dir / "latest_checkpoint.pt.tmp"
    progress = {
        "iteration": trainer.iteration, "measured_training_s": measured_s,
        "next_monitor_s": next_monitor_s,
        "pending_evaluation": pending_evaluation,
        "updated_utc": utc(),
    }
    if next_snapshot_s is not None:
        progress["next_snapshot_s"] = next_snapshot_s
        progress["pending_snapshot"] = pending_snapshot
    # The progress cursor and trainer state are one atomic commit. state.json
    # is a convenient view only; a restart reads the checkpoint's cursor.
    payload = trainer.checkpoint_dict()
    payload["experiment_progress"] = progress
    torch.save(payload, tmp)
    os.replace(tmp, checkpoint)
    atomic_json(run_dir / "state.json", progress)


def read_evals(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def plot_comparison(root: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for arm in ARMS:
        rows = read_evals(root / arm / "evaluations.jsonl")
        if not rows:
            continue
        for ax, key, xlabel in zip(
            axes, ("measured_training_min", "iteration"),
            ("Measured training minutes", "CFR iteration"),
        ):
            ax.plot([r[key] for r in rows], [r["exploitability"] for r in rows],
                    marker="o", markersize=3, label=arm)
            ax.set(xlabel=xlabel, ylabel="Exact exploitability of learned average")
            ax.set_yscale("log")
            ax.grid(alpha=0.25)
    axes[1].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    # Independent arms may evaluate at the same time. Give each writer its
    # own temporary file, then atomically replace the shared final image.
    tmp = root / f"comparison.{os.getpid()}.tmp.png"
    fig.savefig(tmp, dpi=160)
    plt.close(fig)
    os.replace(tmp, root / "comparison.png")


def evaluate(trainer: TabularDiscountTrainer, root: Path,
             run_dir: Path, label: str, measured_s: float) -> None:
    policy_dir = run_dir / "policy_snapshots" / label / "average_policy"
    policy = trainer.average_policy()
    save_policy(policy, str(policy_dir))
    dense = compile_neural_to_dense(policy, batch_size=65_536)
    _, meta = best_response_dense(trainer.spec, dense, store_state_values=False)
    p_first, p_second = meta["computer"].exploitability()
    row = {
        "utc": utc(), "arm": run_dir.name, "snapshot": label,
        "iteration": trainer.iteration,
        "measured_training_min": measured_s / 60,
        "p_first": float(p_first), "p_second": float(p_second),
        "exploitability": float(p_first + p_second - 1),
        "policy_dir": str(policy_dir),
    }
    append_jsonl(run_dir / "evaluations.jsonl", row)
    print(f"[eval] {run_dir.name} {label} iter={trainer.iteration} "
          f"exploitability={row['exploitability']:.6f}", flush=True)
    plot_comparison(root)


def run_arm(root: Path, arm: str, target_minutes: float, threads: int,
            stop: list[bool]) -> None:
    run_dir = root / arm
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.json"
    checkpoint = run_dir / "latest_checkpoint.pt"
    expected = {
        "arm": arm, "update_rule": ARMS[arm][0],
        "strategy_weighting": ARMS[arm][1],
        "seed": SEED, "traversals_per_player": TRAVERSALS,
        "monitor_minutes": MONITOR_MINUTES,
        "checkpoint_minutes": CHECKPOINT_MINUTES,
        "regret_units": "cumulative conditional advantage; no /t",
        "sampled_reach": "one update if visited; none if unvisited",
        "average_policy": "neural strategy network",
    }
    torch.set_num_threads(threads)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if any(manifest.get(k) != v for k, v in expected.items()):
            raise ValueError(f"Manifest mismatch for {arm}; refusing to resume")
        if not checkpoint.exists():
            if (run_dir / "training.jsonl").exists() or (run_dir / "evaluations.jsonl").exists():
                raise FileNotFoundError(f"Cannot resume {arm} without checkpoint")
            # A crash between writing the manifest and first checkpoint has
            # no training to recover, so starting this arm again is safe.
            trainer = make_trainer(arm)
            measured_s = 0.0
            next_monitor_s = MONITOR_MINUTES * 60
            save_progress(trainer, run_dir, measured_s, next_monitor_s, "0000m")
            state = {"pending_evaluation": "0000m"}
        else:
            trainer = TabularDiscountTrainer.load_fork_checkpoint(checkpoint)
            state = trainer._experiment_progress
            if state is None:
                raise ValueError(f"Checkpoint has no experiment cursor for {arm}")
            if (trainer.iteration != state["iteration"] or trainer.update_rule != ARMS[arm][0]
                    or trainer.strategy_weighting != ARMS[arm][1]):
                raise ValueError(f"Checkpoint, state and manifest disagree for {arm}")
            measured_s = float(state["measured_training_s"])
            next_monitor_s = float(state["next_monitor_s"])
            print(f"[resume] {arm} iter={trainer.iteration} train={measured_s/60:.1f}m",
                  flush=True)
    else:
        if checkpoint.exists() or (run_dir / "state.json").exists():
            raise ValueError(f"Incomplete existing run for {arm}; inspect it first")
        trainer = make_trainer(arm)
        measured_s = 0.0
        next_monitor_s = MONITOR_MINUTES * 60
        atomic_json(manifest_path, {**expected, "created_utc": utc()})
        save_progress(trainer, run_dir, measured_s, next_monitor_s, "0000m")
        state = {"pending_evaluation": "0000m"}

    pending = state["pending_evaluation"]
    if pending is not None:
        already_done = any(r["snapshot"] == pending for r in
                           read_evals(run_dir / "evaluations.jsonl"))
        if not already_done:
            evaluate(trainer, root, run_dir, pending, measured_s)
        save_progress(trainer, run_dir, measured_s, next_monitor_s)

    target_s = target_minutes * 60
    while measured_s < target_s and not stop[0] and not (root / "PAUSE").exists():
        start = time.perf_counter()
        row = trainer.run_iteration(traversals_per_player=TRAVERSALS)
        iteration_s = time.perf_counter() - start
        measured_s += iteration_s
        append_jsonl(run_dir / "training.jsonl", {
            "utc": utc(), "iteration": trainer.iteration,
            "measured_training_min": measured_s / 60,
            "iteration_s": iteration_s, "timing": row["timing"],
            "regret_records": row["new_regret_records"],
        })
        if trainer.iteration % 100 == 0:
            print(f"[train] {arm} {measured_s/60:.1f}m iter={trainer.iteration} "
                  f"iteration_s={iteration_s:.2f}", flush=True)
        if measured_s >= next_monitor_s:
            label = f"{int(round(next_monitor_s/60)):04d}m"
            next_monitor_s += MONITOR_MINUTES * 60
            save_progress(trainer, run_dir, measured_s, next_monitor_s, label)
            evaluate(trainer, root, run_dir, label, measured_s)
            save_progress(trainer, run_dir, measured_s, next_monitor_s)

    save_progress(trainer, run_dir, measured_s, next_monitor_s)
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
    parser.add_argument("--arm", choices=[*ARMS, "all"], default="all")
    parser.add_argument("--minutes-per-arm", type=float, default=180)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if args.minutes_per_arm <= 0 or args.threads <= 0:
        parser.error("Minutes and threads must be positive")
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    stop = [False]

    def request_stop(signum, _frame):
        stop[0] = True
        print(f"[signal] {signum}; checkpoint after current iteration", flush=True)

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    for arm in (ARMS if args.arm == "all" else (args.arm,)):
        if stop[0] or (root / "PAUSE").exists():
            break
        run_arm(root, arm, args.minutes_per_arm, args.threads, stop)


if __name__ == "__main__":
    main()
