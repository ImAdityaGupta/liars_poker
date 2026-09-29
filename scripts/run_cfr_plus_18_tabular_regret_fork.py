#!/usr/bin/env python3
"""Continue an 18-claim neural CFR+ checkpoint with a fast tabular regret state.

The strategy network and its replay reservoir continue from the checkpoint.
Regrets are copied from the frozen network on first access, then updated only
at visited infosets. A rolling checkpoint and exact average-policy evaluations
make this continuation resumable and comparable with the neural run.
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
from liars_poker.algo.cfr_plus_tabular_fork import TabularRegretFork
from liars_poker.policies.neural import compile_neural_to_dense
from liars_poker.serialization import save_policy


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


def save_checkpoint(trainer: TabularRegretFork, path: Path, state_path: Path,
                    measured_s: float, next_monitor_s: float) -> None:
    old_size = path.stat().st_size if path.exists() else 0
    if shutil.disk_usage(path.parent).free < old_size + 2 * 1024**3:
        raise OSError("Insufficient free disk for atomic fork checkpoint")
    tmp = path.with_name(path.name + ".tmp")
    trainer.save_checkpoint(tmp)
    os.replace(tmp, path)
    atomic_json(state_path, {
        "iteration": trainer.iteration,
        "measured_fork_s": measured_s,
        "next_monitor_s": next_monitor_s,
        "checkpoint_utc": utc(),
    })


def evaluate_average(trainer: TabularRegretFork, output_dir: Path,
                     label: str, measured_s: float) -> dict:
    start = time.perf_counter()
    policy_dir = output_dir / "policy_snapshots" / label / "average_policy"
    save_policy(trainer.average_policy(), str(policy_dir))
    dense = compile_neural_to_dense(trainer.average_policy(), batch_size=65_536)
    _, meta = best_response_dense(trainer.spec, dense, store_state_values=False)
    p_first, p_second = meta["computer"].exploitability()
    row = {
        "utc": utc(),
        "label": label,
        "iteration": trainer.iteration,
        "measured_fork_min": measured_s / 60,
        "p_first": float(p_first),
        "p_second": float(p_second),
        "exploitability": float(p_first + p_second - 1),
        "evaluation_s": time.perf_counter() - start,
        "policy_dir": str(policy_dir),
    }
    append_jsonl(output_dir / "evaluations.jsonl", row)
    print(f"[eval] fork={measured_s / 60:.1f}m iter={trainer.iteration} "
          f"exploitability={row['exploitability']:.6f}", flush=True)
    plot_evaluations(output_dir)
    return row


def plot_evaluations(output_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    for line in (output_dir / "evaluations.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, field, xlabel in zip(axes, ("measured_fork_min", "iteration"),
                                  ("Additional measured training minutes", "Total CFR+ iteration")):
        ax.plot([r[field] for r in rows], [r["exploitability"] for r in rows],
                color="#176B9A", marker="o")
        ax.set(xlabel=xlabel, ylabel="Exact average-policy exploitability")
        ax.set_yscale("log")
        ax.grid(alpha=0.25)
    fig.tight_layout()
    tmp = output_dir / "average_exploitability.tmp.png"
    fig.savefig(tmp, dpi=150)
    plt.close(fig)
    os.replace(tmp, output_dir / "average_exploitability.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", type=Path,
                        help="Neural checkpoint to fork; required for a new run")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--additional-minutes", type=float, default=120)
    parser.add_argument("--monitor-minutes", type=float, default=15)
    parser.add_argument("--checkpoint-minutes", type=float, default=15)
    parser.add_argument("--traversals", type=int, default=4096)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if min(args.additional_minutes, args.monitor_minutes, args.checkpoint_minutes,
           args.traversals, args.threads) <= 0:
        parser.error("All budgets and thread counts must be positive")
    if args.monitor_minutes != args.checkpoint_minutes:
        parser.error("Use equal monitor and checkpoint intervals for atomic progress")
    torch.set_num_threads(args.threads)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    state_path = output_dir / "state.json"
    checkpoint_path = output_dir / "latest_checkpoint.pt"
    frozen_source = output_dir / "source_checkpoint.pt"

    if args.resume:
        if not all(path.exists() for path in (manifest_path, state_path, checkpoint_path)):
            parser.error("Resume requires manifest.json, state.json, and latest_checkpoint.pt")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (manifest["traversals_per_player"] != args.traversals
                or manifest["monitor_minutes"] != args.monitor_minutes):
            parser.error("Resume settings disagree with manifest")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        trainer = TabularRegretFork.load_fork_checkpoint(checkpoint_path)
        if trainer.iteration != state["iteration"]:
            raise RuntimeError("Fork checkpoint and state disagree")
        measured_s = float(state["measured_fork_s"])
        next_monitor_s = float(state["next_monitor_s"])
        print(f"[resume] iter={trainer.iteration} fork={measured_s/60:.1f}m", flush=True)
    else:
        if manifest_path.exists() or checkpoint_path.exists():
            parser.error("Output directory contains a run; use --resume")
        if args.source_checkpoint is None or not args.source_checkpoint.is_file():
            parser.error("A new run needs an existing --source-checkpoint")
        source = args.source_checkpoint.resolve()
        source_size = source.stat().st_size
        if shutil.disk_usage(output_dir).free < 2 * source_size + 4 * 1024**3:
            raise OSError("Insufficient disk to freeze source and store a resumable fork")
        if not frozen_source.exists():
            # Existing runners replace rolling checkpoints atomically. A hard
            # link therefore pins the source inode without another 726 MiB
            # copy; fall back to a copy across different filesystems.
            try:
                os.link(source, frozen_source)
            except OSError:
                tmp = frozen_source.with_name(frozen_source.name + ".tmp")
                shutil.copyfile(source, tmp)
                os.replace(tmp, frozen_source)
        trainer = TabularRegretFork.load_checkpoint(frozen_source, device="cpu")
        source_iteration = trainer.iteration
        print(f"[fork] source iteration {source_iteration}; activating lazy regret table",
              flush=True)
        trainer.activate_regret_table()
        print(f"[fork] RAM table capacity={trainer.regret_table.numel()*4/1024**2:.1f} MiB; "
              "checkpoints store only visited rows",
              flush=True)
        measured_s = 0.0
        next_monitor_s = args.monitor_minutes * 60
        atomic_json(manifest_path, {
            "created_utc": utc(),
            "source_checkpoint": str(source),
            "frozen_source_checkpoint": str(frozen_source),
            "source_iteration": source_iteration,
            "source_checkpoint_size": source_size,
            "traversals_per_player": args.traversals,
            "monitor_minutes": args.monitor_minutes,
            "regret_update": "visited conditional mean, cumulative clip once",
            "regret_accumulation_mode": trainer.regret_accumulation_mode,
            "average_policy": "continued source neural strategy network and replay",
        })
        save_checkpoint(trainer, checkpoint_path, state_path, measured_s, next_monitor_s)
        evaluate_average(trainer, output_dir, "0000m", measured_s)

    stop_requested = False

    def request_stop(signum, _frame):
        nonlocal stop_requested
        print(f"[signal] {signum}; checkpoint after current iteration", flush=True)
        stop_requested = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    target_s = args.additional_minutes * 60
    while measured_s < target_s and not stop_requested:
        start = time.perf_counter()
        row = trainer.run_iteration(traversals_per_player=args.traversals)
        iteration_s = time.perf_counter() - start
        measured_s += iteration_s
        append_jsonl(output_dir / "training.jsonl", {
            "utc": utc(), "iteration": trainer.iteration,
            "measured_fork_min": measured_s / 60,
            "iteration_s": iteration_s,
            "traversal_s": row["timing"]["traversal_s"],
            "tabular_update_s": row["timing"]["regret_training_s"],
            "strategy_fit_s": row["timing"]["strategy_training_s"],
            "regret_records": row["new_regret_records"],
        })
        if trainer.iteration % 100 == 0:
            print(f"[train] fork={measured_s/60:.1f}m iter={trainer.iteration} "
                  f"iter_s={iteration_s:.2f} traversal_s={row['timing']['traversal_s']:.2f} "
                  f"table_s={row['timing']['regret_training_s']:.2f}", flush=True)
        if measured_s >= next_monitor_s:
            label = f"{int(round(next_monitor_s/60)):04d}m"
            next_monitor_s += args.monitor_minutes * 60
            save_checkpoint(trainer, checkpoint_path, state_path, measured_s, next_monitor_s)
            evaluate_average(trainer, output_dir, label, measured_s)

    save_checkpoint(trainer, checkpoint_path, state_path, measured_s, next_monitor_s)
    atomic_json(output_dir / "summary.json", {
        "status": "paused" if stop_requested else "target_reached",
        "iteration": trainer.iteration,
        "measured_fork_min": measured_s / 60,
        "checkpoint_path": str(checkpoint_path),
        "updated_utc": utc(),
    })
    print(f"[done] fork={measured_s/60:.1f}m iter={trainer.iteration}", flush=True)


if __name__ == "__main__":
    main()
