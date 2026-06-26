#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from liars_poker.training.br_runs import run_best_responder


SNAPSHOT_RE = re.compile(r"^(\d+)m$")


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".new")
    tmp.write_text(
        json.dumps(payload, indent=2, default=json_default, sort_keys=True),
        encoding="utf-8",
    )
    tmp.replace(path)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, default=json_default, sort_keys=True) + "\n")


def snapshot_minute(path: Path) -> int | None:
    match = SNAPSHOT_RE.match(path.name)
    if match is None:
        return None
    return int(match.group(1))


def discover_snapshots(
    run_dir: Path,
    *,
    stride_minutes: int,
    targets: list[int] | None,
) -> list[tuple[int, Path, dict[str, Any]]]:
    root = run_dir / "policy_snapshots"
    if not root.is_dir():
        raise FileNotFoundError(f"Missing policy_snapshots directory: {root}")

    snapshots: list[tuple[int, Path, dict[str, Any]]] = []
    wanted = set(targets or [])
    for policy_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        minute = snapshot_minute(policy_dir)
        if minute is None:
            continue
        if targets is not None:
            if minute not in wanted:
                continue
        elif minute <= 0 or minute % int(stride_minutes) != 0:
            continue
        if not (policy_dir / "metadata.json").exists():
            continue
        snapshot_meta = {}
        snapshot_path = policy_dir / "snapshot.json"
        if snapshot_path.exists():
            snapshot_meta = read_json(snapshot_path)
        snapshots.append((minute, policy_dir, snapshot_meta))

    if not snapshots:
        if targets is None:
            label = f"every {stride_minutes} minutes"
        else:
            label = ", ".join(str(target) for target in targets)
        raise FileNotFoundError(f"No matching policy snapshots found for {label}")
    return sorted(snapshots, key=lambda item: item[0])


def best_summary(evaluations: list[dict[str, Any]]) -> dict[str, float]:
    best_p_first = max(float(row["p_first"]) for row in evaluations)
    best_p_second = max(float(row["p_second"]) for row in evaluations)
    best_p_first_lcb = max(float(row["p_first_lcb"]) for row in evaluations)
    best_p_second_lcb = max(float(row["p_second_lcb"]) for row in evaluations)
    return {
        "best_p_first": best_p_first,
        "best_p_second": best_p_second,
        "best_discovered_estimate": best_p_first + best_p_second - 1.0,
        "best_p_first_lcb": best_p_first_lcb,
        "best_p_second_lcb": best_p_second_lcb,
        "conservative_lower_bound": best_p_first_lcb + best_p_second_lcb - 1.0,
    }


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    columns = [
        "snapshot_minute",
        "snapshot_training_min",
        "snapshot_iteration",
        "responder_seed",
        "responder_training_min",
        "best_p_first",
        "best_p_second",
        "best_discovered_estimate",
        "best_p_first_lcb",
        "best_p_second_lcb",
        "conservative_lower_bound",
        "run_dir",
        "policy_dir",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_plots(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional runtime dependency
        print(f"[plot skipped] matplotlib import failed: {exc}", flush=True)
        return

    ordered = sorted(rows, key=lambda row: float(row["snapshot_training_min"]))
    xs = [float(row["snapshot_training_min"]) for row in ordered]
    estimates = [float(row["best_discovered_estimate"]) for row in ordered]
    lower_bounds = [float(row["conservative_lower_bound"]) for row in ordered]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2))
    axes[0].plot(xs, estimates, marker="o", label="best discovered estimate")
    axes[0].plot(xs, lower_bounds, marker="o", label="conservative lower bound")
    axes[0].set(
        title="Approximate exploitability by CFR+ snapshot",
        xlabel="CFR+ training minutes",
        ylabel="Discovered exploitability",
    )
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    for row in ordered:
        eval_path = Path(str(row["run_dir"])) / "evaluations.jsonl"
        if not eval_path.exists():
            continue
        eval_rows = [
            json.loads(line)
            for line in eval_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not eval_rows:
            continue
        responder_minutes = [
            float(item["measured_training_s"]) / 60.0 for item in eval_rows
        ]
        values = [float(item["exploitability_estimate"]) for item in eval_rows]
        axes[1].plot(
            responder_minutes,
            values,
            marker="o",
            label=f"{int(row['snapshot_minute']):04d}m",
        )

    axes[1].set(
        title="Responder compute curves",
        xlabel="Responder training minutes",
        ylabel="Discovered exploitability",
    )
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.tight_layout()

    plot_path = output_dir / "snapshot_br_summary.png"
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)
    print(f"summary plot: {plot_path}", flush=True)


def parse_targets(value: str | None) -> list[int] | None:
    if value is None or not value.strip():
        return None
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run approximate BRs for saved CFR+ policy snapshots."
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--snapshot-stride-minutes", type=int, default=60)
    parser.add_argument(
        "--targets",
        type=str,
        default=None,
        help="Comma-separated snapshot minutes, e.g. 60,120,180. Overrides stride.",
    )
    parser.add_argument("--minutes", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--episodes-per-role", type=int, default=4096)
    parser.add_argument("--rollout-batch-size", type=int, default=1024)
    parser.add_argument("--eval-every-minutes", type=float, default=1.0)
    parser.add_argument("--eval-episodes-per-role", type=int, default=200_000)
    parser.add_argument("--eval-rollout-batch-size", type=int, default=8192)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else run_dir / f"approx_br_hourly_{int(args.minutes)}m"
    )
    targets = parse_targets(args.targets)
    snapshots = discover_snapshots(
        run_dir,
        stride_minutes=int(args.snapshot_stride_minutes),
        targets=targets,
    )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    trainer_kwargs = {
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
        "seed": int(args.seed),
    }

    manifest = {
        "run_type": "cfr_plus_snapshot_approx_brs",
        "source_run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "snapshot_stride_minutes": int(args.snapshot_stride_minutes),
        "targets": targets,
        "minutes": float(args.minutes),
        "seed": int(args.seed),
        "trainer_kwargs": trainer_kwargs,
        "episodes_per_role": int(args.episodes_per_role),
        "rollout_batch_size": int(args.rollout_batch_size),
        "eval_every_minutes": float(args.eval_every_minutes),
        "eval_episodes_per_role": int(args.eval_episodes_per_role),
        "eval_rollout_batch_size": int(args.eval_rollout_batch_size),
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output_dir / "manifest.json", manifest)

    summary_rows: list[dict[str, Any]] = []
    summary_jsonl = output_dir / "snapshot_br_summary.jsonl"
    summary_csv = output_dir / "snapshot_br_summary.csv"

    for minute, policy_dir, snapshot_meta in snapshots:
        label = f"{minute:04d}m"
        br_dir = output_dir / f"snapshot_{label}"
        metrics_path = br_dir / "metrics.json"
        if metrics_path.exists() and not args.force:
            print(f"[skip] {label}: existing {metrics_path}", flush=True)
            metrics = read_json(metrics_path)
            final_eval = metrics["final_evaluation"]
            evaluations = [final_eval]
            eval_path = br_dir / "evaluations.jsonl"
            if eval_path.exists():
                evaluations = [
                    json.loads(line)
                    for line in eval_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
            row = {
                "snapshot_minute": minute,
                "snapshot_training_min": float(
                    snapshot_meta.get("measured_training_min", minute)
                ),
                "snapshot_iteration": int(snapshot_meta.get("iteration", -1)),
                "responder_seed": int(args.seed),
                "responder_training_min": float(
                    metrics.get("measured_training_s", 60.0 * float(args.minutes))
                )
                / 60.0,
                "run_dir": str(br_dir),
                "policy_dir": str(policy_dir),
                **best_summary(evaluations),
            }
            summary_rows.append(row)
            continue

        print(
            f"\n[BR] snapshot={label} policy={policy_dir} "
            f"minutes={float(args.minutes):.1f} device={device}",
            flush=True,
        )
        result = run_best_responder(
            policy_dir,
            method="action_conditioned_fitted_return",
            minutes=float(args.minutes),
            trainer_kwargs=trainer_kwargs,
            episodes_per_role=int(args.episodes_per_role),
            rollout_batch_size=int(args.rollout_batch_size),
            evaluate_every_minutes=float(args.eval_every_minutes),
            eval_episodes_per_role=int(args.eval_episodes_per_role),
            run_dir=br_dir,
            debug=True,
        )
        final_eval = result.evaluation_records[-1]
        row = {
            "snapshot_minute": minute,
            "snapshot_training_min": float(
                snapshot_meta.get("measured_training_min", minute)
            ),
            "snapshot_iteration": int(snapshot_meta.get("iteration", -1)),
            "responder_seed": int(args.seed),
            "responder_training_min": float(result.measured_training_s) / 60.0,
            "run_dir": str(result.run_dir),
            "policy_dir": str(policy_dir),
            **best_summary(result.evaluation_records),
            "final_p_first": float(final_eval["p_first"]),
            "final_p_second": float(final_eval["p_second"]),
            "final_exploitability_estimate": float(
                final_eval["exploitability_estimate"]
            ),
            "final_exploitability_lower_bound": float(
                final_eval["exploitability_lower_bound"]
            ),
        }
        summary_rows.append(row)
        append_jsonl(summary_jsonl, row)
        write_summary_csv(summary_csv, summary_rows)
        write_plots(output_dir, summary_rows)
        print(
            f"[BR done] {label}: estimate={row['best_discovered_estimate']:.5f} "
            f"LCB={row['conservative_lower_bound']:.5f}",
            flush=True,
        )

    write_summary_csv(summary_csv, summary_rows)
    write_plots(output_dir, summary_rows)
    write_json(
        output_dir / "summary.json",
        {
            **manifest,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "snapshots_evaluated": len(summary_rows),
            "summary_csv": str(summary_csv),
        },
    )
    print(f"\nsummary CSV: {summary_csv}", flush=True)


if __name__ == "__main__":
    main()
