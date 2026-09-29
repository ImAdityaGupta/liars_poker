#!/usr/bin/env python3
"""Run independent 18-claim CFR+ target/sampling arms concurrently on a CPU host.

Each arm uses the existing checkpointing trainer and has its own directory. Exact
BR evaluation happens after training, to avoid evaluation jobs changing the
training jobs' CPU throughput. Pass --resume with the same --output-root after a
host restart. Use --smoke before an expensive run.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINER = REPO_ROOT / "scripts" / "run_cfr_plus_18_target_order_cpu_overnight.py"
EVALUATOR = REPO_ROOT / "scripts" / "evaluate_cfr_plus_18_target_order_cpu.py"
MODES = ("clip_each_record", "aggregate_then_clip")


def available_memory_gib() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / (1024 * 1024)
    raise RuntimeError("MemAvailable not found")


def plan(args: argparse.Namespace) -> list[dict]:
    return [
        {"mode": mode, "traversals": traversals, "seed": seed,
         "name": f"trav{traversals}__{mode}__seed{seed}"}
        for traversals in args.traversals
        for mode in args.modes
        for seed in args.seeds
    ]


def source_hashes() -> dict[str, str]:
    paths = [Path(__file__).resolve(), TRAINER, EVALUATOR,
             REPO_ROOT / "liars_poker/algo/deep_cfr_plus.py",
             REPO_ROOT / "liars_poker/algo/neural_cfr_plus_gpu.py"]
    return {str(path.relative_to(REPO_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths}


def launch(arm: dict, args: argparse.Namespace, root: Path):
    arm_root = root / arm["name"]
    arm_root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": str(args.threads_per_arm),
        "MKL_NUM_THREADS": str(args.threads_per_arm),
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    })
    command = [
        sys.executable, "-u", str(TRAINER),
        "--hours-per-arm", str(args.minutes_per_arm / 60),
        "--traversals", str(arm["traversals"]),
        "--seeds", str(arm["seed"]),
        "--modes", arm["mode"],
        "--reach-mode", args.reach_mode,
        "--torch-threads", str(args.threads_per_arm),
        "--snapshot-minutes", str(args.snapshot_minutes),
        "--checkpoint-minutes", str(args.checkpoint_minutes),
        "--output-root", str(arm_root),
    ]
    if args.resume:
        command.append("--resume")
    log = (arm_root / "train.log").open("a", encoding="utf-8")
    process = subprocess.Popen(command, cwd=REPO_ROOT, env=env,
                               stdout=log, stderr=subprocess.STDOUT)
    print(f"start {arm['name']} pid={process.pid} memory_available={available_memory_gib():.1f}GiB", flush=True)
    return process, log


def evaluate(root: Path, arms: list[dict]) -> list[dict]:
    combined: list[dict] = []
    for arm in arms:
        arm_root = root / arm["name"]
        log_path = arm_root / "evaluate.log"
        with log_path.open("a", encoding="utf-8") as log:
            subprocess.run([sys.executable, "-u", str(EVALUATOR), str(arm_root)],
                           cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        for line in (arm_root / "exact_evaluations.jsonl").read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                row["traversals"] = arm["traversals"]
                combined.append(row)
        print(f"evaluated {arm['name']}", flush=True)
    fieldnames = ["traversals", "mode", "seed", "snapshot_min", "p_first",
                  "p_second", "exploitability", "evaluation_s", "policy_dir"]
    with (root / "exact_evaluations.csv").open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(combined)
    return combined


def plot(root: Path, rows: list[dict], arms: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colours = {MODES[0]: "C0", MODES[1]: "C1"}
    styles = {value: style for value, style in zip(sorted({a["traversals"] for a in arms}), ["-", "--", ":", "-."])}
    for arm in arms:
        subset = sorted((r for r in rows if r["traversals"] == arm["traversals"]
                         and r["mode"] == arm["mode"] and r["seed"] == arm["seed"]),
                        key=lambda r: r["snapshot_min"])
        if not subset:
            continue
        label = f"{arm['mode']}, trav={arm['traversals']}, seed={arm['seed']}"
        for ax in axes:
            ax.plot([r["snapshot_min"] for r in subset],
                    [r["exploitability"] for r in subset],
                    color=colours[arm["mode"]], linestyle=styles[arm["traversals"]],
                    marker=".", alpha=0.85, label=label)
    # The second axis is filled below with iteration indices from snapshot events.
    axes[1].clear()
    for arm in arms:
        arm_dir = root / arm["name"] / f"{arm['mode']}__seed_{arm['seed']}"
        event_path = arm_dir / "events.jsonl"
        if not event_path.exists():
            continue
        iterations = {int(e["label"].removesuffix("m")): e["iteration"]
                      for line in event_path.read_text().splitlines() if line.strip()
                      for e in [json.loads(line)] if e.get("event") == "policy_snapshot"}
        subset = sorted((r for r in rows if r["traversals"] == arm["traversals"]
                         and r["mode"] == arm["mode"] and r["seed"] == arm["seed"]
                         and r["snapshot_min"] in iterations),
                        key=lambda r: r["snapshot_min"])
        axes[1].plot([iterations[r["snapshot_min"]] for r in subset],
                     [r["exploitability"] for r in subset],
                     color=colours[arm["mode"]], linestyle=styles[arm["traversals"]],
                     marker=".", alpha=0.85)
    axes[0].set_xlabel("Measured training minutes")
    axes[1].set_xlabel("CFR+ iterations")
    for ax in axes:
        ax.set_yscale("log")
        ax.set_ylabel("Exact average-policy exploitability")
        ax.grid(True, which="both", alpha=0.25)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(root / "exact_exploitability.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--minutes-per-arm", type=float, default=150)
    parser.add_argument("--traversals", type=lambda s: [int(v) for v in s.split(",")], default=[1024, 4096])
    parser.add_argument("--seeds", type=lambda s: [int(v) for v in s.split(",")], default=[17, 23])
    parser.add_argument("--modes", type=lambda s: [v.strip() for v in s.split(",")], default=list(MODES))
    parser.add_argument("--reach-mode", choices=("none", "visit_fraction"), default="none")
    parser.add_argument("--threads-per-arm", type=int, default=8)
    parser.add_argument("--max-parallel", type=int, default=8)
    parser.add_argument("--snapshot-minutes", type=float, default=15)
    parser.add_argument("--checkpoint-minutes", type=float, default=30)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--train-only", action="store_true",
                        help="Stop after training; useful for short throughput probes")
    args = parser.parse_args()
    if (args.minutes_per_arm <= 0 or args.threads_per_arm <= 0 or args.max_parallel <= 0
            or min(args.traversals) <= 0 or min(args.seeds) < 0
            or args.snapshot_minutes <= 0 or args.checkpoint_minutes <= 0):
        parser.error("all budgets and traversal counts must be positive")
    if not args.modes or len(set(args.modes)) != len(args.modes) or any(m not in MODES for m in args.modes):
        parser.error("modes must be a unique nonempty subset of supported modes")
    if args.reach_mode == "visit_fraction" and args.modes != ["aggregate_then_clip"]:
        parser.error("visit_fraction requires aggregate_then_clip only")
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    arms = plan(args)
    manifest_path = root / "parallel_manifest.json"
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(),
                "minutes_per_arm": args.minutes_per_arm, "traversals": args.traversals,
                "seeds": args.seeds, "modes": args.modes,
                "reach_mode": args.reach_mode, "threads_per_arm": args.threads_per_arm,
                "snapshot_minutes": args.snapshot_minutes,
                "checkpoint_minutes": args.checkpoint_minutes, "arms": arms,
                "source_sha256": source_hashes()}
    if args.resume:
        previous = json.loads(manifest_path.read_text())
        for key in ("traversals", "seeds", "modes", "threads_per_arm", "snapshot_minutes", "checkpoint_minutes", "arms"):
            if previous[key] != manifest[key]:
                raise ValueError(f"Resume manifest differs on {key}")
        if previous.get("reach_mode", "none") != args.reach_mode:
            raise ValueError("Resume manifest differs on reach_mode")
        old_minutes = float(previous["minutes_per_arm"])
        if args.minutes_per_arm < old_minutes:
            raise ValueError("Resume target cannot be shorter than the recorded run")
        if args.minutes_per_arm > old_minutes:
            previous.setdefault("extensions", []).append({
                "utc": datetime.now(timezone.utc).isoformat(),
                "from_minutes_per_arm": old_minutes,
                "to_minutes_per_arm": args.minutes_per_arm,
                "source_sha256": source_hashes(),
            })
            previous["minutes_per_arm"] = args.minutes_per_arm
            temporary = manifest_path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(previous, indent=2), encoding="utf-8")
            temporary.replace(manifest_path)
            print(f"extend target {old_minutes:g} -> {args.minutes_per_arm:g} minutes per arm", flush=True)
    else:
        if manifest_path.exists():
            raise FileExistsError(manifest_path)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"root={root} arms={len(arms)} max_parallel={args.max_parallel}", flush=True)
    active = {}
    pending = list(arms)
    failed = []
    last_report = 0.0
    try:
        while pending or active:
            while pending and len(active) < args.max_parallel:
                if available_memory_gib() < 12:
                    raise MemoryError("Less than 12 GiB available; refusing to start another arm")
                arm = pending.pop(0)
                process, log = launch(arm, args, root)
                active[arm["name"]] = (process, log)
                time.sleep(2)
            for name, (process, log) in list(active.items()):
                code = process.poll()
                if code is not None:
                    log.close()
                    del active[name]
                    print(f"finish {name} exit={code}", flush=True)
                    if code != 0:
                        failed.append(name)
            now = time.monotonic()
            if now - last_report >= 60:
                print(f"running={len(active)} pending={len(pending)} failed={len(failed)} "
                      f"memory_available={available_memory_gib():.1f}GiB", flush=True)
                last_report = now
            if active:
                time.sleep(5)
    except KeyboardInterrupt:
        print("Interrupted; waiting for each arm to reach its next checkpoint is safer than force-killing.", flush=True)
        raise
    if failed:
        raise RuntimeError(f"Training arms failed; inspect their train.log: {failed}")
    if args.train_only:
        print(f"training complete: {root}", flush=True)
        return
    rows = evaluate(root, arms)
    plot(root, rows, arms)
    print(f"complete: {len(rows)} exact policy evaluations, {root}", flush=True)


if __name__ == "__main__":
    main()
