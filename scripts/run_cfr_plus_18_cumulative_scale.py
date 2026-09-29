#!/usr/bin/env python3
"""Run three cumulative-regret neural CFR+ comparisons concurrently on CPU."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "scripts" / "run_cfr_plus_18_target_order_cpu_overnight.py"
CONDITIONS = (
    {"name": "nk1024", "label": "Cumulative + N/K · 1,024 roots",
     "traversals": 1024, "reach_mode": "visit_fraction",
     "regret_buffer_capacity": 500_000, "color": "#007F5F"},
    {"name": "nk4096", "label": "Cumulative + N/K · 4,096 roots",
     "traversals": 4096, "reach_mode": "visit_fraction",
     "regret_buffer_capacity": 4_000_000, "color": "#B21E5B"},
    {"name": "conditional4096", "label": "Cumulative conditional · 4,096 roots",
     "traversals": 4096, "reach_mode": "none",
     "regret_buffer_capacity": 4_000_000, "color": "#C27600"},
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def available_memory_gib() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / (1024 * 1024)
    raise RuntimeError("MemAvailable missing")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--minutes-per-arm", type=float, default=330)
    parser.add_argument("--threads-per-arm", type=int, default=8)
    parser.add_argument("--snapshot-minutes", type=float, default=15)
    parser.add_argument("--checkpoint-minutes", type=float, default=15)
    args = parser.parse_args()
    if min(args.minutes_per_arm, args.threads_per_arm,
           args.snapshot_minutes, args.checkpoint_minutes) <= 0:
        parser.error("all budgets and thread counts must be positive")

    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "parallel_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"Run already exists: {manifest_path}")
    arms = [{**condition, "mode": "aggregate_then_clip", "seed": 17}
            for condition in CONDITIONS]
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "cumulative regret scale and reach weighting",
        "minutes_per_arm": args.minutes_per_arm,
        "snapshot_minutes": args.snapshot_minutes,
        "checkpoint_minutes": args.checkpoint_minutes,
        "threads_per_arm": args.threads_per_arm,
        "regret_accumulation_mode": "cumulative",
        "arms": arms,
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in (Path(__file__).resolve(), TRAINER,
                         ROOT / "liars_poker/algo/deep_cfr_plus.py",
                         ROOT / "liars_poker/algo/neural_cfr_plus_gpu.py")
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (root / "run_started.json").write_text(
        json.dumps({"utc": datetime.now(timezone.utc).isoformat(),
                    "pid": os.getpid(), "status": "running"}, indent=2),
        encoding="utf-8",
    )

    processes: dict[str, tuple[subprocess.Popen, object]] = {}
    try:
        for arm in arms:
            if available_memory_gib() < 12:
                raise MemoryError("Less than 12 GiB available; refusing to start arm")
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
                "--output-root", str(arm_root),
                "--hours-per-arm", str(args.minutes_per_arm / 60),
                "--traversals", str(arm["traversals"]),
                "--seeds", "17", "--modes", "aggregate_then_clip",
                "--reach-mode", arm["reach_mode"],
                "--regret-accumulation-mode", "cumulative",
                "--regret-buffer-capacity", str(arm["regret_buffer_capacity"]),
                "--torch-threads", str(args.threads_per_arm),
                "--snapshot-minutes", str(args.snapshot_minutes),
                "--checkpoint-minutes", str(args.checkpoint_minutes),
            ]
            log = (arm_root / "train.log").open("a", encoding="utf-8")
            process = subprocess.Popen(
                command, cwd=ROOT, env=env, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True,
            )
            processes[arm["name"]] = (process, log)
            print(f"start {arm['label']}: pid={process.pid}, "
                  f"RAM available={available_memory_gib():.1f} GiB", flush=True)

        while processes:
            for name, (process, log) in list(processes.items()):
                code = process.poll()
                if code is not None:
                    log.close()
                    del processes[name]
                    print(f"finish {name}: exit={code}", flush=True)
                    if code:
                        raise subprocess.CalledProcessError(code, name)
            if processes:
                print(f"running={len(processes)} RAM available="
                      f"{available_memory_gib():.1f} GiB", flush=True)
                time.sleep(60)
        (root / "run_started.json").write_text(
            json.dumps({"utc": datetime.now(timezone.utc).isoformat(),
                        "pid": os.getpid(), "status": "complete"}, indent=2),
            encoding="utf-8",
        )
    except BaseException:
        for process, _ in processes.values():
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
        for process, log in processes.values():
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait()
            log.close()
        (root / "run_started.json").write_text(
            json.dumps({"utc": datetime.now(timezone.utc).isoformat(),
                        "pid": os.getpid(), "status": "interrupted"}, indent=2),
            encoding="utf-8",
        )
        raise


if __name__ == "__main__":
    main()
