#!/usr/bin/env python3
"""Evaluate saved average-policy fit-sweep candidates in parallel on CPU."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from liars_poker.serialization import load_policy
from run_cfr_plus_18_offline_average_fitting import evaluate, plot


def evaluate_one(task: dict) -> dict:
    import torch
    torch.set_num_threads(1)
    reference, _ = load_policy(str(task["output"] / "exact"))
    policy, _ = load_policy(str(task["output"] / task["name"]))
    return {
        "name": task["name"],
        "variant": task["variant"],
        "iteration": task["iteration"],
        "measured_training_min": task["measured_training_min"],
        "refit_steps": task["refit_steps"],
        "fit_s": task["fit_s"],
        **evaluate(policy, reference),
    }


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="Fit-only output directory for one checkpoint")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    output = args.output.resolve()
    source = json.loads((output / "source.json").read_text(encoding="utf-8"))
    progress_path = output / "fit_progress.jsonl"
    progress = [json.loads(line) for line in progress_path.read_text(
        encoding="utf-8").splitlines() if line.strip()]
    iteration = int(source["iteration"])
    measured_min = float(source["measured_training_min"])
    milestones = list(source["milestones"])
    tasks = []
    for name in ("exact", "online"):
        tasks.append({"output": output, "name": name, "variant": name,
                      "iteration": iteration, "measured_training_min": measured_min,
                      "refit_steps": 0, "fit_s": 0.0})
    for row in progress:
        tasks.append({"output": output, "name": row["name"],
                      "variant": row["variant"], "iteration": iteration,
                      "measured_training_min": measured_min,
                      "refit_steps": int(row["refit_steps"]),
                      "fit_s": float(row["fit_s"])})
    expected_names = {task["name"] for task in tasks}
    missing = [name for name in expected_names if not (output / name).is_dir()]
    if missing:
        raise FileNotFoundError(f"Missing saved candidate policies: {missing}")
    if len(tasks) != 2 + 2 * len(milestones):
        raise ValueError(f"Expected {2 + 2 * len(milestones)} policies, found {len(tasks)}")

    result_path = output / "results.jsonl"
    if result_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing evaluation data: {result_path}")
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(evaluate_one, task): task for task in tasks}
        for future in as_completed(futures):
            row = future.result()
            results.append(row)
            with result_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
                handle.flush()
            print(f"[eval {len(results)}/{len(tasks)}] {row['name']} "
                  f"exploitability={row['exploitability']:.8f} "
                  f"eval_s={row['evaluation_s']:.1f}", flush=True)

    plot(output)
    print(f"[complete] {result_path}", flush=True)


if __name__ == "__main__":
    main()
