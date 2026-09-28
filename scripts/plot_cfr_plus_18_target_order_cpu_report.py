#!/usr/bin/env python3
"""Copy exact evaluations and plot the 18-claim target-order experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def collect_rows(run_dir: Path) -> list[dict]:
    rows = read_jsonl(run_dir / "exact_evaluations.jsonl")
    for row in rows:
        events = read_jsonl(run_dir / f"{row['mode']}__seed_{row['seed']}" / "events.jsonl")
        matches = [event for event in events if event.get("event") == "policy_snapshot"
                   and int(event["label"].removesuffix("m")) == row["snapshot_min"]]
        if len(matches) != 1:
            raise ValueError(f"Expected one snapshot event for {row['mode']} seed={row['seed']} "
                             f"minute={row['snapshot_min']}; found {len(matches)}")
        row["iteration"] = int(matches[0]["iteration"])
        row["measured_training_s"] = float(matches[0]["measured_training_s"])
    keys = [(row["mode"], row["seed"], row["snapshot_min"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate exact evaluation")
    return sorted(rows, key=lambda row: (row["mode"], row["seed"], row["snapshot_min"]))


def plot(rows: list[dict], output: Path, *, x_field: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for mode, color, label in (
        ("clip_each_record", "C0", "Clip each record"),
        ("aggregate_then_clip", "C1", "Aggregate then clip"),
    ):
        for seed in (17, 23):
            arm = sorted((r for r in rows if r["mode"] == mode and r["seed"] == seed),
                         key=lambda r: r["snapshot_min"])
            if not arm:
                continue
            original = [r for r in arm if r["snapshot_min"] <= 75]
            continuation = [r for r in arm if r["snapshot_min"] >= 75]
            name = f"{label}, seed {seed}"
            ax.plot([r[x_field] for r in original], [r["exploitability"] for r in original],
                    marker="o", color=color, alpha=0.75, label=name)
            if len(continuation) > 1:
                ax.plot([r[x_field] for r in continuation],
                        [r["exploitability"] for r in continuation],
                        marker="o", linestyle="--", color=color, alpha=0.75)
    if x_field == "snapshot_min":
        ax.axvline(75, color="0.4", linestyle=":", label="Original comparison ends")
        ax.set_xlabel("Measured neural-training minutes")
    else:
        ax.set_xlabel("CFR+ iteration")
    ax.set_ylabel("Exact average-policy exploitability (log scale)")
    ax.set_yscale("log")
    ax.set_title("18-claim neural CFR+: regret target construction")
    ax.grid(which="both", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    args = parser.parse_args()
    rows = collect_rows(args.run_dir)
    data_dir = args.docs_dir / "data"
    figures_dir = args.docs_dir / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    data_path = data_dir / "neural_18_claim_target_order_105m.jsonl"
    data_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    stem = "experiment_cfr_plus_18_claim_target_order_105m"
    plot(rows, figures_dir / f"{stem}.png", x_field="snapshot_min")
    plot(rows, figures_dir / f"{stem}_iterations.png", x_field="iteration")
    print(f"Wrote {len(rows)} evaluations, data, and two figures")


if __name__ == "__main__":
    main()
