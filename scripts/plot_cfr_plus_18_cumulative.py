#!/usr/bin/env python3
"""Plot archived exact exploitability for the cumulative-regret screen."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs/data"


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")
for name, label, color in (
    ("nk1024", "Cumulative N/K, K=1024", "#007f5f"),
    ("nk4096", "Cumulative N/K, K=4096", "#b21e5b"),
    ("conditional4096", "Cumulative conditional, K=4096", "#c27600"),
):
    evaluations = rows(DATA / f"cfr_plus_18_cumulative_{name}_20260929.jsonl")
    events = rows(DATA / f"cfr_plus_18_cumulative_{name}_events_20260929.jsonl")
    iterations = {int(event["label"][:-1]): int(event["iteration"])
                  for event in events if event.get("event") == "policy_snapshot"}
    points = [(int(row["snapshot_min"]), iterations[int(row["snapshot_min"])],
               float(row["exploitability"])) for row in evaluations
              if int(row["snapshot_min"]) in iterations]
    for ax, column in zip(axes, (0, 1)):
        ax.plot([point[column] for point in points], [point[2] for point in points],
                marker="o", markersize=4, color=color, label=label)

old = [row for row in rows(DATA / "neural_18_claim_parallel_cpu_330m_20260928.jsonl")
       if row.get("arm") == "trav4096__aggregate_then_clip__seed17"]
for ax, column in zip(axes, ("snapshot_min", "iteration")):
    ax.plot([row[column] for row in old], [row["exploitability"] for row in old],
            marker=".", markersize=3, color="#386cb0", alpha=.8,
            label="Earlier normalized conditional, K=4096")

for ax, xlabel in zip(axes, ("Measured training minutes", "CFR+ iteration")):
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Average-policy exact exploitability")
    ax.set_yscale("log")
    ax.grid(alpha=.25, which="both")
axes[1].legend(fontsize=7.8, loc="upper right")
out = ROOT / "docs/figures/experiment_cfr_plus_18_cumulative_regret_105m.png"
fig.savefig(out, dpi=160)
print(out)

# Standalone view of the two cumulative N/K failures and the conditional control.
fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
for name, label, color in (
    ("nk1024", "Cumulative N/K, 1,024 roots", "#007f5f"),
    ("nk4096", "Cumulative N/K, 4,096 roots", "#b21e5b"),
    ("conditional4096", "Cumulative conditional, 4,096 roots", "#c27600"),
):
    evaluations = rows(DATA / f"cfr_plus_18_cumulative_{name}_20260929.jsonl")
    events = rows(DATA / f"cfr_plus_18_cumulative_{name}_events_20260929.jsonl")
    iterations = {int(event["label"][:-1]): int(event["iteration"])
                  for event in events if event.get("event") == "policy_snapshot"}
    points = [(int(row["snapshot_min"]), iterations[int(row["snapshot_min"])],
               float(row["exploitability"])) for row in evaluations
              if int(row["snapshot_min"]) in iterations and int(row["snapshot_min"]) <= 105]
    for ax, column in zip(axes, (0, 1)):
        ax.plot([point[column] for point in points], [point[2] for point in points],
                marker="o", markersize=4, linewidth=2, color=color, label=label)

normalized = [row for row in rows(DATA / "neural_18_claim_parallel_cpu_330m_20260928.jsonl")
              if row.get("arm") == "trav4096__aggregate_then_clip__seed17"
              and int(row["snapshot_min"]) <= 105]
for ax, field in zip(axes, ("snapshot_min", "iteration")):
    ax.plot([row[field] for row in normalized],
            [row["exploitability"] for row in normalized],
            marker="s", markersize=4, linewidth=2, color="#386cb0",
            label="Earlier normalized conditional, seed 17, 4,096 roots")

for ax, xlabel in zip(axes, ("Measured training minutes", "CFR+ iteration")):
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Exact average-policy exploitability")
    ax.set_yscale("log")
    ax.grid(alpha=.25, which="both")
axes[0].set_title("Same measured training time")
axes[1].set_title("Same iteration count")
axes[1].legend(fontsize=8, loc="best")
out = ROOT / "docs/figures/experiment_cfr_plus_18_nk_failure_105m.png"
fig.savefig(out, dpi=160)
print(out)

# Current continuation graph: preserve the completed N/K failures while
# extending the conditional curve and adding the separate N-weighted follow-up.
live = []
for arm_name, eval_file, event_file in (
    ("conditional4096", "cfr_plus_18_cumulative_conditional4096_extension_20260929.jsonl",
     "cfr_plus_18_cumulative_conditional4096_extension_events_20260929.jsonl"),
    ("n4096", "cfr_plus_18_cumulative_visit_count_n4096_20260929.jsonl",
     "cfr_plus_18_cumulative_visit_count_n4096_events_20260929.jsonl"),
):
    evaluations = rows(DATA / eval_file)
    iterations = {int(event["label"][:-1]): int(event["iteration"])
                  for event in rows(DATA / event_file)
                  if event.get("event") == "policy_snapshot"}
    for row in evaluations:
        minute = int(row["snapshot_min"])
        if minute in iterations:
            live.append({**row, "arm": arm_name, "iteration": iterations[minute]})
fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), layout="constrained")
series = [
    ("nk1024", "Cumulative N/K, 1,024 roots", "#007f5f", "archive"),
    ("nk4096", "Cumulative N/K, 4,096 roots", "#b21e5b", "archive"),
    ("conditional4096", "Cumulative conditional, 4,096 roots", "#c27600", "live"),
    ("n4096", "Cumulative visit-count N, 4,096 roots", "#3157a4", "live"),
]
for name, label, color, source in series:
    if source == "archive":
        evaluations = rows(DATA / f"cfr_plus_18_cumulative_{name}_20260929.jsonl")
        events = rows(DATA / f"cfr_plus_18_cumulative_{name}_events_20260929.jsonl")
        iterations = {int(event["label"][:-1]): int(event["iteration"])
                      for event in events if event.get("event") == "policy_snapshot"}
        points = [(int(row["snapshot_min"]), iterations[int(row["snapshot_min"])],
                   float(row["exploitability"])) for row in evaluations
                  if int(row["snapshot_min"]) in iterations]
    else:
        arm_name = "conditional4096" if name == "conditional4096" else "n4096"
        points = [(float(row["snapshot_min"]), int(row["iteration"]),
                   float(row["exploitability"])) for row in live
                  if row.get("arm") == arm_name]
    points.sort(key=lambda point: point[0])
    for ax, column in zip(axes, (0, 1)):
        ax.plot([point[column] for point in points], [point[2] for point in points],
                marker="o", markersize=4, linewidth=2, color=color, label=label)

normalized = [row for row in rows(DATA / "neural_18_claim_parallel_cpu_330m_20260928.jsonl")
              if row.get("arm") == "trav4096__aggregate_then_clip__seed17"]
for ax, field in zip(axes, ("snapshot_min", "iteration")):
    ax.plot([row[field] for row in normalized],
            [row["exploitability"] for row in normalized],
            marker="s", markersize=4, linewidth=1.8, color="#6f91b5",
            label="Earlier normalized conditional, seed 17, 4,096 roots")

for ax, xlabel, title in zip(
    axes, ("Measured training minutes", "CFR+ iteration"),
    ("Equal measured training time", "Equal iteration count"),
):
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.set_ylabel("Exact average-policy exploitability")
    ax.set_yscale("log")
    ax.grid(alpha=.25, which="both")
axes[1].legend(fontsize=8, loc="best")
out = ROOT / "docs/figures/experiment_cfr_plus_18_cumulative_regret_current.png"
fig.savefig(out, dpi=160)
print(out)
