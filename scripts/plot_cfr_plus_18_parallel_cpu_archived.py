#!/usr/bin/env python3
"""Plot archived exact evaluations from the 18-claim CPU factorial run."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs/data/neural_18_claim_parallel_cpu_330m_20260928.jsonl"
FIGURES = ROOT / "docs/figures"
ROWS = [json.loads(line) for line in DATA.read_text(encoding="utf-8").splitlines()
        if line.strip()]

COLOURS = {
    (1024, "clip_each_record"): "#1769aa",
    (1024, "aggregate_then_clip"): "#d66b16",
    (4096, "clip_each_record"): "#079d76",
    (4096, "aggregate_then_clip"): "#8352a6",
}
MODES = ("clip_each_record", "aggregate_then_clip")
SEEDS = (17, 23)

assert len(ROWS) == 8 * 22
assert {(row["traversals"], row["mode"], row["seed"], row["snapshot_min"])
        for row in ROWS} == {(budget, mode, seed, minute)
                            for budget in (1024, 4096) for mode in MODES
                            for seed in SEEDS for minute in range(15, 331, 15)}

FIGURES.mkdir(exist_ok=True)

# Full comparison: each line is one independent seed, shown against both
# measured training time and completed updates.
fig, axes = plt.subplots(1, 2, figsize=(15, 5.6), sharey=True)
for budget in (1024, 4096):
    for mode in MODES:
        for seed in SEEDS:
            group = sorted((row for row in ROWS if row["traversals"] == budget
                            and row["mode"] == mode and row["seed"] == seed),
                           key=lambda row: row["snapshot_min"])
            for ax, xkey in zip(axes, ("snapshot_min", "iteration")):
                ax.plot([row[xkey] for row in group],
                        [row["exploitability"] for row in group],
                        color=COLOURS[(budget, mode)],
                        linestyle="-" if seed == 17 else "--",
                        marker="o" if seed == 17 else "s", markersize=3.5,
                        linewidth=1.8, alpha=0.95)
for ax, xlabel in zip(axes, ("Measured training minutes", "CFR+ iteration")):
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.grid(True, which="both", alpha=0.22)
axes[0].set_ylabel("Exact average-policy exploitability (lower is better)")
axes[0].set_title("Equal training time")
axes[1].set_title("Equal iteration number")
colour_legend = [Line2D([0], [0], color=COLOURS[(budget, mode)], linewidth=2,
                        label=f"{budget:,} · {'aggregate first' if mode == MODES[1] else 'clip each'}")
                 for budget in (1024, 4096) for mode in MODES]
seed_legend = [Line2D([0], [0], color="#333", linestyle="-" if seed == 17 else "--",
                      marker="o" if seed == 17 else "s", label=f"seed {seed}")
               for seed in SEEDS]
fig.legend(handles=colour_legend + seed_legend, ncol=3, loc="lower center",
           bbox_to_anchor=(0.5, -0.025), frameon=False)
fig.tight_layout(rect=(0, 0.12, 1, 1))
full_path = FIGURES / "experiment_cfr_plus_18_parallel_cpu_330m.png"
fig.savefig(full_path, dpi=160, bbox_inches="tight")
plt.close(fig)

# Close-up: means and both individual seeds expose reversals that are hard to
# see at the full run's wider y range.
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
for ax, budget in zip(axes, (1024, 4096)):
    for mode, colour in zip(MODES, ("#1769aa", "#d66b16")):
        by_minute: dict[int, list[float]] = defaultdict(list)
        for seed in SEEDS:
            group = sorted((row for row in ROWS if row["traversals"] == budget
                            and row["mode"] == mode and row["seed"] == seed
                            and row["snapshot_min"] >= 180),
                           key=lambda row: row["snapshot_min"])
            for row in group:
                by_minute[row["snapshot_min"]].append(row["exploitability"])
            ax.plot([row["snapshot_min"] for row in group],
                    [row["exploitability"] for row in group], color=colour,
                    linestyle="--", marker="o" if seed == 17 else "s",
                    markersize=3, linewidth=1, alpha=0.38)
        minutes = sorted(by_minute)
        ax.plot(minutes, [mean(by_minute[m]) for m in minutes], color=colour,
                linewidth=2.6, marker="o", markersize=4,
                label="aggregate first" if mode == MODES[1] else "clip each")
    ax.set_title(f"{budget:,} traversals per player")
    ax.set_xlabel("Measured training minutes")
    ax.set_xlim(177, 333)
    ax.set_yscale("log")
    ax.set_ylim(0.018, 0.046)
    ax.set_yticks([0.02, 0.025, 0.03, 0.04])
    ax.set_yticklabels(["0.020", "0.025", "0.030", "0.040"])
    ax.grid(True, alpha=0.22)
    ax.legend(frameon=False)
axes[0].set_ylabel("Exact average-policy exploitability")
fig.suptitle("Late snapshots: bold = two-seed mean; faint = individual seeds")
fig.tight_layout()
late_path = FIGURES / "experiment_cfr_plus_18_parallel_cpu_late_330m.png"
fig.savefig(late_path, dpi=160, bbox_inches="tight")
plt.close(fig)
print(full_path)
print(late_path)
