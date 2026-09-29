#!/usr/bin/env python3
"""Plot archived exact exploitability for the same r4_s4_h2 18-claim game.

This reads saved evaluation logs only; it does not train or evaluate policies.
The CPU log is an archived copy from the remote experiment.
"""

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
OUT = ROOT / "docs/figures/18_claim_tabular_vs_neural_iterations.png"


def jsonl(relative: str) -> list[dict]:
    path = ROOT / relative
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


tabular = json.loads((ROOT / "artifacts/benchmark_runs/cfr_plus_runs/"
                      "r4_s4_h2_hp2pt_ss___20260108-213016/metrics.json").read_text())
assert json.loads(tabular["spec"]) == {
    "claim_kinds": ["RankHigh", "Pair", "TwoPair", "Trips"],
    "hand_size": 2, "ranks": 4, "suit_symmetry": True, "suits": 4,
}

curves: list[tuple[str, list[int], list[float], str, str, float]] = []
rows = tabular["exploitability_series"]
curves.append(("Exact tabular CFR+", [row["iter"] for row in rows],
               [row["p_first"] + row["p_second"] - 1 for row in rows],
               "#171717", "-", 2.7))

old_gpu = [row for row in jsonl(
    "artifacts/deep_cfr_plus_reference_runs/r4_s4_h2_hp2pt_ss___20260621-024038/"
    "exact_results.jsonl") if row["label"] == "learned_average"]
curves.append(("June neural CFR+ · 100/50 fit steps", [row["iter"] for row in old_gpu],
               [row["exploitability"] for row in old_gpu], "#6484b7", "--", 1.8))

optimized_gpu = jsonl(
    "artifacts/deep_cfr_plus_reference_runs/"
    "r4_s4_h2_hp2pt_ss___optimized_24r6s___20260621-201555/evaluations.jsonl")
curves.append(("June neural CFR+ · 24/6 fit steps", [row["iteration"] for row in optimized_gpu],
               [row["exploitability"] for row in optimized_gpu], "#2456a6", "-", 2.0))


def add_two_seed_means(relative: str, *, current: bool) -> None:
    data = jsonl(relative)
    modes = ("clip_each_record", "aggregate_then_clip")
    budgets = (1024, 4096) if current else (1024,)
    colours = {(1024, modes[0]): "#9b657c", (1024, modes[1]): "#d95f02",
               (4096, modes[0]): "#1b9e77", (4096, modes[1]): "#7755ad"}
    for budget in budgets:
        for mode in modes:
            grouped = defaultdict(list)
            for row in data:
                if row["mode"] == mode and (not current or row["traversals"] == budget):
                    grouped[row["snapshot_min"]].append(row)
            complete = [(minute, group) for minute, group in sorted(grouped.items())
                        if {row["seed"] for row in group} == {17, 23}]
            if not complete:
                continue
            x = [mean(row["iteration"] for row in group) for _, group in complete]
            y = [mean(row["exploitability"] for row in group) for _, group in complete]
            name = "aggregate first" if mode == modes[1] else "clip each"
            era = "September CPU" if current else "September earlier"
            curves.append((f"{era} · {budget:,} traversals · {name} (2-seed mean)",
                           x, y, colours[(budget, mode)], "-" if current else ":", 1.8))


add_two_seed_means("docs/data/neural_18_claim_target_order_105m.jsonl", current=False)
live_rows = jsonl("docs/data/neural_18_claim_parallel_cpu_330m_20260928.jsonl")
add_two_seed_means("docs/data/neural_18_claim_parallel_cpu_330m_20260928.jsonl", current=True)

fig, axes = plt.subplots(1, 2, figsize=(16, 6.8), sharey=True)
for ax in axes:
    for label, x, y, colour, style, width in curves:
        ax.plot(x, y, color=colour, linestyle=style, linewidth=width,
                marker=None if label.startswith("Exact") else "o", markersize=3,
                alpha=0.9, label=label)
    ax.set_yscale("log")
    ax.set_ylim(0.0015, 0.25)
    ax.set_xlabel("CFR+ iteration")
    ax.grid(True, which="both", alpha=0.22)
axes[0].set_xlim(0, 30_000)
axes[0].set_title("Full range, including a reported June fork minimum")
axes[0].scatter([26_807], [0.013930], marker="*", s=150,
                facecolors="none", edgecolors="#b03030", linewidths=1.5, zorder=5)
axes[0].annotate("June LR-only fork best\n0.01393 at 26,807 iter*",
                 (26_807, 0.013930), xytext=(-152, 14), textcoords="offset points",
                 fontsize=9, color="#8e2525",
                 arrowprops={"arrowstyle": "-", "color": "#b03030"})
axes[1].set_xlim(0, 10_500)
axes[1].set_title("First 10,500 iterations")
axes[0].set_ylabel("Exact average-policy exploitability")

handles = [Line2D([0], [0], color=colour, linestyle=style,
                  marker=None if label.startswith("Exact") else "o",
                  markersize=4, linewidth=width, label=label)
           for label, _, _, colour, style, width in curves]
fig.legend(handles=handles, ncol=3, loc="lower center", frameon=False,
           bbox_to_anchor=(0.5, 0.055), fontsize=8.5)
fig.text(0.5, 0.018,
         "* Fork point from the user's prior summary; full fork evaluation rows are not archived here. "
         f"Neural means use the two saved seeds; current CPU data end at {max(row['snapshot_min'] for row in live_rows)}m. "
         "Iterations have different costs across methods.",
         ha="center", fontsize=8, color="#444444")
fig.tight_layout(rect=(0, 0.20, 1, 1))
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=170)
print(OUT)
print(f"Tabular best: {min(curves[0][2]):.9f} at iteration "
      f"{curves[0][1][curves[0][2].index(min(curves[0][2]))]}")
print(f"September CPU snapshot rows: {len(live_rows)}")
