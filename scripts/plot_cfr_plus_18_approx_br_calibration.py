#!/usr/bin/env python3
"""Plot the 18-claim approximate best-response calibration: recovery by policy, and recovery against cost."""

from __future__ import annotations

import csv
from pathlib import Path
from statistics import median

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs" / "data" / "cfr_plus_18_approx_br_calibration_20261003" / "summary.csv"
OUT = ROOT / "docs" / "figures" / "experiment_cfr_plus_18_approx_br_calibration.png"
INK, GRID = "#1f2430", "#e6e8ec"
SETTINGS = [
    ("d1 eps=0", "LBR (depth 1)", "#9CA3AF", "o"),
    ("d2 eps=0.001", "Expectimax d=2, ε=10⁻³", "#93C5FD", "s"),
    ("d2 eps=0.0001", "Expectimax d=2, ε=10⁻⁴", "#2563EB", "s"),
    ("d3 eps=0.001", "Expectimax d=3, ε=10⁻³", "#FDBA74", "D"),
    ("d3 eps=0.0001", "Expectimax d=3, ε=10⁻⁴", "#D97706", "D"),
]


def style(ax) -> None:
    ax.grid(True, which="both", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def main() -> None:
    rows = list(csv.DictReader(DATA.open(encoding="utf-8")))
    fig, (left, right) = plt.subplots(1, 2, figsize=(14, 5.4), layout="constrained")
    for key, label, color, marker in SETTINGS:
        sel = sorted((r for r in rows if r["setting"] == key), key=lambda r: float(r["exact"]))
        xs = [float(r["exact"]) for r in sel]
        ys = [float(r["recovery"]) for r in sel]
        left.plot(xs, ys, color=color, marker=marker, markersize=6, linewidth=1.2, label=label)
        costs = [float(r["elapsed_s"]) for r in sel]
        right.scatter(costs, ys, color=color, marker=marker, s=26, alpha=0.55)
        right.scatter([median(costs)], [median(ys)], color=color, marker=marker, s=150,
                      edgecolors=INK, linewidths=1.2, zorder=4, label=label)
    for ax in (left, right):
        ax.axhline(1.0, color=INK, linestyle="--", linewidth=1)
        ax.set_ylim(-0.03, 1.05)
        style(ax)
    left.set(xscale="log", xlabel="Exact exploitability of the policy (log)",
             ylabel="Fraction of exact exploitability found")
    left.set_title("Recovery on each of the 12 policies", loc="left", color=INK)
    left.annotate("LBR finds nothing on the two\nnear-equilibrium policies", (0.00105, 0.0), xytext=(0.0016, 0.18),
                  fontsize=9, color="#6B7280", arrowprops=dict(arrowstyle="->", color="#6B7280", lw=0.8))
    right.set(xscale="log", xlabel="CPU seconds per policy, both seats (dense opponent lookups; log)",
              ylabel="Fraction of exact exploitability found")
    right.set_title("Recovery against cost (large markers: medians)", loc="left", color=INK)
    fig.legend(*right.get_legend_handles_labels(), loc="outside lower center", ncol=5, frameon=False, fontsize=9)
    fig.suptitle("18-claim approximate best responses against exact (all values are lower bounds)", color=INK)
    fig.savefig(OUT, dpi=150)
    print(OUT)


if __name__ == "__main__":
    main()
