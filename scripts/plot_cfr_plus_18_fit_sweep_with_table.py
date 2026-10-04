#!/usr/bin/env python3
"""Plot the regret fit-step forks against the same-source tabular-regret fork, by CFR+ iteration."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs" / "data"
OUT = ROOT / "docs" / "figures" / "experiment_cfr_plus_18_fit_sweep_with_table.png"
SOURCE_ITERATION = 15_728
WINDOWS = [(15_700, 20_000), (20_000, 24_000), (24_000, 30_000), (30_000, 37_000),
           (37_000, 45_000), (45_000, 55_000), (55_000, 65_000), (65_000, 75_000)]
INK, MUTED, GRID = "#1f2430", "#5d6b80", "#e6e8ec"


def rows(name: str, keep=lambda row: True) -> list[tuple[int, float]]:
    out = [json.loads(line) for line in (DATA / name).read_text(encoding="utf-8").splitlines() if line.strip()]
    return sorted((r["iteration"], r["exploitability"]) for r in out if keep(r))


def window_means(points):
    xs, ys = [], []
    for lo, hi in WINDOWS:
        inside = [(i, e) for i, e in points if lo <= i < hi]
        if len(inside) >= 2:
            xs.append(np.exp(np.mean(np.log([i for i, _ in inside]))))
            ys.append(np.exp(np.mean(np.log([e for _, e in inside]))))
    return xs, ys


cpu = [p for p in rows("cfr_plus_18_gpu_fit_cpu_s24_exact_full_20260930.jsonl") if p[0] >= SOURCE_ITERATION]
average = lambda r: r.get("kind") == "average"
series = [
    ("Tabular regrets (same source)", "#009e73", "-", rows("cfr_plus_18_gpu_fit_tabular_cumulative_exact_20260930.jsonl")),
    ("GPU S96", "#0072b2", "-", [(SOURCE_ITERATION, cpu[0][1])] + rows("cfr_plus_18_gpu_fit_s96_exact_20260930.jsonl", average)),
    ("GPU S384", "#d55e00", "-", [(SOURCE_ITERATION, cpu[0][1])] + rows("cfr_plus_18_gpu_fit_s384_exact_20260930.jsonl", average)),
    ("CPU S24 (continued trunk)", "#8a93a0", "--", cpu),
]

fig, ax = plt.subplots(figsize=(10, 5.2), layout="constrained")
for label, color, style, points in series:
    x, y = zip(*points)
    ax.plot(x, y, linestyle="none", marker="o", markersize=3.5, color=color, alpha=0.35,
            markeredgewidth=0)
    wx, wy = window_means(points)
    ax.plot(wx, wy, linestyle=style, linewidth=2, color=color, marker="o", markersize=5,
            markeredgecolor="white", markeredgewidth=1.2, label=label, solid_capstyle="round")
    # Direct labels only for the two lines the comparison is about; the legend carries all four.
    if label in {"GPU S96", "Tabular regrets (same source)"}:
        ax.annotate(label, (wx[-1], wy[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=8.5, color=INK)
    if label == "Tabular regrets (same source)":
        best = min(points, key=lambda point: point[1])
        latest = points[-1]
        ax.scatter(*best, marker="*", s=115, color="#e69f00", edgecolor="white",
                   linewidth=0.8, zorder=6, label="Table best snapshot")
        ax.scatter(*latest, marker="D", s=48, color="#009e73", edgecolor="white",
                   linewidth=0.8, zorder=6, label="Table final snapshot")
        ax.annotate(f"best {best[1]:.4f}", best, xytext=(7, -13),
                    textcoords="offset points", fontsize=8, color=INK)
        ax.annotate(f"final {latest[1]:.4f}", latest, xytext=(7, 5),
                    textcoords="offset points", fontsize=8, color=INK)
ax.axvline(SOURCE_ITERATION, color=MUTED, linewidth=1, linestyle=":")
ax.text(SOURCE_ITERATION, 0.00235, " shared source checkpoint, iteration 15,728",
        color=MUTED, fontsize=8, va="bottom")
ax.set_yscale("log")
ax.set_ylim(0.0022, 0.011)
ax.set_xlim(14_000, 84_000)
ax.set_yticks([0.0025, 0.003, 0.004, 0.005, 0.006, 0.008, 0.01])
ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.4f}".rstrip("0")))
ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v/1000:.0f}k"))
ax.grid(color=GRID, linewidth=0.8, which="major")
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
for side in ("left", "bottom"):
    ax.spines[side].set_color(GRID)
ax.tick_params(colors=MUTED, labelsize=9)
ax.set_xlabel("CFR+ iteration", color=INK)
ax.set_ylabel("Exact average-policy exploitability (log scale)", color=INK)
ax.set_title("Same checkpoint, four continuations: faint dots are snapshots, lines are window geometric means",
             color=INK, fontsize=10, loc="left")
ax.legend(frameon=False, fontsize=9, loc="upper right")
fig.savefig(OUT, dpi=160)
print("wrote", OUT)
