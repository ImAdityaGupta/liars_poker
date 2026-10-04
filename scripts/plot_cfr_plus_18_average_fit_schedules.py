#!/usr/bin/env python3
"""Plot Part A of the 18-claim average-fit schedule experiment: per-arm final values and quality against schedule length."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs" / "data"
SCHEDULES = DATA / "cfr_plus_18_average_fit_schedules_20261001"
OPTIMIZER = DATA / "cfr_plus_18_average_fit_optimizer"
OUT = ROOT / "docs" / "figures" / "experiment_cfr_plus_18_average_fit_schedules_length.png"
SUMMARY_OUT = ROOT / "docs" / "figures" / "experiment_cfr_plus_18_average_fit_schedules_summary.png"
INK, GRID = "#1f2430", "#e6e8ec"

CHECKPOINTS = {"0030m": 908, "0045m": 1_424, "0120m": 3_988}
# Exact accumulated average at each checkpoint (optimizer-and-objective study).
EXACT = {"0030m": 0.005249, "0045m": 0.004408, "0120m": 0.002836}

# (label, colour, marker, [(steps, data root, arm directory)])
FAMILIES = [
    ("Warm, cosine 1e-3 → 1e-5", "#2563EB", "o",
     [(500, SCHEDULES, "W500"), (1_000, SCHEDULES, "W1k"), (2_000, SCHEDULES, "W2k"),
      (5_000, OPTIMIZER, "O4_cosine_b16384_ce")]),
    ("Warm, lower peak 3e-4 → 1e-6", "#D97706", "s",
     [(1_000, SCHEDULES, "L1k"), (5_000, SCHEDULES, "L5k")]),
    ("Fresh, cosine 1e-3 → 1e-5", "#7C3AED", "D",
     [(20_000, OPTIMIZER, "FRESH_O4_cosine_b16384_ce"), (40_000, SCHEDULES, "F40k"),
      (80_000, SCHEDULES, "F80k")]),
    ("Fresh, distilling the exact average (X)", "#111827", "*",
     [(40_000, SCHEDULES, "X")]),
]


def final_ratios(root: Path, arm: str, checkpoint: str) -> list[float]:
    """Final exploitability / exact average for the arm and any extra fit seeds."""
    out = []
    for name in (arm, f"{arm}_seed17032", f"{arm}_seed17033"):
        path = root / checkpoint / name / "results.jsonl"
        if path.exists():
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            out.append(max(rows, key=lambda r: r["refit_steps"])["exploitability"] / EXACT[checkpoint])
    return out


def fit_seconds(root: Path, arm: str, checkpoint: str) -> float:
    path = root / checkpoint / arm / "results.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return sum(r.get("fit_s_increment", 0.0) for r in rows)


def plot_summary() -> None:
    """Final exploitability / exact average for every arm, with extra-seed replicates, ordered by GPU fit time."""
    arms = [("W500", SCHEDULES), ("L1k", SCHEDULES), ("W1k", SCHEDULES), ("W2k", SCHEDULES),
            ("L5k", SCHEDULES), ("O4_cosine_b16384_ce", OPTIMIZER), ("R5k", SCHEDULES),
            ("F40k", SCHEDULES), ("F80k", SCHEDULES), ("X", SCHEDULES)]
    names = {"O4_cosine_b16384_ce": "O4"}
    markers = {"0030m": "o", "0045m": "s", "0120m": "D"}
    colors = {"0030m": "#2563EB", "0045m": "#D97706", "0120m": "#059669"}
    fig, ax = plt.subplots(figsize=(12, 5.4), layout="constrained")
    labels = []
    for i, (arm, root) in enumerate(arms):
        seconds = []
        for j, checkpoint in enumerate(CHECKPOINTS):
            offset = (j - 1) * 0.18
            ratios = final_ratios(root, arm, checkpoint)
            seconds.append(fit_seconds(root, arm, checkpoint))
            ax.scatter(i + offset, ratios[0], marker=markers[checkpoint], s=46, color=colors[checkpoint], zorder=3,
                       label=f"{checkpoint} (iteration {CHECKPOINTS[checkpoint]:,})" if i == 0 else None)
            for extra in ratios[1:]:
                ax.scatter(i + offset, extra, marker=markers[checkpoint], s=46, facecolors="none",
                           edgecolors=colors[checkpoint], zorder=3)
        median = np.median(seconds)
        labels.append(f"{names.get(arm, arm)}\n{median:.0f} s" if median >= 10 else f"{names.get(arm, arm)}\n{median:.1f} s")
    ax.axhline(1.0, color=INK, linestyle="--", linewidth=1.2)
    ax.text(-0.4, 0.995, "exact average", color=INK, fontsize=9, va="top", ha="left")
    ax.set_xticks(range(len(arms)), labels)
    ax.set(ylabel="Final exploitability ÷ exact average", ylim=(0.95, 1.45))
    ax.set_xlabel("Arm, ordered by fit cost  (median GPU fit seconds per checkpoint, both players)")
    ax.grid(True, axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="upper right", title="Filled: first fit seed; hollow: extra seeds",
              title_fontsize=8.5)
    fig.suptitle("Part A: how close each refit gets to the exact average", color=INK)
    fig.savefig(SUMMARY_OUT, dpi=150)
    plt.close(fig)


plot_summary()
fig, axes = plt.subplots(1, 3, figsize=(15, 5), layout="constrained", sharey=True)
for ax, checkpoint in zip(axes, CHECKPOINTS):
    for label, color, marker, points in FAMILIES:
        steps, means, lows, highs = [], [], [], []
        for n, root, arm in points:
            ratios = final_ratios(root, arm, checkpoint)
            steps.append(n)
            means.append(np.mean(ratios))
            lows.append(np.mean(ratios) - min(ratios))
            highs.append(max(ratios) - np.mean(ratios))
        ax.errorbar(steps, means, yerr=[lows, highs], color=color, marker=marker,
                    markersize=11 if marker == "*" else 6, linewidth=2, capsize=3, label=label)
        # O4 is the 5k warm-cosine point; name it so it can be found on the plot.
        if points[-1][2] == "O4_cosine_b16384_ce":
            ax.annotate("O4", (steps[-1], means[-1]), xytext=(7, -3), textcoords="offset points",
                        color=color, fontsize=9.5, fontweight="bold")
    ax.axhline(1.0, color=INK, linestyle="--", linewidth=1.2, label="Exact accumulated average")
    ax.set(xscale="log", xlabel="Schedule length: refit steps per player (log)", ylim=(0.95, 1.45))
    ax.set_title(f"{checkpoint} checkpoint (iteration {CHECKPOINTS[checkpoint]:,})", loc="left", color=INK,
                 fontsize=10.5)
    ax.grid(True, which="both", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
axes[0].set_ylabel("Final exploitability ÷ exact average")
fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=5, frameon=False, fontsize=9.5)
fig.suptitle("Part A: final refit quality against schedule length (bars: range over three fit seeds)", color=INK)
fig.savefig(OUT, dpi=150)
