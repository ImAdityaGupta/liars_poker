#!/usr/bin/env python3
"""Plot the completed 18-claim traversal-root schedule experiment."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs" / "data" / "cfr_plus_18_root_schedules_20261002"
REFERENCE = ROOT / "docs" / "data" / "cfr_plus_18_batched_bridge_controls_20260930" / "exact4096.jsonl"
OUT = ROOT / "docs" / "figures" / "experiment_cfr_plus_18_root_schedules.png"

# Colours match the 8769 dashboard (monitor_cfr_plus_18_tabular_discount.py).
CONSTANT = [
    ("k0256", "Constant K=256", "#2563EB"),
    ("k0512", "Constant K=512", "#D97706"),
    ("k1024", "Constant K=1,024", "#059669"),
    ("k2048", "Constant K=2,048", "#DB2777"),
    ("k8192", "Constant K=8,192", "#7C3AED"),
    ("k16384", "Constant K=16,384", "#DC2626"),
]
DYNAMIC = [
    ("ramp_up", "Ramp up: 512 → 7,680 by 540 min → 32,768 by 1,140 min", "#0891B2"),
    ("ramp_down", "Ramp down: 7,680 to 512", "#B45309"),
    ("step_late", "Step late: 512, then 7,680 from 270 min", "#4D7C0F"),
    ("step_early", "Step early: 7,680, then 512 from 270 min", "#BE185D"),
]


def read(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("exploitability", 0) > 0:
            rows.append(row)
    return rows


def main() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.8), layout="constrained")
    for arm, label, color in CONSTANT + DYNAMIC:
        rows = sorted(read(DATA / arm / "evaluations.jsonl"), key=lambda r: r["iteration"])
        if not rows:
            continue
        linestyle = "-" if arm.startswith("k") else "--"
        marker = "o" if arm.startswith("k") else "s"
        for ax, xkey in zip(axes, ("measured_training_min", "iteration", "cumulative_roots_per_player")):
            ax.plot([r[xkey] for r in rows], [r["exploitability"] for r in rows],
                    color=color, linestyle=linestyle, marker=marker,
                    markersize=3, linewidth=1.8, label=label)

    reference = read(REFERENCE)
    if reference:
        xsets = (
            [r["measured_training_min"] for r in reference],
            [r["iteration"] for r in reference],
            [r["iteration"] * 4096 for r in reference],  # constant K per iteration
        )
        for ax, xs in zip(axes, xsets):
            ax.plot(xs, [r["exploitability"] for r in reference], color="#111827",
                    linestyle="--", linewidth=2.1, label="K=4,096 exact-average reference")

    for ax, title, xlabel in zip(
        axes,
        ("Equal measured training time", "Equal CFR iteration", "Equal cumulative roots"),
        ("Measured training minutes", "CFR iteration", "Cumulative roots per player"),
    ):
        ax.set(title=title, xlabel=xlabel, ylabel="Exact exploitability", yscale="log")
        ax.grid(True, which="both", color="#e4e8ee", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle("18-claim tabular CFR+: root count and measured-time schedules", fontsize=15)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4,
               frameon=False, fontsize=8.5)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=160)
    print(OUT)


if __name__ == "__main__":
    main()
