#!/usr/bin/env python3
"""Plot the Part B follow-up: root schedules with O4 averaging, by time, iteration and cumulative roots."""

from __future__ import annotations

import bisect
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs" / "data" / "cfr_plus_18_root_o4_followup_20261003"
PART_B_RAMP = ROOT / "docs" / "data" / "cfr_plus_18_root_schedules_20261002" / "ramp_up" / "evaluations.jsonl"
OUT = ROOT / "docs" / "figures" / "experiment_cfr_plus_18_root_o4_followup.png"
INK, GRID = "#1f2430", "#e6e8ec"
# Default matplotlib cycle in the order the 8772 follow-up dashboard lists the arms.
ARMS = [
    ("k1024", "K=1,024", "#1f77b4"),
    ("k4096", "K=4,096", "#ff7f0e"),
    ("k16384", "K=16,384", "#2ca02c"),
    ("k32768", "K=32,768", "#d62728"),
    ("ramp", "Ramp 512 → 32,768 (600 min) → 65,536 (1,200 min), 2M reservoir", "#9467bd"),
    ("ramp8m", "Ramp, 8M reservoir", "#8c564b"),
    ("ramp_exact", "Ramp-exact arm", "#e377c2"),
]
KEYS = ("measured_training_min", "iteration", "roots")


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def with_roots(arm: str, rows: list[dict]) -> list[dict]:
    training = read(DATA / arm / "training_thinned.jsonl")
    its = [r["iteration"] for r in training]
    for r in rows:
        i = min(bisect.bisect_left(its, r["iteration"]), len(training) - 1)
        r["roots"] = training[i]["cumulative_roots_per_player"]
    return rows


def main() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.2), layout="constrained")
    for arm, label, color in ARMS:
        rows = with_roots(arm, sorted(read(DATA / arm / "evaluations.jsonl"), key=lambda r: r["iteration"]))
        for kind, style, width, suffix in (("o4", "-", 2.0, ""), ("exact", "--", 1.6, ": its exact average")):
            sel = [r for r in rows if r["policy_kind"] == kind]
            if not sel:
                continue
            for ax, key in zip(axes, KEYS):
                ax.plot([r[key] for r in sel], [r["exploitability"] for r in sel], color=color, linestyle=style,
                        linewidth=width, marker="o" if kind == "o4" else None, markersize=2.5,
                        label=label + suffix)
    ref = [r for r in read(PART_B_RAMP) if r.get("exploitability", 0) > 0]
    for r in ref:
        r["roots"] = r["cumulative_roots_per_player"]
    for ax, key in zip(axes[1:], KEYS[1:]):
        ax.plot([r[key] for r in ref], [r["exploitability"] for r in ref], color=INK, linestyle=":", linewidth=2,
                label="Part B ramp, exact average (different time profile)")
    for ax, title, xlabel in zip(axes, ("Equal measured training time", "Equal iteration", "Equal cumulative roots"),
                                 ("Measured training minutes", "Iteration (log)", "Cumulative roots per player (log)")):
        ax.set(title=title, xlabel=xlabel, ylabel="Exact exploitability", yscale="log")
        if "log" in xlabel:
            ax.set_xscale("log")
        ax.grid(True, which="both", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False, fontsize=9)
    fig.suptitle("18-claim tabular CFR+: root schedules averaged with O4 (solid) — exploitability of the O4 policy",
                 color=INK)
    fig.savefig(OUT, dpi=150)
    print(OUT)


if __name__ == "__main__":
    main()
