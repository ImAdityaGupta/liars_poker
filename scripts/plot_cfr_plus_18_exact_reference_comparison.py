#!/usr/bin/env python3
"""Plot exact full-tree CFR+ against sampled K=4,096 runs with table and neural regrets, by iteration."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs" / "data"
OUT = ROOT / "docs" / "figures" / "experiment_cfr_plus_18_exact_reference_comparison.png"
INK, GRID = "#1f2430", "#e6e8ec"

# (label, colour, data file, policy_kind filter, line style)
SERIES = [
    ("Exact full-tree CFR+ (January 2026)", "#6B7280", "cfr_plus_18_exact_full_tree_reference_20260108.jsonl", None, "reference"),
    ("K=4,096: table regrets, exact average", "#111827", "cfr_plus_18_batched_bridge_controls_20260930/exact4096.jsonl", None, "main"),
    ("K=4,096: neural regrets (plain MSE), O4 refit", "#0D9488", "cfr_plus_18_neural_o4_cpu_20261001/neural_o4_k4096.jsonl", "o4", "main"),
    ("K=4,096: neural regrets (weighted MSE), online average [8765]", "#C27600", "cfr_plus_18_cumulative_conditional4096_full_20260930.jsonl", None, "main"),
    ("K=4,096: neural regrets (plain MSE), online average of the O4 run", "#0D9488", "cfr_plus_18_neural_o4_cpu_20261001/neural_o4_k4096.jsonl", "online", "faint"),
]
STYLES = {
    "reference": dict(linestyle="--", linewidth=1.8, marker=None),
    "main": dict(linestyle="-", linewidth=2.2, marker="o", markersize=3),
    "faint": dict(linestyle=":", linewidth=1.4, marker=None, alpha=0.7),
}


def load(path: str, kind: str | None) -> tuple[list[int], list[float]]:
    rows = [json.loads(line) for line in (DATA / path).read_text(encoding="utf-8").splitlines() if line.strip()]
    points = sorted((r["iteration"], r["exploitability"]) for r in rows
                    if kind is None or r.get("policy_kind") == kind)
    return [p[0] for p in points], [p[1] for p in points]


fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), layout="constrained")
for label, color, path, kind, style in SERIES:
    x, y = load(path, kind)
    for ax in axes:
        ax.plot(x, y, color=color, label=label, **STYLES[style])
for ax, scale, title in ((axes[0], "linear", "Iteration (linear)"), (axes[1], "log", "Iteration (log)")):
    ax.set(xscale=scale, yscale="log", xlabel="CFR+ iteration", ylabel="Exact exploitability")
    ax.set_title(title, loc="left", color=INK)
    ax.grid(True, which="both", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
axes[0].set_xlim(0, 31_000)
fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=2, frameon=False, fontsize=9)
fig.suptitle("18 claims: exact CFR+ against sampled K=4,096 runs (average policy)", color=INK)
fig.savefig(OUT, dpi=150)
