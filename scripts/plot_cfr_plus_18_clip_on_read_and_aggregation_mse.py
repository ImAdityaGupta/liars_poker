#!/usr/bin/env python3
"""Plot the 18-claim plain-MSE target arms (clip on read, hybrid, aggregate then clip) against cumulative conditional."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs" / "data"
FIGURES = ROOT / "docs" / "figures"
INK, GRID = "#1f2430", "#e6e8ec"
WINDOWS = [(1_000, 2_000), (2_000, 4_000), (4_000, 6_000), (6_000, 9_000), (9_000, 12_000), (12_000, 15_000)]

# Colours match the 8765 dashboard (extra_arms.json and the cumulative-regret manifest).
ARMS = {
    "conditional4096": ("Aggregate then clip, weighted MSE (cumulative conditional)", "#C27600"),
    "aggregate_plain_mse4096": ("Aggregate then clip, plain MSE", "#EC4FA6"),
    "hybrid_plain_mse4096": ("Hybrid: aggregate signed, clip on read, plain MSE", "#7B2CBF"),
    "clip_on_read4096": ("Clip on read, plain MSE", "#D7191C"),
}
REPLICATE = ("Aggregate then clip, plain MSE: rerun (online average of the O4 run)", "#EC4FA6")


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load() -> dict[str, np.ndarray]:
    """Rows of (iteration, measured minutes, exploitability) per arm."""
    out: dict[str, list] = {}
    for r in read(DATA / "cfr_plus_18_clip_on_read_and_aggregation_mse" / "live_exact_plain_mse_arms.jsonl"):
        out.setdefault(r["arm"], []).append((r["iteration"], r["snapshot_min"], r["exploitability"]))
    out["conditional4096"] = [(r["iteration"], r["snapshot_min"], r["exploitability"])
                              for r in read(DATA / "cfr_plus_18_cumulative_conditional4096_full_20260930.jsonl")]
    out["replicate"] = [(r["iteration"], r["measured_training_min"], r["exploitability"])
                        for r in read(DATA / "cfr_plus_18_neural_o4_cpu_20261001" / "neural_o4_k4096.jsonl")
                        if r["policy_kind"] == "online"]
    return {name: np.array(sorted(rows)) for name, rows in out.items()}


def style(ax, xlabel: str) -> None:
    ax.set(xlabel=xlabel, ylabel="Exact exploitability (online average)", yscale="log")
    ax.grid(True, which="both", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plot_curves(data: dict[str, np.ndarray]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4), layout="constrained", sharey=True)
    for name, (label, color) in ARMS.items():
        d = data[name]
        d = d[d[:, 1] <= 600]  # cumulative conditional ran to 1,200 minutes; the new arms stop at 600
        for ax, column in zip(axes, (1, 0)):
            ax.plot(d[:, column], d[:, 2], color=color, linewidth=1.8, marker="o", markersize=3, label=label)
    style(axes[0], "Measured training minutes")
    axes[0].set_title("By measured training time", loc="left", color=INK)
    style(axes[1], "CFR+ iteration")
    axes[1].set_title("By iteration", loc="left", color=INK)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=2, frameon=False, fontsize=9.5)
    fig.suptitle("18 claims, 4,096 roots: regret-target construction (exact snapshots every 15 minutes)", color=INK)
    fig.savefig(FIGURES / "experiment_cfr_plus_18_clip_on_read_and_aggregation_mse.png", dpi=150)
    plt.close(fig)


def window_means(d: np.ndarray) -> tuple[list[float], list[float]]:
    xs, ys = [], []
    for lo, hi in WINDOWS:
        inside = d[(d[:, 0] >= lo) & (d[:, 0] < hi)]
        if len(inside) >= 2:
            xs.append(float(np.exp(np.log(inside[:, 0]).mean())))
            ys.append(float(np.exp(np.log(inside[:, 2]).mean())))
    return xs, ys


def plot_windows(data: dict[str, np.ndarray]) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.4), layout="constrained")
    for name, (label, color) in ARMS.items():
        x, y = window_means(data[name])
        ax.plot(x, y, color=color, linewidth=2.2, marker="o", markersize=6, label=label)
    x, y = window_means(data["replicate"])
    ax.plot(x, y, color=REPLICATE[1], linewidth=1.6, linestyle="--", marker="o", markersize=5,
            markerfacecolor="white", label=REPLICATE[0])
    style(ax, "CFR+ iteration (log; geometric centre of each window)")
    ax.set_xscale("log")
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("Geometric mean exploitability within iteration windows "
                 "(1–2k, 2–4k, 4–6k, 6–9k, 9–12k, 12–15k)", loc="left", color=INK, fontsize=10.5)
    fig.savefig(FIGURES / "experiment_cfr_plus_18_clip_on_read_and_aggregation_mse_windows.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    data = load()
    plot_curves(data)
    plot_windows(data)
