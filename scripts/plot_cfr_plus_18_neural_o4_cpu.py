#!/usr/bin/env python3
"""Plot the neural O4 CPU runs against the batched tabular-regret controls (the regret-storage x roots 2x2)."""

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
INK, MUTED, GRID = "#1f2430", "#5d6b80", "#e6e8ec"

# Colours match the 8768 dashboard.
RUNS = {
    "exact4096": ("Table regrets, K=4,096, exact average", "#111827",
                  "cfr_plus_18_batched_bridge_controls_20260930/exact4096.jsonl", None),
    "neural1024": ("Table regrets, K=1,024, online neural average", "#8B5CF6",
                   "cfr_plus_18_batched_bridge_controls_20260930/neural1024.jsonl", None),
    "o4_k4096": ("Neural regrets, K=4,096, O4 refit", "#0D9488",
                 "cfr_plus_18_neural_o4_cpu_20261001/neural_o4_k4096.jsonl", "o4"),
    "o4_k1024": ("Neural regrets, K=1,024, O4 refit", "#E11D48",
                 "cfr_plus_18_neural_o4_cpu_20261001/neural_o4_k1024.jsonl", "o4"),
    "online_k4096": ("Neural regrets, K=4,096, online average", "#0D9488",
                     "cfr_plus_18_neural_o4_cpu_20261001/neural_o4_k4096.jsonl", "online"),
    "online_k1024": ("Neural regrets, K=1,024, online average", "#E11D48",
                     "cfr_plus_18_neural_o4_cpu_20261001/neural_o4_k1024.jsonl", "online"),
}


def load(name: str) -> np.ndarray:
    """Rows of (iteration, measured minutes, exploitability), sorted by iteration."""
    _, _, path, kind = RUNS[name]
    rows = [json.loads(line) for line in (DATA / path).read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [r for r in rows if kind is None or r.get("policy_kind") == kind]
    return np.array(sorted((r["iteration"], r["measured_training_min"], r["exploitability"]) for r in rows))


def style(ax, xlabel: str) -> None:
    ax.set(xlabel=xlabel, ylabel="Exact exploitability", yscale="log")
    ax.grid(True, which="both", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plot_overview() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), layout="constrained")
    for name, (label, color, _, kind) in RUNS.items():
        data = load(name)
        online = kind == "online"
        kw = dict(color=color, linewidth=1.2 if online else 2.2, linestyle=":" if online else "-",
                  alpha=0.55 if online else 1.0, marker=None if online else "o", markersize=3, label=label)
        axes[0].plot(data[:, 0], data[:, 2], **kw)
        axes[1].plot(data[:, 1], data[:, 2], **kw)
    style(axes[0], "CFR+ iteration (log)")
    axes[0].set_xscale("log")
    axes[0].set_title("By iteration", loc="left", color=INK)
    style(axes[1], "Measured training minutes")
    axes[1].set_title("By measured training time (excludes O4 refit cost)", loc="left", color=INK)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=3, frameon=False,
               fontsize=9)
    fig.suptitle("18 claims: regret storage × roots per player (seed 17)", color=INK)
    fig.savefig(FIGURES / "experiment_cfr_plus_18_neural_o4_cpu_overview.png", dpi=150)
    plt.close(fig)


def log_interp(x: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    out = np.exp(np.interp(np.log(x), np.log(xs), np.log(ys), left=np.nan, right=np.nan))
    return out


def plot_ratios() -> None:
    """The two clean comparisons: storage at fixed roots, and roots at fixed (O4) averaging."""
    exact = load("exact4096")
    o4_4096, o4_1024 = load("o4_k4096"), load("o4_k1024")
    on_4096, on_1024 = load("online_k4096"), load("online_k1024")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), layout="constrained")

    ax = axes[0]
    ax.plot(o4_4096[:, 0], o4_4096[:, 2] / log_interp(o4_4096[:, 0], exact[:, 0], exact[:, 2]),
            color="#0D9488", linewidth=2.2, marker="o", markersize=3,
            label="Neural regrets + O4  ÷  table regrets + exact average (both K=4,096)")
    ax.plot(o4_1024[:, 0], o4_1024[:, 2] / log_interp(o4_1024[:, 0], o4_4096[:, 0], o4_4096[:, 2]),
            color="#E11D48", linewidth=2.2, marker="o", markersize=3,
            label="Neural regrets + O4: K=1,024  ÷  K=4,096")
    ax.axhline(1, color=MUTED, linewidth=1)
    ax.set(xlabel="CFR+ iteration (log)", ylabel="Exploitability ratio at matched iteration", xscale="log",
           ylim=(0.8, 2.8))
    ax.set_title("Clean comparisons at matched iteration", loc="left", color=INK)
    ax.grid(True, which="both", color=GRID, linewidth=0.8)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")

    ax = axes[1]
    for online, o4, color, label in ((on_4096, o4_4096, "#0D9488", "K=4,096"),
                                     (on_1024, o4_1024, "#E11D48", "K=1,024")):
        common = np.intersect1d(online[:, 0], o4[:, 0])
        ratio = [online[online[:, 0] == i, 2][0] / o4[o4[:, 0] == i, 2][0] for i in common]
        ax.plot(o4[np.isin(o4[:, 0], common), 1], ratio, color=color, linewidth=2.2, marker="o",
                markersize=3, label=label)
    ax.axhline(1, color=MUTED, linewidth=1)
    ax.set(xlabel="Measured training minutes", ylabel="Online ÷ O4 exploitability (same snapshot)",
           ylim=(0.8, None))
    ax.set_title("What O4 averaging buys on the same trajectory", loc="left", color=INK)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.legend(frameon=False, fontsize=9)
    for a in axes:
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
    fig.savefig(FIGURES / "experiment_cfr_plus_18_neural_o4_cpu_ratios.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    plot_overview()
    plot_ratios()
