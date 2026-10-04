#!/usr/bin/env python3
"""Plot the long-run O4 check: refit quality on exact4096 checkpoints from 908 to 31,538 iterations."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs" / "data"
LONG = DATA / "cfr_plus_18_average_fit_long_run_check_20261002" / "results.jsonl"
CHAIN = DATA / "cfr_plus_18_average_fit_chain_check_20261003" / "results.jsonl"
SCHEDULES = DATA / "cfr_plus_18_average_fit_schedules_20261001"
OPTIMIZER = DATA / "cfr_plus_18_average_fit_optimizer"
TABLE = DATA / "cfr_plus_18_batched_bridge_controls_20260930" / "exact4096.jsonl"
NEURAL = DATA / "cfr_plus_18_neural_o4_cpu_20261001" / "neural_o4_k4096.jsonl"
FIG = ROOT / "docs" / "figures"
INK, GRID = "#1f2430", "#e6e8ec"
# Part A colours for the refit families; 8768 dashboard colours for the runs.
O4_COLOR, FRESH_COLOR, X_COLOR, CHAIN_COLOR = "#2563EB", "#7C3AED", "#111827", "#D97706"
TABLE_COLOR, NEURAL_COLOR = "#111827", "#0D9488"

CHECKPOINTS = {"0030m": 908, "0045m": 1_424, "0120m": 3_988}
EXACT = {"0030m": 0.005249, "0045m": 0.004408, "0120m": 0.002836, "1080m": 0.001005141776837748}
ITERATION = {**CHECKPOINTS, "1080m": 31_538}


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def final(root: Path, checkpoint: str, arm: str, steps: int) -> float | None:
    path = root / checkpoint / arm / "results.jsonl"
    if not path.exists():
        return None
    rows = [r for r in read(path) if r.get("refit_steps") == steps]
    return rows[-1]["exploitability"] if rows else None


def collect() -> dict[str, dict[str, list[float]]]:
    """Exploitability of every final fit, by family and checkpoint."""
    out = {"O4": {}, "F40k": {}, "X": {}, "Chain": {}}
    for checkpoint in CHECKPOINTS:
        out["O4"][checkpoint] = [v for arm in ("O4_cosine_b16384_ce", "O4_cosine_b16384_ce_seed17032",
                                               "O4_cosine_b16384_ce_seed17033")
                                 if (v := final(OPTIMIZER, checkpoint, arm, 5_000)) is not None]
        out["F40k"][checkpoint] = [v for arm in ("F40k", "F40k_seed17032", "F40k_seed17033")
                                   if (v := final(SCHEDULES, checkpoint, arm, 40_000)) is not None]
        out["X"][checkpoint] = [v for v in [final(SCHEDULES, checkpoint, "X", 40_000)] if v is not None]
    long_rows = read(LONG)
    out["O4"]["1080m"] = [r["exploitability"] for r in long_rows if r["arm"].startswith("O4")]
    out["F40k"]["1080m"] = [r["exploitability"] for r in long_rows
                            if r["arm"] == "F40k" and r["refit_steps"] == 40_000]
    # Chained O4: each link starts from the previous checkpoint's 5k refit (two fit seeds).
    for r in read(CHAIN):
        if r["arm"].startswith("CH_hi_5k"):
            out["Chain"].setdefault(r["checkpoint"], []).append(r["exploitability"])
    return out


def style(ax) -> None:
    ax.grid(True, which="both", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plot_averaging(fits) -> None:
    table = sorted((r for r in read(TABLE) if r.get("exploitability", 0) > 0), key=lambda r: r["iteration"])
    families = [("O4", "O4: warm, 5k steps", O4_COLOR, "o", -0.03),
                ("F40k", "F40k: fresh, 40k steps", FRESH_COLOR, "D", 0.03),
                ("X", "X: distil the exact average (Part A only)", X_COLOR, "*", 0.0),
                ("Chain", "Chained O4: start from the previous checkpoint's refit", CHAIN_COLOR, "s", 0.015)]
    fig, (left, right) = plt.subplots(1, 2, figsize=(14, 5.4), layout="constrained")
    left.plot([r["iteration"] for r in table], [r["exploitability"] for r in table], color=TABLE_COLOR,
              linewidth=1.6, label="Exact average of the table run (exact4096)")
    for key, label, color, marker, shift in families:
        for ax, scale in ((left, None), (right, EXACT)):
            xs, means = [], []
            for checkpoint, values in fits[key].items():
                if not values:
                    continue
                x = ITERATION[checkpoint] * 10 ** shift
                ys = [v / scale[checkpoint] for v in values] if scale else values
                ax.scatter([x] * len(ys), ys, color=color, marker=marker, s=90 if marker == "*" else 34,
                           alpha=0.85, zorder=3)
                xs.append(x)
                means.append(float(np.mean(ys)))
            ax.plot(xs, means, color=color, linewidth=1.8, label=label if ax is left else None)
    right.axhline(1.0, color=INK, linestyle="--", linewidth=1.2)
    right.text(12_000, 0.95, "exact average", color=INK, fontsize=9)
    right.annotate("F40k: 2.30×", (31_538 * 10 ** 0.03, 2.30), xytext=(-95, -4), textcoords="offset points",
                   color=FRESH_COLOR, fontsize=10, fontweight="bold")
    right.annotate("O4: 1.24× (mean)", (31_538 * 10 ** -0.03, 1.24), xytext=(-115, 10), textcoords="offset points",
                   color=O4_COLOR, fontsize=10, fontweight="bold")
    left.set(xscale="log", yscale="log", xlabel="Source checkpoint iteration (log)", ylabel="Exact exploitability")
    left.set_title("Refits against the exact average they approximate", loc="left", color=INK, fontsize=11)
    right.set(xscale="log", xlabel="Source checkpoint iteration (log)", ylabel="Refit exploitability ÷ exact average",
              ylim=(0.9, 2.5))
    right.set_title("Ratio to the exact average: O4 holds, a fresh fit does not", loc="left", color=INK, fontsize=11)
    for ax in (left, right):
        ax.set_xticks([1_000, 4_000, 10_000, 31_538], ["1k", "4k", "10k", "31.5k"])
        style(ax)
    fig.legend(*left.get_legend_handles_labels(), loc="outside lower center", ncol=3, frameon=False, fontsize=9.5)
    fig.suptitle("Long-run averaging check on exact4096 (dots: individual fit seeds; lines: seed means)", color=INK)
    fig.savefig(FIG / "experiment_cfr_plus_18_average_fit_long_run_check.png", dpi=150)
    plt.close(fig)


def plot_neural(fits) -> None:
    table = sorted((r for r in read(TABLE) if r.get("exploitability", 0) > 0), key=lambda r: r["iteration"])
    neural = sorted((r for r in read(NEURAL) if r.get("policy_kind") == "o4"), key=lambda r: r["iteration"])
    ratios = [v / EXACT[c] for c, values in fits["O4"].items() for v in values]
    low, mean, high = min(ratios), float(np.mean(ratios)), max(ratios)
    x = np.array([r["iteration"] for r in table])
    y = np.array([r["exploitability"] for r in table])
    fig, ax = plt.subplots(figsize=(10, 5.4), layout="constrained")
    ax.fill_between(x, y * low, y * high, color=TABLE_COLOR, alpha=0.12, linewidth=0,
                    label=f"Table averaged with O4: exact × {low:.2f}–{high:.2f} (all 12 O4 fits)")
    ax.plot(x, y, color=TABLE_COLOR, linewidth=1.6, label="Table + exact average (exact4096)")
    ax.plot(x, y * mean, color=TABLE_COLOR, linewidth=1.0, linestyle="--", label=f"Table averaged with O4, mean ratio {mean:.2f}")
    ax.plot([r["iteration"] for r in neural], [r["exploitability"] for r in neural], color=NEURAL_COLOR,
            marker="o", markersize=3, linewidth=2, label="Neural regrets + O4 (neural_o4_k4096)")
    best = min(neural, key=lambda r: r["exploitability"])
    ax.annotate(f"best {best['exploitability']:.4f}\nat {best['iteration']:,}", (best["iteration"], best["exploitability"]),
                xytext=(-30, 40), textcoords="offset points", color=NEURAL_COLOR, fontsize=9,
                arrowprops=dict(arrowstyle="-", color=NEURAL_COLOR, lw=0.8))
    last = neural[-1]
    ax.annotate(f"{last['exploitability']:.4f}\nat {last['iteration']:,}", (last["iteration"], last["exploitability"]),
                xytext=(8, -6), textcoords="offset points", color=NEURAL_COLOR, fontsize=9)
    ax.set(xscale="log", yscale="log", xlabel="CFR+ iteration (log)", ylabel="Exact exploitability")
    ax.set_title("The neural rise is not the averaging: a table averaged with O4 stays in the grey band",
                 loc="left", color=INK, fontsize=11)
    style(ax)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    fig.savefig(FIG / "experiment_cfr_plus_18_average_fit_long_run_neural.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    fits = collect()
    for family, by_checkpoint in fits.items():
        for checkpoint, values in by_checkpoint.items():
            if values and checkpoint in EXACT:
                print(family, checkpoint, [round(v / EXACT[checkpoint], 3) for v in values])
    plot_averaging(fits)
    plot_neural(fits)
