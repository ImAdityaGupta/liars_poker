#!/usr/bin/env python3
"""Plot the first 30-claim run: precise depth-2 expectimax exploitability of O4 snapshots, with June's policies."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs" / "data" / "cfr_plus_30_claim_first_run_20261003" / "precise_summary.json"
OUT = ROOT / "docs" / "figures" / "experiment_cfr_plus_30_claim_first_run.png"
INK, GRID = "#1f2430", "#e6e8ec"
ARMS = [("w512", "W512 (regret 512×512), O4", "#2563EB", -3), ("w2048", "W2048 (regret 2048×2048), O4", "#D97706", 3)]
JUNE = "#6B7280"


def style(ax) -> None:
    ax.grid(True, which="both", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def main() -> None:
    rows = json.loads(DATA.read_text(encoding="utf-8"))
    fig, (left, right) = plt.subplots(1, 2, figsize=(15, 5.6), layout="constrained")
    for arm, label, color, shift in ARMS:
        sel = sorted((r for r in rows if r["arm"] == arm and r["kind"] == "o4"), key=lambda r: r["snapshot"])
        x = [int(r["snapshot"][:-1]) + shift for r in sel]
        y = [r["discovered_exploitability"] for r in sel]
        e = [r["half_width_95"] for r in sel]
        left.errorbar(x, y, yerr=e, color=color, marker="o", markersize=4, capsize=2, linewidth=1.8, label=label)
        for ax_, key, tag in ((right, "p_first", "first-seat responder"), (right, "p_second", "second-seat responder")):
            ax_.plot(x, [r[key] for r in sel], color=color, linestyle="-" if key == "p_first" else "--",
                     marker="o", markersize=3, label=f"{arm}: {tag}")
    june = [r for r in rows if r["kind"] == "june"]
    for i, r in enumerate(june):
        left.errorbar(60, r["discovered_exploitability"], yerr=r["half_width_95"], color=JUNE, marker="s",
                      markersize=5, capsize=2, linestyle="none", label="June runs, 60 min (online average)" if i == 0 else None)
        right.scatter([60, 60], [r["p_first"], r["p_second"]], color=JUNE, marker="s", s=18,
                      label="June runs, 60 min" if i == 0 else None)
    left.set(xlabel="Measured training minutes", ylabel="Exploitability found (depth-2 expectimax; lower bound)",
             ylim=(0, 0.08))
    left.set_title("Depth-2 expectimax, 60,900 games per seat (bars: 95% interval)", loc="left", color=INK,
                   fontsize=10.5)
    right.axhline(0.5, color=INK, linewidth=0.8, linestyle=":")
    right.set(xlabel="Measured training minutes", ylabel="Responder win probability")
    right.set_title("The two seats separately (exploitability = first + second − 1)", loc="left", color=INK,
                    fontsize=10.5)
    left.legend(frameon=False, fontsize=9)
    right.legend(frameon=False, fontsize=8, ncol=2)
    for ax in (left, right):
        style(ax)
    fig.suptitle("30-claim first run (r5_s4_h3_hp2ptq_ss): O4 snapshots against June's baseline", color=INK)
    fig.savefig(OUT, dpi=150)
    print(OUT)


if __name__ == "__main__":
    main()
