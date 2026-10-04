#!/usr/bin/env python3
"""Plot the N/T experiment: exploitability by iteration, and each network's policy gap to its shadow table."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs" / "data"
NT = Path(sys.argv[1]) if len(sys.argv) > 1 else DATA / "cfr_plus_18_regret_bootstrap_teacher_forced_20261002"
TABLE = DATA / "cfr_plus_18_batched_bridge_controls_20260930" / "exact4096.jsonl"
CPU_NEURAL = DATA / "cfr_plus_18_neural_o4_cpu_20261001" / "neural_o4_k4096.jsonl"
OUT = ROOT / "docs" / "figures" / "experiment_cfr_plus_18_regret_bootstrap_teacher_forced.png"
INK, GRID = "#1f2430", "#e6e8ec"
# Colours match the 8770 dashboard; the CPU neural run keeps its 8768 colour.
COLORS = {"N": "#0072b2", "T": "#d55e00", "table": "#333333", "cpu": "#0D9488"}
BINS = [("ge_10", "≥10 visits per iteration", "-"), ("1_10", "1–10", "--"), ("lt_0.1", "<0.1", ":")]


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def style(ax) -> None:
    ax.grid(True, which="both", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def main() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.6), layout="constrained")
    ax = axes[0]
    table = sorted((r for r in read(TABLE) if r.get("exploitability", 0) > 0), key=lambda r: r["iteration"])
    ax.plot([r["iteration"] for r in table], [r["exploitability"] for r in table], color=COLORS["table"],
            linewidth=2, label="Table + exact average (exact4096)")
    cpu = sorted((r for r in read(CPU_NEURAL) if r.get("policy_kind") == "o4"), key=lambda r: r["iteration"])
    ax.plot([r["iteration"] for r in cpu], [r["exploitability"] for r in cpu], color=COLORS["cpu"],
            linewidth=1.4, linestyle="--", label="Earlier neural run, CPU, O4 average")
    for arm, label in (("N", "N: bootstrapped"), ("T", "T: teacher-forced")):
        rows = sorted((r for r in read(NT / arm / "evaluations.jsonl") if r["kind"] == "average"),
                      key=lambda r: r["iteration"])
        ax.plot([r["iteration"] for r in rows], [r["exploitability"] for r in rows], color=COLORS[arm],
                linewidth=2, marker="o", markersize=2.5, label=f"{label}, exact average")
    ax.set(xscale="log", yscale="log", xlabel="CFR+ iteration (log)", ylabel="Exact exploitability of the average")
    ax.set_title("Average policy", loc="left", color=INK)
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")

    for ax, pid in zip(axes[1:], (1, 2)):
        for arm in ("N", "T"):
            diag = read(NT / arm / "diagnostics.jsonl")
            for key, label, linestyle in BINS:
                xs, ys = [], []
                for d in diag:
                    b = next(x for x in d["bins"] if x["pid"] == pid and x["visit_bin"] == key)
                    xs.append(d["iteration"])
                    ys.append(b["tv_reach"])
                ax.plot(xs, ys, color=COLORS[arm], linestyle=linestyle, linewidth=1.8,
                        label=f"{arm}, {label}")
        ax.set(xscale="log", yscale="log", xlabel="CFR+ iteration (log)",
               ylabel="Reach-weighted TV: network policy vs its table's", ylim=(5e-4, 0.6))
        ax.set_title(f"Player {pid}: how far the network plays from its true regrets", loc="left", color=INK)
        ax.legend(frameon=False, fontsize=8, ncol=2, loc="lower left")
    for ax in axes:
        style(ax)
    fig.suptitle("N/T: bootstrapped versus teacher-forced regret networks (K=4,096, seed 17, GPU)", color=INK)
    fig.savefig(OUT, dpi=150)
    print(OUT)


if __name__ == "__main__":
    main()
