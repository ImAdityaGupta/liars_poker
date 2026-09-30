#!/usr/bin/env python3
"""Plot the completed O/E/S/N continuation and its tabular-regret fork."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs" / "data"


def rows(name: str) -> list[dict]:
    return [json.loads(line) for line in (DATA / name).read_text(encoding="utf-8").splitlines()
            if line.strip()]


normal = rows("cfr_plus_18_oens_monitors_20260930.jsonl")
fork = rows("cfr_plus_18_tabular_regret_fork_evaluations_20260930.jsonl")
fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout="constrained")
minutes = [row["training_min"] for row in normal]
axes[0].plot(minutes, [row["average"]["exploitability"] for row in normal],
             marker="o", markersize=3.5, color="#0072b2", label="Neural average (normalized)")
axes[0].plot(minutes, [row["current"]["exploitability"] for row in normal],
             marker=".", markersize=3, alpha=.6, color="#86909c", label="Neural current")
axes[0].plot([300 + row["measured_fork_min"] for row in fork],
             [row["exploitability"] for row in fork], marker="s", markersize=3.5,
             color="#d55e00", label="Tabular-regret fork from 300m")
axes[0].axvline(300, color="#aab0b9", linestyle="--", linewidth=1)
axes[0].set_yscale("log")
axes[0].set_ylabel("Exact exploitability")
axes[0].set_xlabel("Total-equivalent training minutes")
axes[0].set_title("Policy quality")
axes[0].legend(fontsize=8)

for key, label, color in (
    ("old_vs_exact_g", "Old to exact target", "#0072b2"),
    ("exact_g_vs_sampled", "Exact to sampled", "#009e73"),
    ("sampled_vs_fitted", "Sampled to fitted", "#d55e00"),
    ("exact_g_vs_fitted", "Exact to fitted", "#8759a5"),
):
    axes[1].plot(minutes, [row["audit"]["visited"][key]["mean_tv"] for row in normal],
                 marker="o", markersize=3, linewidth=1.5, label=label, color=color)
axes[1].set_yscale("log")
axes[1].set_ylabel("Mean TV on visited information sets")
axes[1].set_xlabel("Neural training minutes")
axes[1].set_title("One-step audit distances")
axes[1].legend(fontsize=8)
for ax in axes:
    ax.grid(alpha=.22, which="both")
out = ROOT / "docs" / "figures" / "experiment_cfr_plus_18_oens_final.png"
fig.savefig(out, dpi=160)
print(out)
