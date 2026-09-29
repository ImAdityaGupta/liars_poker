#!/usr/bin/env python3
"""Plot the completed low-root 18-claim tabular bridge against K=1024 context."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs/data/low_root_tabular_bridge_20260929"
OLD = ROOT / "docs/data/tabular_18_claim_bridge_20260929.jsonl"
OUT = ROOT / "docs/figures/experiment_cfr_plus_18_tabular_bridge_low_roots_average.png"

COLORS = {128: "#0072B2", 256: "#D55E00", 512: "#009E73", 1024: "#555555"}
ARMS = (("sample_both", "Arm 3: sample reach + value", "o", "-"),
        ("conditional", "Arm 4: conditional value", "s", "--"))


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def plot_kind(kind: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), layout="constrained")
    for k in (128, 256, 512):
        for arm, arm_label, marker, linestyle in ARMS:
            rows = [r for r in read(DATA / f"k{k:04d}_{arm}.jsonl")
                    if r["kind"] == kind]
            rows.sort(key=lambda r: r["measured_training_min"])
            label = f"K={k} · {arm_label}"
            for ax, xkey in zip(axes, ("measured_training_min", "iteration")):
                ax.plot([r[xkey] for r in rows], [r["exploitability"] for r in rows],
                        label=label, color=COLORS[k], marker=marker, markersize=3.5,
                        linewidth=1.8, linestyle=linestyle)

    old_rows = read(OLD)
    for arm, arm_label, marker, linestyle in ARMS:
        rows = [r for r in old_rows if r["arm"] == arm and r["kind"] == kind]
        rows.sort(key=lambda r: r["measured_training_min"])
        label = f"K=1024 · {arm_label} (earlier run)"
        for ax, xkey in zip(axes, ("measured_training_min", "iteration")):
            ax.plot([r[xkey] for r in rows], [r["exploitability"] for r in rows],
                    label=label, color=COLORS[1024], marker=marker, markersize=3,
                    linewidth=1.5, linestyle=linestyle, alpha=0.85)

    for ax, xlabel in zip(axes, ("Measured training minutes", "CFR+ iteration")):
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(f"Exact {kind}-policy exploitability (lower is better)")
        ax.grid(True, which="both", alpha=0.22)
    axes[0].set_title("Equal training time")
    axes[1].set_title("Equal iteration count")
    axes[0].set_xlim(0, 305)
    axes[1].legend(fontsize=7.2, ncol=2, loc="upper right")
    fig.suptitle(f"18-claim tabular bridge: root-count extension — {kind} policy", fontsize=14)
    path = OUT.with_name(OUT.name.replace("_average", f"_{kind}"))
    fig.savefig(path, dpi=180)
    print(path)


def main() -> None:
    for kind in ("average", "current"):
        plot_kind(kind)


if __name__ == "__main__":
    main()
