#!/usr/bin/env python3
"""Plot the exact-average tabular discount rerun and its reference curves."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs" / "data" / "cfr_plus_18_tabular_discount_exact_average_20261002"
OUT = ROOT / "docs" / "figures" / "experiment_cfr_plus_18_tabular_discount_exact_average.png"

ARMS = {
    "V_cfr_uniform": ("V · CFR, uniform average", "#7C3AED"),
    "B_cfr_plus_quadratic": ("B · CFR+, quadratic average", "#D55E00"),
    "C_dcfr_plus_quadratic": ("C · DCFR+, quadratic average", "#009E73"),
    "D_dcfr_exact_quadratic": ("D · DCFR, exact decay", "#CC79A7"),
    "E_dcfr_visited_quadratic": ("E · DCFR, visited decay", "#E69F00"),
}


def read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue  # Ignore an incomplete final JSONL line during an active run.
        if row.get("exploitability", 0) > 0:
            rows.append(row)
    return rows


def points(rows: list[dict], kind: str | None = None) -> list[dict]:
    rows = [r for r in rows if kind is None or r.get("policy_kind") == kind]
    return sorted(rows, key=lambda r: (r["iteration"], r.get("measured_training_min", 0)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    root = args.data_root

    curves: list[tuple[str, str, list[dict], str, str]] = []
    for arm, (label, color) in ARMS.items():
        rows = points(read(root / arm / "evaluations.jsonl"))
        if rows:
            curves.append((label, color, rows, "-", "o"))

    references = [
        ("A · K=4096 exact-average control", "#111827", root / "references" / "exact4096.jsonl", None, "--", None),
        ("Neural K=1024 · O4 average", "#E11D48", root / "references" / "neural_o4_k1024.jsonl", "o4", ":", "s"),
        ("Neural K=4096 · O4 average", "#0D9488", root / "references" / "neural_o4_k4096.jsonl", "o4", ":", "D"),
    ]
    for label, color, path, kind, linestyle, marker in references:
        rows = points(read(path), kind)
        # Keep the comparison focused on the new runs' 9-hour training window.
        rows = [r for r in rows if r.get("measured_training_min", 0) <= 540.1]
        if rows:
            curves.append((label, color, rows, linestyle, marker))

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.4), layout="constrained")
    for label, color, rows, linestyle, marker in curves:
        for ax, xkey in zip(axes, ("measured_training_min", "iteration")):
            ax.plot([r[xkey] for r in rows], [r["exploitability"] for r in rows],
                    color=color, linestyle=linestyle, marker=marker,
                    markersize=3.3 if marker else 0, linewidth=2 if linestyle == "-" else 1.7,
                    label=label)

    for ax, title, xlabel in zip(
        axes,
        ("Equal measured training time", "Equal CFR iteration"),
        ("Measured training minutes", "CFR iteration"),
    ):
        ax.set(title=title, xlabel=xlabel, ylabel="Exact exploitability", yscale="log")
        ax.grid(True, which="both", color="#e4e8ee", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[1].set_xlim(left=0)
    fig.suptitle("18-claim sampled regret rules with exact tabular averages", fontsize=15)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3,
               frameon=False, fontsize=8.5)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    print(args.output)


if __name__ == "__main__":
    main()
