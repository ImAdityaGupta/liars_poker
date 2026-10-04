#!/usr/bin/env python3
"""Plot exact vs online-neural averages from preserved exact4096 checkpoints."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs" / "data"
OUT = ROOT / "docs" / "figures" / "experiment_cfr_plus_18_exact_vs_online_average.png"


def read_pair(path: Path) -> tuple[int, float, float]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    by_name = {row["name"]: row for row in rows}
    exact, online = by_name["exact"], by_name["online"]
    if exact["iteration"] != online["iteration"]:
        raise ValueError(f"Mismatched iterations in {path}")
    return int(exact["iteration"]), float(exact["exploitability"]), float(online["exploitability"])


def main() -> None:
    stages = [(30, DATA / "cfr_plus_18_offline_average_fitting_0030m.jsonl"),
              (45, DATA / "cfr_plus_18_offline_average_fitting_0045m.jsonl"),
              (120, DATA / "cfr_plus_18_offline_average_fitting_0120m" / "results.jsonl")]
    points = [(minute, *read_pair(path)) for minute, path in stages]
    minutes = [point[0] for point in points]
    exact = [point[2] for point in points]
    online = [point[3] for point in points]

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(7.6, 4.8), layout="constrained")
    ax.plot(minutes, exact, color="#2563EB", marker="o", linewidth=2.2,
            markersize=7, label="Exact tabular average")
    ax.plot(minutes, online, color="#D97706", marker="o", linewidth=2.2,
            markersize=7, label="Online neural average")
    for minute, iteration, e, _ in points:
        ax.annotate(f"iter {iteration:,}", (minute, e),
                    xytext=(0, 10), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8, color="#4B5563")
    ax.set(title="Exact and online neural averages on the exact4096 trajectory",
           xlabel="CFR+ measured training minutes",
           ylabel="Exact exploitability (lower is better)")
    ax.set_xticks(minutes, [f"{m} min" for m in minutes])
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.22)
    ax.legend(frameon=False)
    fig.savefig(OUT, dpi=180)
    print(OUT)
    for minute, iteration, e, n in points:
        print(f"{minute}m iter={iteration}: exact={e:.8f}, online_neural={n:.8f}")


if __name__ == "__main__":
    main()
