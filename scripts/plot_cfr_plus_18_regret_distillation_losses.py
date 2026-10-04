#!/usr/bin/env python3
"""Plot recorded minibatch losses from the 18-claim regret-table fits."""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIT_LINE = re.compile(
    r"^\[fit\] (?P<source>\d{4}m)/(?P<arm>R-visit|R-mix) "
    r"p(?P<player>[12]) (?P<step>\d+)/40000 loss=(?P<loss>[\d.eE+-]+)$"
)
SOURCES = ("0030m", "0045m", "0120m", "1080m")
ARMS = ("R-visit", "R-mix")
COLORS = {"R-visit": "#0072b2", "R-mix": "#d55e00"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for source in SOURCES:
        for line in (args.logs / f"{source}.log").read_text(encoding="utf-8").splitlines():
            match = FIT_LINE.match(line)
            if match:
                rows.append({
                    "source": source,
                    "arm": match["arm"],
                    "player": int(match["player"]),
                    "step": int(match["step"]),
                    "loss": float(match["loss"]),
                })

    fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharex=True, layout="constrained")
    for col, source in enumerate(SOURCES):
        for row_index, player in enumerate((1, 2)):
            ax = axes[row_index, col]
            for arm in ARMS:
                data = sorted(
                    (r for r in rows if r["source"] == source and r["arm"] == arm and r["player"] == player),
                    key=lambda r: r["step"],
                )
                ax.plot(
                    [r["step"] for r in data],
                    [r["loss"] for r in data],
                    label=arm,
                    color=COLORS[arm],
                    linewidth=1.8,
                )
            ax.set_yscale("log")
            ax.grid(alpha=0.25)
            ax.set_title(f"{int(source[:-1])} min source · player {player}")
            if col == 0:
                ax.set_ylabel("Minibatch masked MSE")
            if row_index == 1:
                ax.set_xlabel("Optimizer steps for this player")
            if row_index == 0 and col == 0:
                ax.legend(frameon=False)
    fig.suptitle("Regret-table distillation: training minibatch loss (log scale)", fontsize=16)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160)
    plt.close(fig)

    csv_path = args.logs / "loss_points.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("source", "arm", "player", "step", "loss"))
        writer.writeheader()
        writer.writerows(rows)
    for source in SOURCES:
        parts = []
        for arm in ARMS:
            for player in (1, 2):
                final = max(
                    (r for r in rows if r["source"] == source and r["arm"] == arm and r["player"] == player),
                    key=lambda r: r["step"],
                )
                parts.append(f"{arm} p{player}={final['loss']:.5g}")
        print(source, ", ".join(parts))
    print(args.out)


if __name__ == "__main__":
    main()
