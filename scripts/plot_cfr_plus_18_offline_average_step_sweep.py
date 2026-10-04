#!/usr/bin/env python3
"""Plot the exact exploitability refit sweeps for three 18-claim checkpoints."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs" / "data" / "cfr_plus_18_offline_average_step_sweep"
OUT = ROOT / "docs" / "figures" / "experiment_cfr_plus_18_offline_average_step_sweep.png"
STAGES = (("30 min", "0030m"), ("45 min", "0045m"), ("120 min", "0120m"))


def load_rows(stage: str) -> list[dict]:
    path = DATA / f"step_sweep_{stage}" / "results.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def main() -> None:
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), sharey=True,
                             layout="constrained")
    colors = {"warm": "#2563EB", "fresh": "#A855F7"}
    rows_by_stage = {}
    for ax, (stage_label, stage) in zip(axes, STAGES):
        rows = load_rows(stage)
        rows_by_stage[stage] = rows
        by_name = {row["name"]: row for row in rows}
        for baseline, color, linestyle in (
            ("exact", "#374151", "--"), ("online", "#D97706", ":")):
            value = by_name[baseline]["exploitability"]
            ax.axhline(value, color=color, linestyle=linestyle, linewidth=1.6,
                       label=f"{baseline}: {value:.4g}")
        for variant in ("warm", "fresh"):
            selected = sorted((r for r in rows if r.get("variant") == variant),
                              key=lambda r: r["refit_steps"])
            ax.plot([r["refit_steps"] for r in selected],
                    [r["exploitability"] for r in selected],
                    color=colors[variant], marker="o", linewidth=1.8,
                    markersize=4.2, label=f"{variant} refit")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks([6, 24, 96, 384, 1536, 5000],
                      ["6", "24", "96", "384", "1,536", "5,000"])
        ax.set_title(f"{stage_label} checkpoint · iter {by_name['exact']['iteration']:,}")
        ax.set_xlabel("Total offline fit steps per player")
        ax.grid(True, which="both", alpha=0.2)
        ax.legend(fontsize=7.7, frameon=False, loc="best")
    axes[0].set_ylabel("Exact exploitability (lower is better)")
    fig.suptitle("Offline average-policy fitting on frozen strategy reservoirs", fontsize=14)
    fig.savefig(OUT, dpi=180)
    print(OUT)
    for _, stage in STAGES:
        rows = rows_by_stage[stage]
        by_name = {row["name"]: row for row in rows}
        print(f"{stage}: exact={by_name['exact']['exploitability']:.8f} "
              f"online={by_name['online']['exploitability']:.8f}")
        for variant in ("warm", "fresh"):
            selected = [r for r in rows if r.get("variant") == variant]
            best = min(selected, key=lambda r: r["exploitability"])
            print(f"  {variant}: best={best['exploitability']:.8f} "
                  f"at {best['refit_steps']} steps; "
                  f"5k={next(r['exploitability'] for r in selected if r['refit_steps'] == 5000):.8f}")


if __name__ == "__main__":
    main()
