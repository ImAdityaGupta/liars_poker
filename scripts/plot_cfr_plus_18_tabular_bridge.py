#!/usr/bin/env python3
"""Archive and plot exact evaluations of the six-arm 18-claim tabular bridge."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ARMS = (
    ("exact", "0  Exact", "#242933"),
    ("sample_reach", "1a  Sample reach", "#087eac"),
    ("ignore_reach", "1b  Ignore reach", "#d48612"),
    ("sample_value", "2  Sample value", "#9d4b91"),
    ("sample_both", "3  Sample both", "#238257"),
    ("conditional", "4  Conditional", "#bd433f"),
)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path,
                   default=Path("artifacts/cfr_plus_18_tabular_bridge/main_20260929"))
    p.add_argument("--data-file", type=Path,
                   help="Use the curated evaluation JSONL instead of per-arm run files")
    p.add_argument("--figure-dir", type=Path, default=Path("docs/figures"))
    p.add_argument("--data-dir", type=Path, default=Path("docs/data"))
    args = p.parse_args()
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    args.data_dir.mkdir(parents=True, exist_ok=True)

    source_rows = read_jsonl(args.data_file) if args.data_file else None
    all_rows = []
    by_arm = {}
    for name, _label, _color in ARMS:
        rows = ([r for r in source_rows if r["arm"] == name] if source_rows is not None
                else read_jsonl(args.root / name / "evaluations.jsonl"))
        if len(rows) != 40 or {r["kind"] for r in rows} != {"average", "current"}:
            raise ValueError(f"Expected 20 evaluations of each policy for {name}; got {len(rows)} rows")
        by_arm[name] = rows
        all_rows.extend(rows)
    data_path = args.data_dir / "tabular_18_claim_bridge_20260929.jsonl"
    with data_path.open("w", encoding="utf-8") as out:
        for row in all_rows:
            out.write(json.dumps(row, sort_keys=True) + "\n")

    for kind in ("average", "current"):
        fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), layout="constrained")
        for name, label, color in ARMS:
            rows = sorted((r for r in by_arm[name] if r["kind"] == kind),
                          key=lambda r: r["measured_training_min"])
            y = [r["exploitability"] for r in rows]
            for ax, xkey in zip(axes, ("measured_training_min", "iteration")):
                ax.plot([r[xkey] for r in rows], y, label=label, color=color,
                        marker="o", markersize=3.5, linewidth=1.8)
        for ax, xlabel in zip(axes, ("Measured training minutes", "CFR+ iteration")):
            ax.set_yscale("log")
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Exact exploitability (lower is better)")
            ax.grid(True, which="both", alpha=0.22)
        axes[0].set_title("Equal training time")
        axes[1].set_title("Equal CFR+ iteration")
        axes[0].set_xlim(0, 310)
        axes[1].legend(fontsize=8, loc="best")
        fig.suptitle(f"18-claim tabular bridge — {kind} policy", fontsize=14)
        path = args.figure_dir / f"experiment_cfr_plus_18_tabular_bridge_{kind}.png"
        fig.savefig(path, dpi=170)
        plt.close(fig)
        print(path)
    print(data_path)


if __name__ == "__main__":
    main()
