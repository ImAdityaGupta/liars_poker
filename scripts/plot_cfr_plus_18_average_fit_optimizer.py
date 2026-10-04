#!/usr/bin/env python3
"""Plot the completed 18-claim offline average fitting experiment."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs/data/cfr_plus_18_average_fit_optimizer"
BASE = ROOT / "docs/data/cfr_plus_18_offline_average_step_sweep"
FIGURES = ROOT / "docs/figures"
STAGES = (("30 min · iter 908", "0030m"),
          ("45 min · iter 1,424", "0045m"),
          ("120 min · iter 3,988", "0120m"))
ARMS = {
    "O0": "O0_lr1e-3_b1024_ce",
    "O1": "O1_cosine_b1024_ce",
    "O2": "O2_lr1e-4_b1024_ce",
    "O3": "O3_lr1e-3_b16384_ce",
    "O4": "O4_cosine_b16384_ce",
    "M1": "M1_O4_cosine_b16384_ce_prob_mse",
    "M2": "M2_O0_lr1e-3_b1024_prob_mse",
    "fresh": "FRESH_O4_cosine_b16384_ce",
}
COLORS = {"O0": "#D55E00", "O1": "#009E73", "O2": "#CC79A7",
          "O3": "#7C3AED", "O4": "#0072B2"}
LINE_LABELS = {
    "O0": "O0: constant 1e-3, batch 1,024",
    "O1": "O1: cosine, batch 1,024",
    "O2": "O2: constant 1e-4, batch 1,024",
    "O3": "O3: constant 1e-3, batch 16,384",
    "O4": "O4: cosine, batch 16,384",
}


def rows(stage: str, arm: str) -> list[dict]:
    path = DATA / stage / ARMS[arm] / "results.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def baseline(stage: str) -> dict[str, dict]:
    path = BASE / f"step_sweep_{stage}" / "results.jsonl"
    data = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    return {r["variant"]: r for r in data if r["variant"] in {"exact", "online"}}


def setup_axes(title: str, *, ylim=None):
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.5), sharey=True)
    for ax, (label, stage) in zip(axes, STAGES):
        b = baseline(stage)
        ax.axhline(b["exact"]["exploitability"], color="#374151", ls="--", lw=1.6,
                   label="Exact accumulated average")
        ax.axhline(b["online"]["exploitability"], color="#B9770E", ls=":", lw=1.7,
                   label="Saved online neural average")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(label)
        ax.set_xlabel("Refit steps per player")
        ax.grid(True, which="both", alpha=0.17)
        if ylim:
            ax.set_ylim(*ylim)
    axes[0].set_ylabel("Exact exploitability (lower is better)")
    fig.suptitle(title, fontsize=14)
    return fig, axes


def save(fig, axes, filename: str, *, columns: int) -> None:
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=columns, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, 0.01))
    fig.tight_layout(rect=(0, 0.16, 1, 0.94))
    path = FIGURES / filename
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(path)


def stage1() -> None:
    fig, axes = setup_axes("Optimizer recipes on frozen average-policy data", ylim=(0.0024, 0.05))
    for ax, (_, stage) in zip(axes, STAGES):
        for arm in ("O0", "O1", "O2", "O3", "O4"):
            data = sorted((r for r in rows(stage, arm) if r.get("variant") == "warm"
                           and r["refit_steps"] >= 192), key=lambda r: r["refit_steps"])
            ax.plot([r["refit_steps"] for r in data],
                    [r["exploitability"] for r in data],
                    color=COLORS[arm], marker="o", markersize=4, lw=1.7,
                    label=LINE_LABELS[arm])
        ax.set_xlim(180, 5500)
        ax.set_xticks([250, 500, 1000, 2000, 5000],
                      ["250", "500", "1k", "2k", "5k"])
    save(fig, axes, "experiment_cfr_plus_18_average_fit_optimizer_stage1.png", columns=4)


def objective() -> None:
    fig, axes = setup_axes("Loss and fresh-start comparison", ylim=(0.0024, 0.05))
    choices = (
        ("O0", "O0: CE, batch 1,024", COLORS["O0"], "-"),
        ("M2", "O0 optimizer + probability MSE", COLORS["O0"], "--"),
        ("O4", "O4: CE, batch 16,384", COLORS["O4"], "-"),
        ("M1", "O4 optimizer + probability MSE", COLORS["O4"], "--"),
        ("fresh", "Fresh O4: CE, 20k steps", COLORS["O1"], "-."),
    )
    for ax, (_, stage) in zip(axes, STAGES):
        for arm, label, color, ls in choices:
            data = sorted((r for r in rows(stage, arm) if r["refit_steps"] >= 250),
                          key=lambda r: r["refit_steps"])
            ax.plot([r["refit_steps"] for r in data],
                    [r["exploitability"] for r in data], color=color,
                    marker="o" if arm != "fresh" else "D", markersize=4,
                    ls=ls, lw=1.7, label=label)
        ax.set_xlim(220, 22000)
        ax.set_xticks([250, 1000, 5000, 20000], ["250", "1k", "5k", "20k"])
    save(fig, axes, "experiment_cfr_plus_18_average_fit_optimizer_objective.png", columns=4)


def seeds() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 5.1), sharey=True)
    for ax, (label, stage) in zip(axes, STAGES):
        b = baseline(stage)
        ax.axhline(b["exact"]["exploitability"], color="#374151", ls="--", lw=1.5,
                   label="Exact accumulated average")
        ax.axhline(b["online"]["exploitability"], color="#B9770E", ls=":", lw=1.7,
                   label="Saved online neural average")
        for x, arm in enumerate(("O0", "O4")):
            arm_names = (ARMS[arm], ARMS[arm] + "_seed17032",
                         ARMS[arm] + "_seed17033")
            for offset, name in zip((-0.1, 0, 0.1), arm_names):
                if name == ARMS[arm]:
                    data = rows(stage, arm)
                else:
                    path = DATA / stage / name / "results.jsonl"
                    data = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                            if line.strip()]
                point = next(r for r in data if r["refit_steps"] == 5000)
                ax.scatter(x + offset, point["exploitability"], color=COLORS[arm],
                           marker="o" if offset == -0.1 else "D", s=55,
                           edgecolor="white", linewidth=0.6, zorder=4,
                           label=("O0, constant" if arm == "O0" else "O4, cosine + large batch")
                           if offset == -0.1 else None)
        ax.set_xticks([0, 1], ["O0", "O4"])
        ax.set_xlim(-0.35, 1.35)
        ax.set_yscale("log")
        ax.set_ylim(0.0024, 0.055)
        ax.set_title(label)
        ax.grid(True, axis="y", which="both", alpha=0.18)
        ax.set_xlabel("5,000-step warm refit · three fit seeds")
    axes[0].set_ylabel("Exact exploitability (lower is better)")
    fig.suptitle("Minibatch-seed variation from the same source trajectory", fontsize=14)
    save(fig, axes, "experiment_cfr_plus_18_average_fit_optimizer_seeds.png", columns=4)


def main() -> None:
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False})
    FIGURES.mkdir(parents=True, exist_ok=True)
    stage1()
    objective()
    seeds()


if __name__ == "__main__":
    main()
