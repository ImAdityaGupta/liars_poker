#!/usr/bin/env python3
"""Recreate figures for the CPU CFR+ experiment notes from saved JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_frozen(frozen: dict, output: Path) -> None:
    cases = frozen["cases"]
    labels = ["Old full", "Old cap 2", "Streamed full", "Streamed cap 2"]
    keys = ["old_full", "old_2", "streamed_full", "streamed_2"]
    exact = [cases[key]["mean_exact_positive_regret"] for key in keys]
    sampled = [cases[key]["mean_clipped_sample_regret"] for key in keys]
    x = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - 0.18, exact, width=0.36, label="Clip exact mean regret")
    ax.bar(x + 0.18, sampled, width=0.36, label="Mean clipped sampled regret")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Mean positive regret per root action")
    ax.set_title("Sample noise changes the clipped target")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)


def plot_tabular(rows: list[dict], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    styles = [
        ("cap=full clip=after", "Full, aggregate then clip", "C0"),
        ("cap=full clip=before", "Full, clip each sample", "C1"),
        ("cap=2 clip=after", "Cap 2, aggregate then clip", "C2"),
        ("cap=2 clip=before", "Cap 2, clip each sample", "C3"),
    ]
    exact = [r for r in rows if r["method"] == "exact dense CFR+"]
    ax.plot(
        [r["iteration"] for r in exact],
        [r["exploitability"] for r in exact],
        color="black", marker="o", label="Exact dense CFR+",
    )
    for prefix, label, color in styles:
        seeds = sorted({r["method"].split("seed=")[-1] for r in rows if r["method"].startswith(prefix)})
        by_seed = {}
        for seed in seeds:
            by_seed[seed] = {
                r["iteration"]: r["exploitability"]
                for r in rows if r["method"] == f"{prefix} seed={seed}"
            }
        iterations = sorted(set.intersection(*(set(v) for v in by_seed.values())))
        data = np.array([[by_seed[s][it] for it in iterations] for s in seeds])
        mean = data.mean(axis=0)
        ax.plot(iterations, mean, marker="o", color=color, label=label)
        if len(seeds) > 1:
            ax.fill_between(iterations, data.min(axis=0), data.max(axis=0), color=color, alpha=0.12)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Exact exploitability (log scale; lower is better)")
    ax.set_yscale("log")
    ax.set_title("Clip order changes learning with the same sampler")
    ax.grid(which="both", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)


def plot_shadow(rows: list[dict], output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    caps = sorted({str(r["cap"]) for r in rows})
    for cap, color in zip(caps, ["C0", "C1"]):
        seeds = sorted({r["seed"] for r in rows if str(r["cap"]) == cap})
        for field, ax, linestyle, name in [
            ("shadow_average_exploitability", axes[0], "-", "Exact played average"),
            ("learned_average_exploitability", axes[0], "--", "Learned average"),
            ("current_exploitability", axes[1], "-", "Neural current"),
            ("root_shadow_policy_tv", axes[2], "-", "Root policy TV"),
        ]:
            by_seed = {
                seed: {r["iteration"]: r[field] for r in rows if str(r["cap"]) == cap and r["seed"] == seed}
                for seed in seeds
            }
            iterations = sorted(set.intersection(*(set(v) for v in by_seed.values())))
            data = np.array([[by_seed[s][it] for it in iterations] for s in seeds])
            cap_label = "Cap 2" if cap == "2" else "Full"
            ax.plot(iterations, data.mean(axis=0), color=color, linestyle=linestyle, marker="o", label=f"{cap_label}: {name}")
            if len(seeds) > 1:
                ax.fill_between(iterations, data.min(axis=0), data.max(axis=0), color=color, alpha=0.08)
    axes[0].set_title("Average-policy gap")
    axes[0].set_ylabel("Exact exploitability (log scale; lower is better)")
    axes[1].set_title("Current neural policy")
    axes[1].set_ylabel("Exact exploitability (log scale; lower is better)")
    axes[2].set_title("Neural vs exact regret matching")
    axes[2].set_ylabel("Root total variation")
    axes[0].set_yscale("log")
    axes[1].set_yscale("log")
    for ax in axes:
        ax.set_xlabel("Neural CFR+ iteration")
        ax.grid(which="both", alpha=0.25)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)


def plot_root_audit(rows: list[dict], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for cap, color in [("full", "C0"), ("2", "C1")]:
        seeds = sorted({r["seed"] for r in rows if str(r["cap"]) == cap})
        for field, ax, name in [
            ("root_target_mean_signed_error", axes[0], "signed target error"),
            ("root_target_mean_abs_error", axes[1], "absolute target error"),
        ]:
            by_seed = {
                seed: {
                    r["iteration"]: r["iteration"] * r[field]
                    for r in rows if str(r["cap"]) == cap and r["seed"] == seed
                }
                for seed in seeds
            }
            iterations = sorted(set.intersection(*(set(v) for v in by_seed.values())))
            data = np.array([[by_seed[s][it] for it in iterations] for s in seeds])
            ax.plot(iterations, data.mean(axis=0), marker="o", color=color, label=f"{cap}: {name}")
            if len(seeds) > 1:
                ax.fill_between(iterations, data.min(axis=0), data.max(axis=0), color=color, alpha=0.12)
    axes[0].set_title("Mean sample target minus exact target")
    axes[1].set_title("Mean absolute target error")
    for ax in axes:
        ax.set_xlabel("Neural CFR+ iteration")
        ax.set_ylabel("Error × iteration")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)


def plot_neural_clip_order(rows: list[dict], output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    panels = [
        ("shadow_average_exploitability", axes[0, 0], "Exact average of played policies"),
        ("learned_average_exploitability", axes[0, 1], "Learned average policy"),
        ("current_exploitability", axes[1, 0], "Current regret-matching policy"),
        ("root_target_mean_abs_error", axes[1, 1], "Root target error × iteration"),
    ]
    styles = [
        ("full", "clip_each_record", "C0", "--", "Full: clip each"),
        ("full", "aggregate_then_clip", "C0", "-", "Full: aggregate first"),
        ("2", "clip_each_record", "C1", "--", "Cap 2: clip each"),
        ("2", "aggregate_then_clip", "C1", "-", "Cap 2: aggregate first"),
    ]
    for cap, mode, color, linestyle, label in styles:
        selected = [
            r for r in rows
            if str(r["cap"]) == cap and r.get("clip_mode") == mode
        ]
        if not selected:
            continue
        seeds = sorted({r["seed"] for r in selected})
        for field, ax, _ in panels:
            by_seed = {
                seed: {
                    r["iteration"]: r["iteration"] * r[field]
                    if field == "root_target_mean_abs_error" else r[field]
                    for r in selected if r["seed"] == seed
                }
                for seed in seeds
            }
            iterations = sorted(set.intersection(*(set(v) for v in by_seed.values())))
            values = np.array([[by_seed[seed][it] for it in iterations] for seed in seeds])
            ax.plot(iterations, values.mean(axis=0), color=color, linestyle=linestyle,
                    marker="o", markersize=3, label=label)
            if len(seeds) > 1:
                ax.fill_between(iterations, values.min(axis=0), values.max(axis=0),
                                color=color, alpha=0.07)
    for field, ax, title in panels:
        ax.set_title(title)
        ax.set_xlabel("Neural CFR+ iteration")
        ax.set_ylabel("Error × iteration" if field == "root_target_mean_abs_error"
                      else "Exact exploitability (log scale)")
        if field != "root_target_mean_abs_error":
            ax.set_yscale("log")
        ax.grid(which="both", alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)


def plot_depth_target_audit(rows: list[dict], output: Path) -> None:
    iterations = sorted({r["iteration"] for r in rows})
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for iteration, ax in zip(iterations, axes.ravel()):
        for mode, color, name in [
            ("clip_each_record", "C0", "Clip each"),
            ("aggregate_then_clip", "C1", "Aggregate first"),
        ]:
            selected = [r for r in rows if r["iteration"] == iteration and r["mode"] == mode]
            depths = sorted({r["depth"] for r in selected})
            for field, linestyle, suffix in [
                ("target_abs_error_x_t", "-", "target"),
                ("fitted_abs_error_x_t", "--", "fitted net"),
            ]:
                means = [np.mean([r[field] for r in selected if r["depth"] == d]) for d in depths]
                ax.plot(depths, means, color=color, linestyle=linestyle,
                        marker="o", label=f"{name}: {suffix}")
        ax.set_title(f"Iteration {iteration}")
        ax.set_xlabel("Public claims already made")
        ax.set_ylabel("Mean absolute error × iteration")
        ax.set_xticks(sorted({r["depth"] for r in rows}))
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("docs/data"))
    parser.add_argument("--output-dir", type=Path, default=Path("docs/figures"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frozen = json.loads((args.results_dir / "frozen_uniform_root_4096.json").read_text(encoding="utf-8"))
    tabular = json.loads((args.results_dir / "tabular_conditional_regression_200.json").read_text(encoding="utf-8"))
    plot_frozen(frozen, args.output_dir / "experiment_cfr_plus_clipping_gap.png")
    plot_tabular(tabular, args.output_dir / "experiment_cfr_plus_tabular_clip_order.png")
    shadow_path = args.results_dir / "shadow_neural_300.json"
    if shadow_path.exists():
        shadow = json.loads(shadow_path.read_text(encoding="utf-8"))
        plot_shadow(shadow, args.output_dir / "experiment_cfr_plus_shadow_neural.png")
    audit_path = args.results_dir / "shadow_root_target_audit_80.json"
    if audit_path.exists():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        plot_root_audit(audit, args.output_dir / "experiment_cfr_plus_root_target_audit.png")
    clip_path = args.results_dir / "neural_clip_order_300.json"
    if clip_path.exists():
        clip_rows = json.loads(clip_path.read_text(encoding="utf-8"))
        plot_neural_clip_order(clip_rows, args.output_dir / "experiment_cfr_plus_neural_clip_order.png")
    long_clip_path = args.results_dir / "neural_clip_order_long_800.json"
    if long_clip_path.exists():
        long_rows = json.loads(long_clip_path.read_text(encoding="utf-8"))
        plot_neural_clip_order(long_rows, args.output_dir / "experiment_cfr_plus_neural_clip_order_long.png")
    depth_path = args.results_dir / "neural_depth_audit_300.json"
    if depth_path.exists():
        depth_rows = json.loads(depth_path.read_text(encoding="utf-8"))
        plot_depth_target_audit(depth_rows, args.output_dir / "experiment_cfr_plus_neural_depth_audit.png")
    print("Wrote figures to", args.output_dir)


if __name__ == "__main__":
    main()
