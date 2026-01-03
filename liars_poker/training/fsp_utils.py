from __future__ import annotations

import json
import numpy as np
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Tuple

from liars_poker.core import GameSpec, ARTIFACTS_ROOT
from liars_poker.serialization import save_policy, load_policy


def basic_eta_control(episodes: int) -> float:
    return 1 / (episodes + 2)


def plot_exploitability_series(logs: Dict | Iterable[Dict], *, figsize: tuple[int, int] = (12, 6)):
    """Plot exploitability over iterations using the logs dict returned by fsp_loop/dense_fsp_loop."""
    import matplotlib.pyplot as plt

    series_list = logs if isinstance(logs, Iterable) and not isinstance(logs, Dict) else [logs]

    fig, ax = plt.subplots(figsize=figsize)
    for idx, entry in enumerate(series_list, start=1):
        series = entry.get("exploitability_series", [])
        if not series:
            continue
        vals = [
            (2*pt.get("predicted_avg", pt.get("rollout_avg", 0.0)))-1
            for pt in series
        ]
        ax.plot(range(1, len(vals) + 1), vals, marker="o", label=f"run {idx}")

    ax.set_xlabel("Episode")
    ax.set_ylabel("Exploitability")
    ax.set_title("Exploitability over FSP iterations")
    ax.grid(True, alpha=0.3)
    if ax.lines:
        ax.legend()
    fig.tight_layout()
    plt.xscale('log')
    plt.yscale('log')
    return ax

def plot_with_trend(info, *, last_n=None, targets=(1e-2, 2e-3)):
    series = info.get("exploitability_series", [])
    if not series:
        print("No exploitability data.")
        return None

    # Use predicted_avg when available, else rollout_avg
    y_vals = []
    for pt in series:
        y = pt.get("predicted_avg")
        if y is None:
            y = pt.get("rollout_avg", 0.0)
        y_vals.append(2*float(y)-1)

    n = len(y_vals)
    x_vals = np.arange(1, n + 1, dtype=float)

    if last_n is None:
        last_n = max(5, n - 50)
    last_n = min(max(2, int(last_n)), n)

    x_fit = x_vals[-last_n:]
    y_fit = np.array(y_vals[-last_n:], dtype=float)
    mask = y_fit > 0
    if mask.sum() < 2:
        print("Not enough positive points to fit a trendline.")
        plot_exploitability_series(info)
        return None

    logx = np.log10(x_fit[mask])
    logy = np.log10(y_fit[mask])
    slope, intercept = np.polyfit(logx, logy, 1)

    ax = plot_exploitability_series(info)
    if ax is None:
        return None
    y_pred = 10 ** (intercept + slope * np.log10(x_vals))
    ax.plot(x_vals, y_pred, linestyle='--', color='black', alpha=0.7, label='trend (power law)')
    ax.legend()

    print(f"Fit (log10): y = 10^( {intercept:.4f} + {slope:.4f} * log10(x) )")

    for target in targets:
        if target <= 0:
            continue
        # Solve for x: log10(y) = intercept + slope * log10(x)
        # log10(x) = (log10(target) - intercept) / slope
        if slope == 0:
            print(f"Target {target:.4g}: slope is 0, no crossing predicted.")
            continue
        logx_star = (math.log10(target) - intercept) / slope
        x_star = 10 ** logx_star
        print(f"Target {target:.4g}: predicted at step ~{x_star:.1f}")

    return ax


def save_fsp_run(run_id: str, policy, info: dict, spec: GameSpec, root: Path | None = None) -> None:
    """Save a policy and its FSP run metrics to a run directory."""
    root = root or Path(ARTIFACTS_ROOT)
    run_dir = root / "benchmark_runs" / run_id
    (run_dir / "policy").mkdir(parents=True, exist_ok=True)

    # Save policy
    save_policy(policy, run_dir / "policy")

    # Save metrics
    metrics = {
        "spec": spec.to_json(),
        "exploitability_series": info.get("exploitability_series", []),
        "p_values": info.get("p_values", []),
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Saved run to {run_dir}")


def dense_fsp_resume(
    run_dir: str | Path,
    *,
    remaining_episodes: int,
    eta_control=basic_eta_control,
    episodes_test: int = 10_000,
    efficient: bool = False,
    debug: bool = False,
) -> Tuple[object, Dict]:
    """Resume a dense FSP run from a saved directory (loose resume)."""
    run_path = Path(run_dir)
    policy_dir = run_path / "policy"
    policy, spec = load_policy(str(policy_dir))

    metrics_path = run_path / "metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    else:
        metrics = {}

    base_series = list(metrics.get("exploitability_series", []) or [])
    base_p_values = list(metrics.get("p_values", []) or [])

    if base_series and base_p_values:
        i0 = min(len(base_series), len(base_p_values))
    else:
        i0 = len(base_series)

    base_series = base_series[:i0]
    base_p_values = base_p_values[:i0]

    def eta_resume(i: int) -> float:
        return eta_control(i + i0)

    from liars_poker.training.dense_fsp import dense_fsp_loop

    new_pol, new_info = dense_fsp_loop(
        spec=spec,
        episodes=remaining_episodes,
        initial_pol=policy,
        eta_control=eta_resume,
        episodes_test=episodes_test,
        efficient=efficient,
        debug=debug,
    )

    merged = {
        "exploitability_series": base_series + list(new_info.get("exploitability_series", []) or []),
        "p_values": base_p_values + list(new_info.get("p_values", []) or []),
    }
    if "br_meta" in new_info:
        merged["br_meta"] = new_info["br_meta"]

    return new_pol, merged
